"""
cs5481 MovieLens-1M recommender
==============================

src/evaluate.py  —  指标的**唯一**出口
-------------------------------------
任何模型的效果都必须经由本文件计算。禁止在 adapter / notebook 里另写一套
Recall@K。否则三个人口径不同，「三算法对比」这张表就失效了。

两种评估模式（来自 DATA.md §5.3 / FRAMEWORK.md §1.3）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    full        全量排序：对全部 3,416 个物品打分后排序            ← 主表
    candidates  101 候选：1 正 + 100 固定种子负样本内排序          ← 副表

**两种模式必须来自同一个分数矩阵**（先铺满 (6040, 3416)，再切片）。
本文件的 `evaluate_all()` 就是这个保证：一次矩阵，两张表。

用法
~~~~
    # 自检：随机打分，验证指标实现是否正确
    python -m src.evaluate --random

    # 自检：最流行基线，验证"已交互物品排除"是否生效
    python -m src.evaluate --popularity

    # 在代码里
    from src.evaluate import evaluate_all, score_matrix_from_adapter
    S = adapter.score_matrix()          # (N_USERS, N_ITEMS) float32
    reports = evaluate_all(S)           # {"full": ..., "candidates": ...}

指标定义
~~~~~~~~
Leave-One-Out 下每个用户只有 **1 个** 正样本，因此：

    Recall@K = 命中用户数 / 用户总数
    HR@K     = 有命中用户数 / 用户总数        ← 与 Recall@K 恒等
    NDCG@K   = mean( 1 / log2(rank + 1) )    rank 为命中时的名次，未命中记 0

Recall@K 与 HR@K 恒等是 LOO 的固有性质（不是 bug），报告里说明即可。
本文件仍分别计算两者：代码按"多正样本"的一般形式写，将来扩展不用改。
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.contract import (
    CANDIDATES_PER_USER, EVAL_MODE_CANDIDATES, EVAL_MODE_FULL, EVAL_MODES,
    EXCLUDE_TRAIN_ITEMS, K, METRICS, N_ITEMS, N_USERS, OUTPUT_DIR, SEED,
    SPLIT_TEST, SPLIT_TRAIN, SPLIT_VAL, SPLITS, read_canon_table,
)

__all__ = [
    "EvalReport", "load_ground_truth", "evaluate_matrix", "evaluate_all",
    "score_matrix_from_adapter", "save_report", "print_reports",
]


# =============================================================================
# 结果容器
# =============================================================================
@dataclass
class EvalReport:
    """一次评估的结果。字段名与 METRICS 对齐，外加若干诊断量。"""

    model: str
    mode: str
    split: str
    k: int
    n_users: int
    n_candidates_mean: float          # 平均候选数量（full 模式会随用户已交互数变化）
    recall: float
    ndcg: float
    hr: float
    # 诊断：命中时平均名次，用于判断"差一点就中"的程度
    mean_hit_rank: float = float("nan")
    # 诊断：目标分数在候选中的名次分布（分位）
    recall_at_1: float = float("nan")
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    def as_row(self) -> str:
        return (
            f"{self.model:<22} {self.mode:<11} {self.split:<5} "
            f"R@{self.k}={self.recall:.4f}  NDCG@{self.k}={self.ndcg:.4f}  "
            f"HR@{self.k}={self.hr:.4f}  (n={self.n_users:,})"
        )


# =============================================================================
# 真值加载
# =============================================================================
def load_ground_truth(split: str = SPLIT_TEST) -> dict[str, np.ndarray]:
    """从 canon 取评估所需的三样东西。

    返回
    ----
    targets : (N_USERS,) int32      每个用户的目标 iid
    seen    : (N_USERS, N_ITEMS) bool  该用户在**目标之前**已交互过的物品
    cand    : (N_USERS, 101) int32    仅 candidates 模式需要，101 候选的 iid

    "目标之前"的定义（很关键，涉及泄漏）：
        split == "test"  →  seen = train ∪ val
        split == "val"   →  seen = train          （test 是未来，绝不能用）
    """
    if split not in SPLITS:
        raise ValueError(f"未知 split {split!r}，应为 {SPLITS}")

    r = read_canon_table("ratings")

    test_part = r[r["split"] == split]
    targets = np.full(N_USERS, -1, dtype=np.int32)
    targets[test_part["uid"].to_numpy()] = test_part["iid"].to_numpy()
    if (targets < 0).any():
        raise RuntimeError(f"有用户没有 {split} 目标，canon 不完整")

    seen_splits = [SPLIT_TRAIN] if split == SPLIT_VAL else [SPLIT_TRAIN, SPLIT_VAL]
    hist = r[r["split"].isin(seen_splits)]
    seen = np.zeros((N_USERS, N_ITEMS), dtype=bool)
    seen[hist["uid"].to_numpy(), hist["iid"].to_numpy()] = True

    # 目标不能出现在 seen 里，否则评估无意义
    if seen[np.arange(N_USERS), targets].any():
        raise RuntimeError(f"有 {split} 目标出现在历史集里 —— canon 有泄漏")

    cand = _load_candidates()
    return {"targets": targets, "seen": seen, "cand": cand}


def _load_candidates() -> np.ndarray:
    """把 101 候选整理成 (N_USERS, 101) 的 iid 矩阵，保持文件内的原始顺序。"""
    c = read_canon_table("eval_candidates")
    expected_uid = np.repeat(np.arange(N_USERS, dtype=np.int32), CANDIDATES_PER_USER)
    if not np.array_equal(c["uid"].to_numpy(), expected_uid):
        raise RuntimeError("eval_candidates 不是按 (uid, 块内顺序) 排列的，无法 reshape")
    cand = c["iid"].to_numpy().reshape(N_USERS, CANDIDATES_PER_USER)
    return cand.astype(np.int32)


# =============================================================================
# 核心：给定候选内分数，算排名与指标
# =============================================================================
def _rank_metrics(scores: np.ndarray, target_pos: np.ndarray, k: int) -> dict[str, float]:
    """从"每用户一组候选分数"算出指标。

    参数
    ----
    scores     : (n_users, n_cand) float32   该用户在其候选集合上的分数
    target_pos : (n_users,) int32            目标在 axis=1 上的列下标

    并列的处理
    ----------
    名次定义为：**分数更高者在前；分数相同则列下标更小者在前**。
    这个规则完全确定，不依赖 np.argsort 的实现细节，也不受候选顺序影响
    （候选顺序已在 canon 生成时被固定）。
    实现上用计数而非排序：
        rank = #{分数严格大于目标} + #{分数等于目标且下标小于目标} + 1
    """
    n, m = scores.shape
    rows = np.arange(n)

    target_score = scores[rows, target_pos]                      # (n,)

    gt = (scores > target_score[:, None]).sum(axis=1)            # 严格大于
    eq = scores == target_score[:, None]
    eq[rows, target_pos] = False                                 # 排除目标自己
    before = (eq & (np.arange(m)[None, :] < target_pos[:, None])).sum(axis=1)

    rank = gt + before + 1                                       # 1-based
    hit = rank <= k

    n_hit = int(hit.sum())
    recall = n_hit / n                       # 每用户 1 个正样本 ⇒ 等价于命中率
    hr = n_hit / n
    ndcg = float(np.where(hit, 1.0 / np.log2(rank + 1.0), 0.0).mean())
    return {
        "recall": float(recall),
        "ndcg": float(ndcg),
        "hr": float(hr),
        "recall_at_1": float((rank <= 1).mean()),
        "mean_hit_rank": float(rank[hit].mean()) if n_hit else float("nan"),
        "n_hit": n_hit,
    }


# =============================================================================
# 两种评估模式
# =============================================================================
def evaluate_matrix(
    scores: np.ndarray,
    mode: str = EVAL_MODE_FULL,
    split: str = SPLIT_TEST,
    model: str = "model",
    k: int = K,
    gt: dict[str, np.ndarray] | None = None,
) -> EvalReport:
    """在给定的 (N_USERS, N_ITEMS) 分数矩阵上评估。

    分数越大越推荐。NaN / ±inf 会被视为"不可推荐"（排序时落到最后）。
    """
    if scores.shape != (N_USERS, N_ITEMS):
        raise ValueError(
            f"scores 形状应为 ({N_USERS}, {N_ITEMS})，实际 {scores.shape}"
        )
    if mode not in EVAL_MODES:
        raise ValueError(f"未知 mode {mode!r}，应为 {EVAL_MODES}")

    gt = gt or load_ground_truth(split)
    targets: np.ndarray = gt["targets"]
    seen: np.ndarray = gt["seen"]

    S = np.array(scores, dtype=np.float32, copy=True)
    # 统一把非有限值压到最低，避免 NaN 参与比较
    np.nan_to_num(S, copy=False, nan=-np.inf, posinf=np.inf, neginf=-np.inf)

    if mode == EVAL_MODE_FULL:
        if split != SPLIT_TEST and split != SPLIT_VAL:
            raise ValueError(f"full 模式不支持 split={split!r}")
        if EXCLUDE_TRAIN_ITEMS:
            S[seen] = -np.inf          # 该用户已交互过的不参与排序
        target_pos = targets           # 全量模式下列下标就是 iid
        n_cand_mean = float((~seen).sum(axis=1).mean())

    else:  # EVAL_MODE_CANDIDATES
        if split != SPLIT_TEST:
            # 候选集文件只为 test 生成；val 调参请用 full 模式
            raise ValueError(
                "candidates 模式只支持 split='test'（canon 的候选集按 test 生成）。"
                "调参阶段请用 mode='full'。"
            )
        cand = gt["cand"]
        rows = np.arange(N_USERS)[:, None]
        S = S[rows, cand]                                    # 切片 → (N, 101)
        match = cand == targets[:, None]
        if not match.sum(axis=1).all():
            raise RuntimeError("有用户的目标不在其 101 候选集里")
        target_pos = match.argmax(axis=1)
        n_cand_mean = float(CANDIDATES_PER_USER)

    res = _rank_metrics(S, target_pos, k)
    return EvalReport(
        model=model, mode=mode, split=split, k=k,
        n_users=N_USERS, n_candidates_mean=n_cand_mean,
        recall=res["recall"], ndcg=res["ndcg"], hr=res["hr"],
        mean_hit_rank=res["mean_hit_rank"], recall_at_1=res["recall_at_1"],
    )


def evaluate_all(
    scores: np.ndarray,
    model: str = "model",
    split: str = SPLIT_TEST,
    k: int = K,
) -> dict[str, EvalReport]:
    """一次矩阵 → 两张表（主表 full + 副表 candidates）。"""
    gt = load_ground_truth(split)          # 只加载一次，两种模式共用
    out: dict[str, EvalReport] = {}
    out[EVAL_MODE_FULL] = evaluate_matrix(
        scores, EVAL_MODE_FULL, split=split, model=model, k=k, gt=gt
    )
    if split == SPLIT_TEST:
        out[EVAL_MODE_CANDIDATES] = evaluate_matrix(
            scores, EVAL_MODE_CANDIDATES, split=split, model=model, k=k, gt=gt
        )
    return out


# =============================================================================
# 与 Adapter 的桥接
# =============================================================================
def score_matrix_from_adapter(adapter, batch: int = 256) -> np.ndarray:
    """把实现了 `predict_scores(uid, iid)` 的 adapter 铺成 (N_USERS, N_ITEMS) 矩阵。

    FRAMEWORK.md §4 规定 adapter 的评估入口是 predict_scores(uid, iid)。
    本函数按用户分批把整个物品空间查一遍，避免一次构造 2,000 万行的查询表。
    """
    S = np.empty((N_USERS, N_ITEMS), dtype=np.float32)
    all_iid = np.arange(N_ITEMS, dtype=np.int32)
    if not hasattr(adapter, "predict_scores"):
        raise TypeError("adapter 必须实现 predict_scores(uid, iid)")
    for start in range(0, N_USERS, batch):
        stop = min(start + batch, N_USERS)
        uids = np.arange(start, stop, dtype=np.int32)
        uu = np.repeat(uids, N_ITEMS)
        ii = np.tile(all_iid, stop - start)
        s = np.asarray(adapter.predict_scores(uu, ii), dtype=np.float32)
        if s.shape != uu.shape:
            raise ValueError(
                f"predict_scores 返回长度 {s.shape}，应为 {uu.shape}"
            )
        S[start:stop] = s.reshape(stop - start, N_ITEMS)
    return S


# =============================================================================
# 输出
# =============================================================================
def save_report(reports: dict[str, EvalReport], stem: str) -> Path:
    """把结果写到 outputs/metrics/<stem>.json。"""
    out_dir = OUTPUT_DIR / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.json"
    payload = {
        "contract": {
            "n_users": N_USERS, "n_items": N_ITEMS, "k": K,
            "exclude_seen": EXCLUDE_TRAIN_ITEMS, "seed": SEED,
        },
        "reports": {mode: asdict(rep) for mode, rep in reports.items()},
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def print_reports(reports: dict[str, EvalReport], title: str = "") -> None:
    if title:
        print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")
    for rep in reports.values():
        print("  " + rep.as_row())
    full = reports.get(EVAL_MODE_FULL)
    cand = reports.get(EVAL_MODE_CANDIDATES)
    if full and cand:
        ratio = cand.recall / full.recall if full.recall > 0 else float("nan")
        print(f"\n  副表(full)对照：candidates 模式 Recall@{K} 是 full 模式的 {ratio:.1f}×")
        print("  说明：候选集里只有 100 个干扰项，命中难度远低于 3,416 个物品的全量排序；")
        print("        绝对数值不可跨模式比较，两种模式都报文是为了与不同论文口径对照。")


# =============================================================================
# 自检：两个合成基线
# =============================================================================
def _selfcheck_random(k: int = K, tol: float = 0.02) -> int:
    """随机打分 → 指标应接近理论基线。这是验证指标实现的最直接手段。"""
    rng = np.random.default_rng(SEED)
    S = rng.random((N_USERS, N_ITEMS), dtype=np.float32)

    reports = evaluate_all(S, model="random-baseline", k=k)
    print_reports(reports, "自检 ①：随机打分")

    expect_full = k / N_ITEMS
    expect_cand = k / CANDIDATES_PER_USER
    ok = True
    for mode, expect in ((EVAL_MODE_FULL, expect_full), (EVAL_MODE_CANDIDATES, expect_cand)):
        got = reports[mode].recall
        diff = abs(got - expect)
        flag = "✓" if diff <= tol else "✗"
        print(f"  {flag} {mode:<11} Recall@{k} = {got:.5f}   理论 {expect:.5f}   差 {diff:.5f}")
        ok = ok and diff <= tol
    print(
        "\n  随机模型 NDCG 的理论值（未命中记 0，命中则 1/log2(r+1)）约为 "
        f"{np.mean([1/np.log2(r+1) for r in range(1, k+1)]) * expect_full:.6f}，"
        f"实测 {reports[EVAL_MODE_FULL].ndcg:.6f}"
    )
    return 0 if ok else 1


def _selfcheck_popularity(k: int = K) -> int:
    """最流行基线：给所有人推同样的"训练期被评分最多的 10 部"。

    这个检查的价值：如果"排除已交互物品"没生效，所有人都会收到同一批爆款，
    而爆款往往正是用户的 test 目标，Recall 会异常高（>0.3）。正常应在 0.05–0.15。
    """
    r = read_canon_table("ratings")
    tr = r[r["split"] == SPLIT_TRAIN]
    counts = np.bincount(tr["iid"].to_numpy(), minlength=N_ITEMS).astype(np.float32)
    S = np.tile(counts, (N_USERS, 1))

    reports = evaluate_all(S, model="popularity-baseline", k=k)
    print_reports(reports, "自检 ②：最流行基线")

    full_recall = reports[EVAL_MODE_FULL].recall
    top10 = np.argsort(-counts, kind="stable")[:k]
    print(f"\n  全站最热的 {k} 部（训练期被评分次数）：")
    for rank_, iid in enumerate(top10, 1):
        print(f"    {rank_:>2}. iid={iid:<5} {int(counts[iid]):>5} 次")

    print(f"\n  full 模式 Recall@{k} = {full_recall:.4f}")
    if full_recall > 0.20:
        print("  ✗ 异常偏高：很可能'排除已交互物品'没有生效，正在把用户看过的片重新推给他")
        return 1
    if full_recall < 0.01:
        print("  ✗ 异常偏低：检查分数矩阵是否全是常数、或目标全被 mask 掉了")
        return 1
    print("  ✓ 落在合理区间（full 模式下爆款基线典型值约 0.02–0.10），排除逻辑正常")
    print(f"  candidates 模式 {reports[EVAL_MODE_CANDIDATES].recall:.4f}：候选里的正样本本身"
          "偏向热门片，所以爆款基线在这个口径下会显得高得多 —— 这正是两模式不可跨比的原因。")
    return 0


# =============================================================================
# CLI
# =============================================================================
def main(argv: list[str]) -> int:
    t0 = time.perf_counter()
    print(f"evaluate.py · N_USERS={N_USERS} N_ITEMS={N_ITEMS} K={K} SEED={SEED}")

    if "--random" in argv:
        rc = _selfcheck_random()
    elif "--popularity" in argv:
        rc = _selfcheck_popularity()
    else:
        print(__doc__)
        print("请指定自检模式：--random 或 --popularity")
        return 2

    print(f"\n用时 {time.perf_counter() - t0:.1f}s")
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
