"""
cs5481 MovieLens-1M recommender
==============================

src/adapters/itemcf.py  —  课内算法：Item-based Collaborative Filtering
-----------------------------------------------------------------------
本项目三个算法中的第 ① 个（课内算法），也是整条链路的第一个验证对象。

算法定义
~~~~~~~~
构造二值交互矩阵 R ∈ {0,1}^{|U|×|I|}（依据 contract.RELEVANCE_RULE = "all_observed"：
所有观测到的交互均为正样本，`rating` 数值不参与）。

**第 1 步：用户活跃度衰减**
    R̃[u, i] = R[u, i] / n_u^α        n_u = 用户 u 的历史长度，α 默认 0.5

**第 2 步：物品-物品余弦相似度**
                    R̃^T R̃
    sim(i, j) = ─────────────────
                ‖r̃_i‖ · ‖r̃_j‖

**第 3 步：top-k 邻居截断 + 打分**
    score(u, i) = Σ_{j ∈ top-k(i) ∩ I_u}  sim(i, j)

为什么必须加第 1 步（实测结论）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
在 ML-1M 上，原始余弦相似度矩阵有 **92.02% 的条目非零** —— 3416 个物品两两之间
几乎都至少有一个共同用户。此时"对用户全部历史求和"会退化成**变相的流行度打分**，
表现为 val Recall@10 只有 0.060，仅比爆款基线（0.035）高一点。

根因是用户活跃度极度不均：平均 163 条、中位 94 条、最长 **2275** 条。
长历史用户在求和里贡献的项数压倒性多于短历史用户，且它们与热门片的相似度普遍偏高。

加 `1/n_u^α` 衰减后（α = 0.5，按 sqrt 衰减是协同过滤里常用的折中）：
每个用户的总贡献被拉平，相似度衡量的是"**共同偏好**"而不是"**共同数量**"。

    val Recall@10:  0.0599 (α=0, k=100)  →  0.0786 (α=0.5, k=20)   **+31%**

由此带来的普适教训（可写进报告）：**在稠密相似度矩阵上，邻域方法需要显式的活跃度归一化，
否则会向 popularity baseline 退化。**

超参敏感性（val）
~~~~~~~~~~~~~~~~~
- `α`：0 → 0.060；0.25 → 0.075；**0.5 → 0.079**；0.75 → 0.078；1.0 → 0.076
- `k_neighbors`：在 α=0.5 下，k ∈ [20, all] 的 val Recall 都在 0.074–0.079，
  差异在统计噪声内（6040 用户的 SE ≈ 0.0035）。用 **1-SE 规则**（取最优值 1 个标准误
  以内、且 k 最小的配置）与直接取 argmax 得到同一个点：**α=0.5, k=20**，故设为默认值。
- `shrinkage` λ：几乎无收益（λ=10 时 0.0710 vs 0.0707），默认关闭。
  推测原因是 α 衰减已经承担了"惩罚虚假共现"的角色。

不在评分时排除已交互物品
~~~~~~~~~~~~~~~~~~~~~~~~
`score_matrix()` 会把用户看过的片也打分。这是**故意的**：排除逻辑统一由
`src/evaluate.py` 负责（contract.EXCLUDE_TRAIN_ITEMS）。模型与评估职责分离，
避免两边规则不一致或重复排除。

用法
~~~~
    python -m src.adapters.itemcf --eval              # 训练 + test 评估 + 导出 top10
    python -m src.adapters.itemcf --tune              # 在 val 上搜索 α 与 k（不碰 test）
    python -m src.adapters.itemcf --top --uid 42      # 看某个用户的前 10 推荐
"""

from __future__ import annotations

import json
import sys
import time
import numpy as np
import pandas as pd
from scipy import sparse

from src.adapters.base import AdapterBase, build_binary_matrix
from src.contract import (
    K, N_ITEMS, N_USERS, OUTPUT_DIR, SEED, SPLIT_TEST, SPLIT_TRAIN, SPLIT_VAL,
    read_canon_table,
)
from src.evaluate import evaluate_all, load_ground_truth, print_reports, save_report

__all__ = [
    "ItemCFAdapter", "weight_users", "cosine_similarity", "truncate_topk",
]


# =============================================================================
# 纯函数：Item-CF 的数学部分（便于单测与复用）
# =============================================================================


def weight_users(R: sparse.csr_matrix, alpha: float) -> sparse.csr_matrix:
    """用户活跃度衰减：R̃[u,i] = R[u,i] / n_u^α。α=0 时原样返回。"""
    if alpha <= 0:
        return R.copy()
    n_u = np.asarray(R.getnnz(axis=1)).ravel()
    w = 1.0 / np.power(np.maximum(n_u, 1), alpha)
    return (sparse.diags(w.astype(np.float32)) @ R).tocsr()


def cosine_similarity(R: sparse.csr_matrix, shrinkage: float = 0.0) -> np.ndarray:
    """物品-物品余弦相似度，返回稠密 (N_ITEMS, N_ITEMS) float32，对角线为 0。

    `R` 可以是二值矩阵，也可以是 `weight_users()` 加权后的矩阵。
    只做**一次**稀疏矩阵乘 `R^T R` 得到分子，范数由列平方和直接算出。
    """
    counts = (R.T @ R).tocsr()
    counts.setdiag(0)
    counts.eliminate_zeros()

    # 列 L2 范数（加权后 sum 与 sum-of-squares 不再相等，必须用平方和）
    norm = np.sqrt(np.maximum(
        np.asarray(R.multiply(R).sum(axis=0)).ravel(), 1e-12
    ))

    dense = counts.toarray().astype(np.float32)
    sim = dense / np.outer(norm, norm)

    if shrinkage > 0:
        # 显著性加权 sim *= cooc / (cooc + λ)，仅作用于非零共现
        sim *= dense / (dense + shrinkage)

    np.fill_diagonal(sim, 0.0)
    np.clip(sim, 0.0, None, out=sim)     # 加权后本应非负，防浮点误差
    return sim


def truncate_topk(S: np.ndarray, k: int | None) -> np.ndarray:
    """每个物品只保留相似度最高的 k 个邻居，其余置 0。

    并列时按 iid 升序（stable argsort）—— 跨机器、跨 numpy 版本可复现。
    """
    if k is None or k >= S.shape[1]:
        return S
    order = np.argsort(-S, axis=1, kind="stable")[:, :k]
    out = np.zeros_like(S)
    rows = np.arange(S.shape[0])[:, None]
    out[rows, order] = S[rows, order]
    return out


# =============================================================================
# Item-CF
# =============================================================================
class ItemCFAdapter(AdapterBase):
    """Item-based CF。

    参数
    ----
    k_neighbors        : 每个物品保留的邻居数（None = 全保留；默认 20，见文档"超参敏感性"）
    shrinkage          : 显著性加权 λ（0 = 关闭；实测收益很小）
    user_weight_alpha  : 用户活跃度衰减指数（0 = 原始余弦；默认 0.5）
    """

    name = "ItemCF"

    def __init__(
        self,
        k_neighbors: int | None = 20,
        shrinkage: float = 0.0,
        user_weight_alpha: float = 0.5,
    ):
        self.k_neighbors = k_neighbors
        self.shrinkage = shrinkage
        self.user_weight_alpha = user_weight_alpha
        self.R: sparse.csr_matrix | None = None       # 二值历史（打分用）
        self.sim: np.ndarray | None = None            # 未截断的相似度
        self._scores: np.ndarray | None = None

    # -- 训练 ----------------------------------------------------------------
    def fit(self) -> None:
        t0 = time.perf_counter()
        self.R = build_binary_matrix(SPLIT_TRAIN)
        Rw = weight_users(self.R, self.user_weight_alpha)
        self.sim = cosine_similarity(Rw, shrinkage=self.shrinkage)
        self._scores = None

        nnz = int((self.sim > 0).sum())
        print(
            f"  相似度 {self.sim.shape}  非零 {nnz:,} "
            f"({100 * nnz / self.sim.size:.2f}% 稠密)  "
            f"α={self.user_weight_alpha}  λ={self.shrinkage}  "
            f"用时 {time.perf_counter() - t0:.1f}s"
        )

    # -- 打分 ----------------------------------------------------------------
    def score_matrix(self) -> np.ndarray:
        if self.R is None or self.sim is None:
            raise RuntimeError("请先调用 fit()")
        if self._scores is None:
            t0 = time.perf_counter()
            S_k = truncate_topk(self.sim, self.k_neighbors)
            # 稀疏(用户历史) × 稠密(top-k 相似度) ≈ 3.4 GFLOP，秒级
            self._scores = np.asarray(self.R @ S_k, dtype=np.float32)
            nnz = int((self._scores > 0).sum())
            print(
                f"  打分 {self._scores.shape}  非零 {nnz:,} "
                f"({100 * nnz / self._scores.size:.2f}%)  k={self.k_neighbors}  "
                f"用时 {time.perf_counter() - t0:.1f}s"
            )
        return self._scores

    def predict_scores(self, uid: np.ndarray, iid: np.ndarray) -> np.ndarray:
        return self.score_matrix()[uid, iid]


# =============================================================================
# CLI
# =============================================================================
def cmd_eval(model: ItemCFAdapter) -> int:
    print(f"\n[训练] k={model.k_neighbors}  α={model.user_weight_alpha}  λ={model.shrinkage}")
    model.fit()
    S = model.score_matrix()

    reports = evaluate_all(S, model=model.name)
    print_reports(reports, f"Item-CF 评估（test，K={K}）")

    path = save_report(reports, "itemcf_test")
    print(f"\n  指标 → {path.relative_to(OUTPUT_DIR.parent)}")

    df = model.export_top10()
    (OUTPUT_DIR / "top10").mkdir(parents=True, exist_ok=True)
    tp = OUTPUT_DIR / "top10" / "itemcf.parquet"
    df.to_parquet(tp, index=False)
    print(f"  top-10 → {tp.relative_to(OUTPUT_DIR.parent)}  ({len(df):,} 行)")

    payload = {
        "model": model.name, "seed": SEED,
        "params": {
            "k_neighbors": model.k_neighbors,
            "user_weight_alpha": model.user_weight_alpha,
            "shrinkage": model.shrinkage,
        },
        "n_users": N_USERS, "n_items": N_ITEMS,
    }
    (OUTPUT_DIR / "metrics" / "itemcf_params.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


def cmd_tune() -> int:
    """在 **val** 上搜索 (α, k)。绝不能看 test。"""
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    ks = [20, 50, 100, None]
    print(f"\n调参：α ∈ {alphas}   k ∈ {[str(x) for x in ks]}")
    print("只用 val；相似度按 α 逐个重算（每个 α 只算一次）\n")

    R = build_binary_matrix(SPLIT_TRAIN)
    gt_val = load_ground_truth(SPLIT_VAL)
    from src.contract import EVAL_MODE_FULL
    from src.evaluate import evaluate_matrix

    rows = []
    for alpha in alphas:
        t0 = time.perf_counter()
        sim = cosine_similarity(weight_users(R, alpha))
        for k in ks:
            S = np.asarray(R @ truncate_topk(sim, k), dtype=np.float32)
            rep = evaluate_matrix(S, EVAL_MODE_FULL, split=SPLIT_VAL, gt=gt_val)
            rows.append({
                "alpha": alpha, "k_neighbors": "all" if k is None else k,
                "val_recall@10": round(rep.recall, 4),
                "val_ndcg@10": round(rep.ndcg, 4),
            })
            print(f"  α={alpha:<5} k={str(k):>4}  val Recall@10={rep.recall:.4f}"
                  f"  NDCG@10={rep.ndcg:.4f}")
        print(f"        （α={alpha} 相似度重算 {time.perf_counter() - t0:.1f}s）")

    tbl = pd.DataFrame(rows)
    out = OUTPUT_DIR / "metrics"
    out.mkdir(parents=True, exist_ok=True)
    p = out / "itemcf_tuning_val.csv"
    tbl.to_csv(p, index=False)
    print(f"\n{tbl.to_string(index=False)}")

    best = tbl.loc[tbl["val_recall@10"].idxmax()]
    print(f"\n  val 最优：α={best['alpha']}  k={best['k_neighbors']}  "
          f"Recall@10={best['val_recall@10']}")
    print(f"  表格 → {p.relative_to(OUTPUT_DIR.parent)}")
    print("  选定配置后才允许在 test 上跑一次 --eval，之后不得回头再调。")
    return 0


def cmd_top(model: ItemCFAdapter, uid: int) -> int:
    print(f"\n[训练] k={model.k_neighbors}  α={model.user_weight_alpha}")
    model.fit()
    S = model.score_matrix()

    movies = read_canon_table("movies")
    title = dict(zip(movies["iid"], movies["title"]))
    genre = dict(zip(movies["iid"], movies["genres"]))
    ratings = read_canon_table("ratings")

    hist = ratings[ratings["uid"] == uid].sort_values("ts")
    print(f"\n用户 uid={uid} 历史评分 {len(hist)} 条，最近 5 条：")
    for _, row in hist.tail(5).iterrows():
        print(f"    {title[int(row['iid'])]:<45.45} {row['rating']:.0f}★  [{row['split']}]")

    # 评估口径（排除 train ∪ val）下，test 目标排第几
    gt = load_ground_truth(SPLIT_TEST)
    masked = S.copy()
    masked[gt["seen"]] = -np.inf
    tgt = int(gt["targets"][uid])
    rank = int((masked[uid] > masked[uid, tgt]).sum()) + 1
    flag = "✅ 命中" if rank <= K else "未命中"
    print(f"\n评估口径（排除 train∪val）：test 目标《{title[tgt]}》排在第 {rank} 位  {flag}")

    top = model.export_top10().query("uid == @uid")
    print(f"\n部署口径 top-{len(top)}（已排除该用户评过的全部影片）：")
    for _, row in top.iterrows():
        iid = int(row["iid"])
        print(f"  {int(row['rank']):>2}. {title[iid]:<45.45} "
              f"score={row['score']:.4f}  [{', '.join(genre[iid])}]")
    return 0


_UNSET = object()


# =============================================================================
# 结果展示：可复现的表格 + 图
# =============================================================================
BUCKET_EDGES = [0, 5, 20, 100, 500, 1500, np.inf]
BUCKET_LABELS = ["<5", "5-19", "20-99", "100-499", "500-1499", ">=1500"]


def _hit_by_bucket(S: np.ndarray, targets: np.ndarray, seen: np.ndarray,
                   pop: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """在给定排除集合下，按目标物品的训练期热度分档统计命中率。"""
    M = S.copy()
    M[seen] = -np.inf
    tgt_score = M[np.arange(N_USERS), targets]
    rank = (M > tgt_score[:, None]).sum(axis=1) + 1
    hit = rank <= K
    bucket = np.digitize(pop[targets], BUCKET_EDGES) - 1
    rates = np.array([
        hit[bucket == b].mean() if (bucket == b).any() else np.nan
        for b in range(len(BUCKET_LABELS))
    ])
    counts = np.array([int((bucket == b).sum()) for b in range(len(BUCKET_LABELS))])
    return rates, counts


def cmd_analyze(model: ItemCFAdapter) -> int:
    """产出报告用的两张图 + 一张错误分析表。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = OUTPUT_DIR / "figures"
    met_dir = OUTPUT_DIR / "metrics"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[训练] k={model.k_neighbors}  α={model.user_weight_alpha}")
    model.fit()
    S = model.score_matrix()

    r = read_canon_table("ratings")
    R = build_binary_matrix(SPLIT_TRAIN)
    pop = np.asarray(R.sum(axis=0)).ravel()

    def targets_of(split):
        t = r[r["split"] == split]
        arr = np.full(N_USERS, -1, np.int32)
        arr[t["uid"].to_numpy()] = t["iid"].to_numpy()
        return arr

    seen_train = np.zeros((N_USERS, N_ITEMS), dtype=bool)
    tr = r[r["split"] == SPLIT_TRAIN]
    seen_train[tr["uid"].to_numpy(), tr["iid"].to_numpy()] = True
    from src.evaluate import load_ground_truth
    seen_test = load_ground_truth(SPLIT_TEST)["seen"]

    v_rate, v_cnt = _hit_by_bucket(S, targets_of(SPLIT_VAL), seen_train, pop)
    t_rate, t_cnt = _hit_by_bucket(S, targets_of(SPLIT_TEST), seen_test, pop)

    tbl = pd.DataFrame({
        "target_popularity": BUCKET_LABELS,
        "val_n": v_cnt, "val_hit@10": np.round(v_rate, 4),
        "test_n": t_cnt, "test_hit@10": np.round(t_rate, 4),
    })
    csv = met_dir / "itemcf_popularity_buckets.csv"
    tbl.to_csv(csv, index=False)
    print("\n错误分析：按目标物品热度分档的命中率")
    print(tbl.to_string(index=False))
    print(f"  表格 → {csv.relative_to(OUTPUT_DIR.parent)}")

    # ---- 图 1：α × k 的 val Recall@10 ----
    tun = met_dir / "itemcf_tuning_val.csv"
    if tun.is_file():
        d = pd.read_csv(tun)
        ks = ["20", "50", "100", "all"]
        fig, ax = plt.subplots(figsize=(7, 4.2))
        for alpha in sorted(d["alpha"].unique()):
            sub = d[d["alpha"] == alpha].copy()
            sub["k"] = sub["k_neighbors"].astype(str)
            sub = sub.set_index("k")
            y = [sub["val_recall@10"].get(k, np.nan) for k in ks]
            ax.plot(ks, y, marker="o", label=f"alpha={alpha}")
        ax.axhline(0.0353, ls="--", c="grey", lw=1, label="popularity baseline")
        ax.set_xlabel("k_neighbors")
        ax.set_ylabel("val Recall@10")
        ax.set_title("Item-CF: sensitivity to k and user-activity damping")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        p1 = fig_dir / "itemcf_alpha_k_val.png"
        fig.savefig(p1, dpi=150)
        plt.close(fig)
        print(f"  图 1 → {p1.relative_to(OUTPUT_DIR.parent)}")

    # ---- 图 2：热度分档命中率（val vs test） ----
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(len(BUCKET_LABELS))
    w = 0.38
    ax.bar(x - w / 2, v_rate, w, label="val targets")
    ax.bar(x + w / 2, t_rate, w, label="test targets")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b}\n(n={c})" for b, c in zip(BUCKET_LABELS, t_cnt)], fontsize=8)
    ax.set_xlabel("target popularity (# train ratings)")
    ax.set_ylabel("hit rate @10")
    ax.set_title("Item-CF hits concentrate on popular targets (popularity bias)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    p2 = fig_dir / "itemcf_popularity_bias.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    print(f"  图 2 → {p2.relative_to(OUTPUT_DIR.parent)}")
    return 0


def main(argv: list[str]) -> int:
    """超参默认值**只定义在 ItemCFAdapter 的构造函数里**，此处不重复。

    命令行给了就覆盖，没给就用类默认（α=0.5, k=20，即 val 上选出的配置）。
    """
    def opt(flag, cast=str):
        if flag in argv:
            return cast(argv[argv.index(flag) + 1])
        return _UNSET

    if "--tune" in argv:
        return cmd_tune()
    if "--eval" not in argv and "--top" not in argv and "--analyze" not in argv:
        print(__doc__)
        print("用法：--eval | --tune | --analyze | --top --uid N   "
              "[--k N] [--alpha A] [--shrinkage L]")
        return 2

    kwargs: dict = {}
    for flag, cast, key in (
        ("--k", lambda v: None if v.lower() in ("all", "none") else int(v), "k_neighbors"),
        ("--alpha", float, "user_weight_alpha"),
        ("--shrinkage", float, "shrinkage"),
    ):
        val = opt(flag, cast)
        if val is not _UNSET:
            kwargs[key] = val
    model = ItemCFAdapter(**kwargs)

    if "--top" in argv:
        uid = opt("--uid", int)
        return cmd_top(model, 0 if uid is _UNSET else uid)
    if "--analyze" in argv:
        return cmd_analyze(model)
    return cmd_eval(model)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
