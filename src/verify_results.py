"""
cs5481 MovieLens-1M recommender
==============================

src/verify_results.py  —  独立复算已提交的 full-mode 指标
----------------------------------------------------------
所有模型都可以用 `results/<model>/top10_eval.parquet` 复算 full 模式的
Recall@10 / NDCG@10 —— **不需要模型权重，也不需要 lightfm / torch**。
这既是交叉验证，也能在某个模型没提交 `*_test.json` 时补上主结果（主表）。

写入 `results/<model>/metrics_recomputed.json`，结构与 `*_test.json` 的
`reports.full` 一致，但顶层带 `"recomputed": true` 与 `"source"`，
以免被误当作模型侧导出的原始文件。**只含 full 模式**：副表（101 候选）
无法从 top-10 反推。

用法
~~~~
    python -m src.verify_results            # 复算全部 + 写出 json
    python -m src.verify_results --check    # 只比对，不写文件
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.contract import (
    K, N_USERS, REPO_ROOT, SPLIT_TEST, SPLIT_TRAIN, SPLIT_VAL, read_canon_table,
)

RESULTS_DIR = REPO_ROOT / "results"

# 模型负责人未提交 *_test.json 时，副表（101 候选）数值无法从 top-10 反推，
# 只能引用其文档中的记录。写入 JSON 时带 "_source"，界面上以 * 标注，
# 以免与"独立复算"的结果混为一谈。
DOCUMENTED_CANDIDATES: dict[str, dict] = {
    "lightfm": {
        "recall": 0.6621,
        "ndcg": 0.3895,
        "hr": 0.6621,
        "_source": "documented in LIGHTFM.md §4.1 — cited, NOT recomputed",
    },
}

__all__ = ["recompute", "main"]


def _ground_truth():
    """返回 (每个用户的 test 目标 iid, train∪val 的已见掩码)。"""
    r = read_canon_table("ratings")
    uids = np.arange(N_USERS)

    test = r[r["split"] == SPLIT_TEST].set_index("uid")["iid"].reindex(uids)
    if test.isna().any():
        raise RuntimeError("canon 里有用户没有 test 目标")

    seen = np.zeros((N_USERS, read_canon_table("movies").shape[0]), dtype=bool)
    hist = r[r["split"].isin([SPLIT_TRAIN, SPLIT_VAL])]
    seen[hist["uid"].to_numpy(), hist["iid"].to_numpy()] = True
    return test.to_numpy(dtype=np.int32), seen


def recompute(model_dir: Path, targets: np.ndarray) -> dict | None:
    """从 top10_eval.parquet 复算 full 模式指标。"""
    path = model_dir / "top10_eval.parquet"
    if not path.is_file():
        return None

    df = pd.read_parquet(path)
    df = df[df["rank"] <= K].merge(
        pd.DataFrame({"uid": np.arange(N_USERS), "target": targets}), on="uid"
    )
    hit = df["iid"].to_numpy() == df["target"].to_numpy()
    rank = np.full(N_USERS, 0, dtype=np.int32)
    rank[df["uid"].to_numpy()[hit]] = df["rank"].to_numpy()[hit]

    n_hits = int((rank > 0).sum())
    recall = n_hits / N_USERS
    gains = np.zeros(N_USERS, dtype=np.float64)
    hit_mask = rank > 0
    gains[hit_mask] = 1.0 / np.log2(rank[hit_mask] + 1.0)
    ndcg = float(gains.mean())
    return {
        "n_users": N_USERS,
        "n_hits": n_hits,
        "recall": float(recall),
        "ndcg": ndcg,
        "hr": float(recall),
    }


def main(argv: list[str]) -> int:
    check_only = "--check" in argv
    targets, _ = _ground_truth()

    rows = []
    print(f"{'model':<10} {'recall@10':>10} {'ndcg@10':>9} {'order':>7}  comparison")
    print("-" * 78)

    for model_dir in sorted(p for p in RESULTS_DIR.iterdir() if p.is_dir()):
        name = model_dir.name
        got = recompute(model_dir, targets)
        if got is None:
            print(f"{name:<10} {'—':>10} {'—':>9} {'—':>7}  no top10_eval.parquet")
            continue

        # 与模型侧导出的 *_test.json 比对
        ref = model_dir / f"{name}_test.json"
        if ref.is_file():
            reported = json.loads(ref.read_text(encoding="utf-8"))["reports"]["full"]
            d_r = abs(got["recall"] - reported["recall"])
            d_n = abs(got["ndcg"] - reported["ndcg"])
            note = f"vs {ref.name}: dR={d_r:.2e} dN={d_n:.2e}"
            agree = d_r < 1e-6 and d_n < 1e-6
        else:
            note = f"no {ref.name} (owner did not commit it)"
            agree = None

        rows.append((name, got, note, agree, ref.is_file()))
        print(f"{name:<10} {got['recall']:>10.6f} {got['ndcg']:>9.6f} "
              f"{got['n_hits']:>7d}  {note}")

    if check_only:
        bad = [r for r in rows if r[3] is False]
        print(f"\n{'FAIL' if bad else 'OK'}: {len(bad)} mismatch")
        return 1 if bad else 0

    # 写回：只在模型侧没有 *_test.json 时补，避免覆盖原始交付物
    written = []
    for name, got, note, agree, has_ref in rows:
        if has_ref:
            continue
        payload = {
            "model": name,
            "recomputed": True,
            "source": (
                f"recomputed by src/verify_results.py from "
                f"results/{name}/top10_eval.parquet (no model code involved)"
            ),
            "protocol": {
                "split": SPLIT_TEST,
                "k": K,
                "mode": "full",
                "exclude": [SPLIT_TRAIN, SPLIT_VAL],
            },
            "reports": {
                "full": {
                    "model": name,
                    "mode": "full",
                    "split": SPLIT_TEST,
                    "k": K,
                    "n_users": N_USERS,
                    "recall": got["recall"],
                    "ndcg": got["ndcg"],
                    "hr": got["hr"],
                }
            },
            "note": (
                "full-mode numbers are recomputed here from the committed top-10 export. "
                "candidates-mode cannot be derived from a top-10 export; where it appears it is "
                "cited from the owner's documentation and marked with _source."
            ),
        }

        cand = DOCUMENTED_CANDIDATES.get(name)
        if cand:
            payload["reports"]["candidates"] = dict(cand)

        out = RESULTS_DIR / name / "metrics_recomputed.json"
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        written.append(out)

    print()
    for p in written:
        print(f"  wrote {p}")
    if not written:
        print("  所有模型都已有 *_test.json，未写补充文件")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
