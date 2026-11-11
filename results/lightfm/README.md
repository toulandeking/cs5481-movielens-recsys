# results/lightfm — 交付物清单

由一条命令生成（**不需要手动拼），可复现：

```bash
python -m src.adapters.lightfm --export
```

超参默认读 `outputs/metrics/lightfm_params.json`（定稿配置：`c=32 / warp / epochs=20 / lr=0.05 / max_sampled=100 / identity=true`）。
本次生成：`seed=42`、`lightfm 1.16`、单线程（Windows 构建无 OpenMP）、fit **171.7 s**。

## 文件

| 文件 | 行数 | 说明 |
|---|---|---|
| `top10.parquet` | 60,400 | **部署口径**：排除该用户 train ∪ val ∪ test 的全部影片 —— 给演示界面用 |
| `top10_eval.parquet` | 60,400 | **评估口径**：只排除 train ∪ val，**保留 test 目标** —— 供交叉验证复算 Recall@10 |
| `params.json` | — | 超参 + 特征维度 + 训练耗时 + lightfm 版本 + 轮 4 的选配置依据（`selection` / `tuning`） |
| `lightfm_tuning_sensitivity.png` | — | 20 组 val 网格：侧信息-only vs 加身份列 |
| `lightfm_popularity_bias.png` | — | 按目标热度分档的命中率（val vs test） |

## 列

两份 parquet 列完全一致，`6040 × 10` 行、按 `(uid, rank)` 升序排列：

| 列 | 类型 | 含义 |
|---|---|---|
| `uid` | int32 | 用户 id，0–6039 |
| `rank` | int8 | 1–10，每用户恰好 10 行 |
| `iid` | int32 | 物品 id，0–3415 |
| `score` | float32 | 模型分数（warp 下绝对值无意义，只用于排序；被遮掉的位置是 `-inf`，实际不会出现） |

## 独立复算 Recall@10（不碰模型代码）

```python
import numpy as np, pandas as pd, json
from src.evaluate import load_ground_truth
from src.contract import SPLIT_TEST

df = pd.read_parquet("results/lightfm/top10_eval.parquet").sort_values(["uid", "rank"])
gt = load_ground_truth(SPLIT_TEST)
t = gt["targets"]                                  # 每用户唯一的 test 目标 iid
hit = df["iid"].to_numpy() == t[df["uid"].to_numpy()]
rank = np.zeros(6040, dtype=int); rank[df["uid"].to_numpy()[hit]] = df["rank"].to_numpy()[hit]
print("Recall@10 =", (rank > 0).mean())            # 0.077483
print("NDCG@10   =", np.where(rank > 0, 1/np.log2(rank+1), 0).mean())   # 0.039772
```

应与 `outputs/metrics/lightfm_test.json` 的 `reports.full` 完全一致（已实测：**0.077483 / 0.039772，468/6040 命中**）。

## 两条口径为什么不同

`top10.parquet` 连 test 目标也排除（作业要求"推荐用户没看过的片"），所以它**不能**拿来算 Recall；
`top10_eval.parquet` 保留 test 目标，才能复现 `evaluate.py` 的 full 口径（`seen = train ∪ val`）。
`evaluate.py` 里 `S[seen] = -inf` 遮的正是 train ∪ val，两份文件的差别只在这一列上。

> 注：`top10*.parquet` 用稳定排序（同分下标小者在前）取 top-10，与 `evaluate.py` 的名次定义逐位一致；
> `outputs/top10/lightfm.parquet`（`--eval` 生成的那份）走的是 `base.top_k_indices` 的 argpartition，
> 当第 10/11 名分数**完全相等**时两者可能差一行。本目录的两份文件以评估口径为准。
