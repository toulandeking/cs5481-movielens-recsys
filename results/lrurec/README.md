# LRURec：完整全新 20 组实验

本目录为 2026-11-09 完成的实验摘要。20 套配置均重新初始化、重新训练，旧 10 组不计入；`checkpoint_reused=False`。完整方法、结论和复现步骤见 [LRUREC.md](../../LRUREC.md)。

| 文件 | 内容 |
|---|---|
| [top10.parquet](top10.parquet) | 演示版：60,400 行，排除 train∪val∪test 全部已评分影片 |
| [top10_eval.parquet](top10_eval.parquet) | full 评估版：60,400 行，仅排除 train∪val，保留 test 目标 |
| [top10_verification.json](top10_verification.json) | 两份推荐的格式、SHA-256 和交叉验证结果 |
| [lrurec_tuning_val.csv](lrurec_tuning_val.csv) | 全部 20 组验证集调参记录 |
| [lrurec_params.json](lrurec_params.json) | 按 val 选出的第 20 组配置 |
| [lrurec_test.json](lrurec_test.json) | full / candidates 指标 |
| [lrurec_popularity_buckets.csv](lrurec_popularity_buckets.csv) | 热度分档错误分析 |
| [epochs/](epochs/) | 20 组逐轮训练记录 |
| [verification.txt](verification.txt) | 4 项算法测试与 3 项新实验验收 |
| [environment.txt](environment.txt) | 本次实际依赖版本 |
| [lrurec_fresh20_manifest.json](lrurec_fresh20_manifest.json) | 预先固定配置、数据指纹、原实验保护哈希 |
| [lrurec_fresh20_complete.json](lrurec_fresh20_complete.json) | 完成证明：20 组、旧记录计入数为 0 |

只在本次 20 组选参结束后计算一次 test；同一个 test 此前在旧实验中已查看，本次属于重复实验，不是新的未见测试集验证。原始数据、完整分数和模型由本地命令生成；本目录额外提供两份推荐文件供演示接入与指标核对。

## 推荐文件的生成与验证

两份文件均来自全新 20 组实验已经保存的原始分数矩阵，不重新训练、不重新选参。模型输入上下文仍为 train 历史。分别应用两种排除规则后，按分数降序、同分 iid 升序取前 10 名。

每份文件仅含 `uid, rank, iid, score` 四列：`uid` 为 canon 内部编号 0–6039，`iid` 为 canon 内部编号 0–3415，`rank` 为 1–10，`score` 为模型推荐分数。每个用户恰好 10 条且无重复影片，已检查没有命中相应排除集合，并完成 Parquet 写入后的回读验证。

评估版命中 1,372 / 6,040 个 test 目标，full Recall@10 = **0.2271523178807947**，NDCG@10 = **0.12328319157758873**；从导出文件计算的结果与 `src/evaluate.py` 及冻结实验报告一致（浮点容差 1e-12）。这是 full 全量排序口径，不能据此重建 101 candidates 排序结果。演示版排除了 test 目标，不能用于计算上述指标。

本地复现导出（先按项目说明生成数据和实验分数）：

```powershell
python -m src.export_recommendations --model lrurec --scores outputs/lrurec_fresh20_cuda_20261109/scores/lrurec_test.npy --reference outputs/lrurec_fresh20_cuda_20261109/metrics/lrurec_test.json
```

演示页面读取 `pd.read_parquet("results/lrurec/top10.parquet")`；独立重算指标时读取 `top10_eval.parquet`，将各用户推荐的 iid 与 canon 中同一 uid 的 test 目标匹配。Recall@10 = 命中用户数 / 6040；NDCG@10 = 全部用户的命中贡献均值，命中时贡献为 `1/log2(rank+1)`，未命中为 0。
