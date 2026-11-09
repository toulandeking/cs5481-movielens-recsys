# LRURec：完整全新 20 组实验

本目录为 2026-11-09 完成的实验摘要。20 套配置均重新初始化、重新训练，旧 10 组不计入；`checkpoint_reused=False`。完整方法、结论和复现步骤见 [LRUREC.md](../../LRUREC.md)。

| 文件 | 内容 |
|---|---|
| [lrurec_tuning_val.csv](lrurec_tuning_val.csv) | 全部 20 组验证集调参记录 |
| [lrurec_params.json](lrurec_params.json) | 按 val 选出的第 20 组配置 |
| [lrurec_test.json](lrurec_test.json) | full / candidates 指标 |
| [lrurec_popularity_buckets.csv](lrurec_popularity_buckets.csv) | 热度分档错误分析 |
| [epochs/](epochs/) | 20 组逐轮训练记录 |
| [verification.txt](verification.txt) | 4 项算法测试与 3 项新实验验收 |
| [environment.txt](environment.txt) | 本次实际依赖版本 |
| [lrurec_fresh20_manifest.json](lrurec_fresh20_manifest.json) | 预先固定配置、数据指纹、原实验保护哈希 |
| [lrurec_fresh20_complete.json](lrurec_fresh20_complete.json) | 完成证明：20 组、旧记录计入数为 0 |

只在本次 20 组选参结束后计算一次 test；同一个 test 此前在旧实验中已查看，本次属于重复实验，不是新的未见测试集验证。原始数据、完整分数、模型和全部推荐由本地命令生成，沿用仓库的数据与生成物约定。
