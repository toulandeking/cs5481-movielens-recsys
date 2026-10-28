# 任务清单 

> **更新**：2026-10-28

> **配套**：[README.md](README.md) · [DATA.md](DATA.md) · [ITEMCF.md](ITEMCF.md)（模板）

---

## 0. 总览

| # | 算法 | 类别（作业要求） | 负责人 | 状态 | 截止 |
|---|---|---|---|---:|---|
| ① | **Item-based CF** | 课内讲过的算法 | **A** | ✅ **完成** | — |
| ② | **LightFM** | Reference Algorithms | **B** | ⬜ 待开始 | 11/16 |
| ③ | **LRURec** | 2024–2025 新算法 | **C** | ⬜ 待开始 | 11/16 |

> 作业原文要求「恰好三个算法」：一个课内、一个 Reference、一个 2024–2025 新算法。
> **参数变体不算不同算法**，不能重复。

---

## 1. 共同前置：已经就绪，不用重做

| 已就绪 | 位置 | 说明 |
|---|---|---|
| 冻结数据集 | `data/canon/` | 5 个 parquet，6.0 MB；**不要重新切分** |
| 配置常量 | `src/contract.py` | `N_USERS=6040`、`N_ITEMS=3416`、`SEED=42`、`K=10` |
| 评估出口 | `src/evaluate.py` | full / candidates 双模式；**禁止自己写 Recall@10** |
| 适配器基类 | `src/adapters/base.py` | 继承 `AdapterBase`，实现两个方法即可 |
| 模板文档 | `ITEMCF.md` | 照这个结构写自己那份 |
| 自检基线 | `python -m src.evaluate --random / --popularity` | 指标实现的回归检查 |

---

## 2. 每个人都要遵守的四条铁律

1. **数据只从 `read_canon_table("ratings")` 读**，不要自己解析 `.dat`、不要自己切分。
2. **特征/统计量只能从 `split == "train"` 算**。用了 val/test 就是泄漏。
3. **超参只能在 `val` 上选**，`test` 只允许跑一次（选定配置之后）。
4. **指标只认 `evaluate.py`**。不要在 adapter 或 notebook 里另写一套。

---

## 3. ① Item-CF（A）— ✅ 已完成

作为**参照标准**，列在这里是为了让 B/C 知道"完成"长什么样。

| 交付物 | 状态 |
|---|---|
| `src/adapters/itemcf.py` | ✅ |
| `outputs/metrics/itemcf_test.json`（两模式指标） | ✅ |
| `outputs/metrics/itemcf_tuning_val.csv`（20 组调参） | ✅ |
| `outputs/metrics/itemcf_params.json`（最终超参） | ✅ |
| `outputs/metrics/itemcf_popularity_buckets.csv`（错误分析） | ✅ |
| `outputs/figures/itemcf_*.png`（2 张图） | ✅ |
| `outputs/top10/itemcf.parquet`（演示数据） | ✅ |
| `ITEMCF.md` | ✅ |

**test 结果**：full Recall@10 = **0.0641**，NDCG@10 = 0.0335；candidates 0.5560 / 0.3224。

---

## 4. ② LightFM（B）

### 4.1 任务清单

**环境（W2 上半周）**
- [ ] `python -m pip install lightfm`，版本记录到 `requirements.txt`
- [ ] 跑通 LightFM 官方 `movielens` 快速示例（它自带玩具数据集，不依赖我们的 canon）
- [ ] 确认能多线程训练、能在 5 分钟内跑完一次 fit

**理解算法（W2 上半周）**
- [ ] 读 Metadata Embeddings for User and Item Cold-start Recommendations（arXiv 1507.08439）
- [ ] 弄清核心思想：用户表示 = 属性 embedding 之和，物品同理
      $\hat r_{ui} = \big(\sum_{j\in f_u} e_j\big)^\top\big(\sum_{k\in f_i} e_k\big) + b_u + b_i$
- [ ] 弄清 WARP loss 为什么适合隐式反馈的 top-N

**实现（W2）**
- [ ] 新建 `src/adapters/lightfm.py`，`class LightFMAdapter(AdapterBase)`，`name = "LightFM"`
- [ ] `build_user_features()`：`gender` + `age_group` + `occupation` one-hot → CSR
      ⚠️ **行号必须等于 `uid`，矩阵必须覆盖全部 6,040 行，缺失补全零行**
- [ ] `build_item_features()`：18 个 genres 多热 + `year` 分桶 one-hot → CSR
      ⚠️ **行号必须等于 `iid`，覆盖全部 3,416 行**
- [ ] `fit()`：只用 `split == "train"`；`loss="warp"`；记录训练耗时
- [ ] `score_matrix()`：`user_emb @ item_emb.T + user_bias + item_bias`，返回 `(6040, 3416)` float32
- [ ] 在 `src/adapters/base.py` 的 `_BUILTIN` 里登记 `"lightfm"`
- [ ] 用 `load_adapter("lightfm")` 能实例化并跑通 `evaluate_all()`

**调参（W3 上半周，只用 val）**
- [ ] `--tune` 子命令，网格覆盖 `no_components ∈ {16,32,64,128}`、
      `loss ∈ {warp, bpr}`、`epochs`、`learning_rate`、`item_alpha`、`user_alpha`
- [ ] 至少 10 组，结果写 `outputs/metrics/lightfm_tuning_val.csv`
- [ ] 按 1-SE 规则或 argmax 选定配置，写入 params json

**评估（W3）**
- [ ] `--eval`：**test 只跑一次**，输出 `outputs/metrics/lightfm_test.json`
- [ ] `--analyze`：错误分析表 + 2 张图（沿用 itemcf 的样式）
- [ ] 导出 `outputs/top10/lightfm.parquet`

**文档（W3 末）**
- [ ] 写 `LIGHTFM.md`，沿用 `ITEMCF.md` 的小节结构
- [ ] 把结果行填进 `README.md` §6 的主结果表

### 4.2 已知的坑

| 坑 | 症状 | 对策 |
|---|---|---|
| **特征矩阵行序错位** | 不报错，但结果全错 | 行号直接用 `uid`/`iid`；用 `assert features.shape[0] == N_USERS/N_ITEMS` |
| 稀疏特征未补零行 | `ValueError` 或静默少一行 | `scipy.sparse.vstack` 前先建满 `(N,1)` |
| `num_threads` 默认过高 | 训练卡住或内存爆 | 显式设 `num_threads=4` |
| 把 `rating` 当权重 | 与其他模型口径不一致 | 本项目用隐式协议，`rating` 不参与 |
| 用 LogisticLoss 但数据是隐式 | 收敛慢、指标差 | 用 `warp`（或 `bpr`） |

### 4.3 额外加分（可选，W4）

- [ ] **冷启动实验**：按 `README.md` / `DATA.md` 之外的约定，遮挡 20% 热门物品的评分后重训，
      对比 LightFM 与 Item-CF 在冷物品上的 Recall@10。这是本算法最大的卖点。

---

## 5. ③ LRURec（C）

### 5.1 任务清单

**环境（W2 上半周，全组最高风险，必须最先验证）**
- [ ] `git clone https://github.com/yueqirex/LRURec`
- [ ] 确认 PyTorch 可训练（CPU 也可，但记录单 epoch 耗时）
- [ ] 跑通 README 里的 MovieLens-1M 命令（用官方数据即可，不必用我们的 canon）
- [ ] ⚠️ **不要碰 Mamba4Rec**（需要 `cuda` + `causal-conv1d` + `mamba-ssm`，装不上概率高）
- [ ] 若环境失败 → **立刻上报**，备选 ICSRec 或换算法

**理解算法（W2 上半周）**
- [ ] 读 LRURec（WSDM 2024，DOI 10.1145/3616855.3635760）
- [ ] 弄清：线性 RNN 的并行训练 + 用户序列建模

**实现（W2）**
- [ ] 新建 `src/adapters/lrurec.py`，继承 `AdapterBase`，`name = "LRURec"`
- [ ] 从 canon 造 `(user_id, item_id, timestamp)` 三列无表头文件（官方 `.inter` 格式）
      ⚠️ **item id 必须 1..N，`0` 留给 padding**（canon 是 0-based，需要 `+1`）
- [ ] 序列按 `ts` 升序；train 用 train 段，val/test 各取一条作目标
- [ ] **序列最大长度 `maxlen` 需在这里定稿**（`DATA.md` 的 D3 待定项），建议从 200 起试
- [ ] `fit()`：训练到 early stopping，记录 epoch 数与耗时
- [ ] `score_matrix()`：`user_emb @ item_emb.T`
- [ ] 在 `base.py` 的 `_BUILTIN` 里登记 `"lrurec"`

**调参（W3 上半周，只用 val）**
- [ ] 网格覆盖 `hidden_size`、`num_layers`、`maxlen`、`dropout`、`learning_rate`
- [ ] ≥10 组，写 `outputs/metrics/lrurec_tuning_val.csv`

**评估（W3）**
- [ ] `--eval` 一次 → `outputs/metrics/lrurec_test.json`
- [ ] `--analyze` → 表 + 2 图
- [ ] `outputs/top10/lrurec.parquet`

**文档（W3 末）**
- [ ] 写 `LRUREC.md`
- [ ] 结果行填进 `README.md` §6

### 5.2 已知的坑

| 坑 | 症状 | 对策 |
|---|---|---|
| item id 不是从 1 开始 | embedding 越界 / 结果全错 | canon 的 `iid + 1`；`0` 留 padding |
| 序列长度不足 | 短序列用户被跳过 | 过滤长度 < 3 的用户；本项目最少 16 条，风险低 |
| 直接加载官方 checkpoint | 结果虚高、无法复现 | **必须自己训练** |
| `maxlen` 设太小 | 丢失历史 | 200 覆盖 74% 用户；见 DATA.md §5.4 |
| 用 test 调 early stopping | 泄漏 | early stopping 只看 val |

---

## 6. 非算法线的工作（需要额外指派）

三个算法之外，还有四块工作。

| 工作 | 内容 | 建议归属 | 截止 |
|---|---|---|---|
| **EDA 与数据章** | 评分分布、长尾图、时间趋势、人口统计、类型分布（报告 40% 的分数） | 第 4 人 / A 分摊 | 11/09 |
| **报告整合** | 统一三章的术语、指标口径、图表编号 | A | 11/23 |
| **Streamlit 演示** | 读 `outputs/top10/*.parquet`，做用户选择 + 三模型并排对比 + 画像输入 | 第 4 人 / B | 11/23 |
| **5 分钟彩排** | 按 README §6 的顺序讲：任务→数据→三算法→指标→发现 | 全体 | 11/23 前 ≥2 次 |

---

## 7. 时间节点

| 周 | 日期 | 目标 | 出口条件 |
|---|---|---|---|
| **W2** | 11/03 – 11/09 | B/C 环境跑通 + adapter 出第一版分数 | `load_adapter("lightfm")` / `("lrurec")` 能跑通 `evaluate_all()` |
| **W3** | 11/10 – 11/16 | 调参 + 评估 + 错误分析 + 文档 | 三份 `outputs/metrics/*_test.json` 齐全；三份算法文档完成 |
| **W4** | 11/17 – 11/23 | 演示 + 报告定稿 + 彩排 | Streamlit 可运行；README 主结果表填满；彩排 ≥2 次 |
| buffer | 11/24 | 提交前检查 | README / 种子 / 引用 / 源码打包复核 |
| **DDL** | **11/25（周三）** | 提交 | — |

---

## 8. 依赖关系

```
canon（已完成）──┬──→ ItemCF（已完成）
                 ├──→ LightFM（B）──┬──→ 冷启动对照实验（可选）
                 └──→ LRURec（C）   │
                                     └──→ Streamlit 演示（需三个 top10/*.parquet）
                                                  └──→ 报告定稿 → 彩排
```

**关键路径**：B 和 C 在 11/16 前必须交出 `*_test.json`，否则演示与报告无法收尾。

---

## 9. 验收标准（每人都要满足）

一个算法算"完成"，要同时满足：

- [ ] `python -m src.adapters.<model> --eval` 能跑通，产出 `outputs/metrics/<model>_test.json`
- [ ] `--tune` 的记录 ≥ 10 组，且**全部在 val 上**
- [ ] test 只跑过一次（配置在跑之前已定稿）
- [ ] `--analyze` 产出错误分析表 + 2 张图
- [ ] `outputs/top10/<model>.parquet` 存在，且已排除该用户评过的全部影片
- [ ] 在 `src/adapters/base.py` 的 `_BUILTIN` 中登记
- [ ] `README.md` §6 主结果表填上了自己那一行
- [ ] `<MODEL>.md` 按模板写完，含算法定义、结果、至少一条错误分析
- [ ] 随机种子、超参、环境版本记录在 params json 中
