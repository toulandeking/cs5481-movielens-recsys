# 任务清单 

> **更新**：2026-11-09

> **配套**：[README.md](README.md) · [DATA.md](DATA.md) · [ITEMCF.md](ITEMCF.md)（模板）

---

## 0. 总览

| # | 算法 | 类别（作业要求） | 负责人 | 状态 | 截止 |
|---|---|---|---|---:|---|
| ① | **Item-based CF** | 课内讲过的算法 | **A** | ✅ **完成** | — |
| ② | **LightFM** | Reference Algorithms | **B** | ✅ **完成** | 11/16 |
| ③ | **LRURec** | 2024–2025 新算法 | **C** | ✅ 完成（CPU 有限预算） | 11/16 |

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

## 4. ② LightFM（B）— ✅ 已完成

> **test 结果**：full Recall@10 = **0.0775**，NDCG@10 = 0.0398；candidates 0.6621 / 0.3895。
> 相对 Item-CF 四个指标 +18.7%~+20.9%。细节见 [LIGHTFM.md](LIGHTFM.md)。

### 4.1 任务清单

**环境（W2 上半周）**
- [x] 版本记录到 `requirements.txt` —— ⚠️ **不是 pip 装的**：PyPI 无 Windows wheel、本机无 MSVC；
      最终用 conda-forge 的 win-64 构建 **lightfm 1.16** + 内嵌 CPython 3.9.13（`1.17` 在 Windows 上 `loss="warp"` 必崩，见 §4.2）
- [x] 跑通 LightFM 官方 `movielens` 快速示例（自带玩具集，不依赖我们的 canon）：precision@10 = 0.0677
- [x] 能在 5 分钟内跑完一次 fit：全量 6040×3416 训练 20 epoch 实测 **173 s**
- [ ] ⚠️ 多线程：**做不到**（不是没试）—— lightfm 的 `setup.py` 里 `use_openmp = not darwin and not win`，
      Windows 构建一律 no-OpenMP，导入即 `UserWarning: ... Only a single thread will be used`；
      `num_threads=4` 已显式传入但被忽略。Linux/Colab 上同一份代码可多线程（约 60 s）

**理解算法（W2 上半周）**
- [x] 读 Metadata Embeddings for User and Item Cold-start Recommendations（arXiv 1507.08439）
- [x] 弄清核心思想：用户表示 = 属性 embedding 之和，物品同理
      $\hat r_{ui} = \big(\sum_{j\in f_u} e_j\big)^\top\big(\sum_{k\in f_i} e_k\big) + b_u + b_i$
- [x] 弄清 WARP loss 为什么适合隐式反馈的 top-N（含实测证据：logistic 0.0026 < bpr 0.0119 < warp 0.0159）

**实现（W2）**
- [x] 新建 `src/adapters/lightfm.py`，`class LightFMAdapter(AdapterBase)`，`name = "LightFM"`
- [x] `build_user_features()`：`gender` + `age_group` + `occupation` one-hot → CSR
      ⚠️ **行号必须等于 `uid`，矩阵必须覆盖全部 6,040 行，缺失补全零行**（已加 `assert` + strict 校验；实测 6040 行、无全零行）
- [x] `build_item_features()`：18 个 genres 多热 + `year` 分桶 one-hot → CSR
      ⚠️ **行号必须等于 `iid`，覆盖全部 3,416 行**（实测 3416 行、每行 3–8 个非零）
- [x] `fit()`：只用 `split == "train"`；`loss="warp"`；记录训练耗时
- [x] `score_matrix()`：`user_emb @ item_emb.T + user_bias + item_bias`，返回 `(6040, 3416)` float32
- [x] 在 `src/adapters/base.py` 的 `_BUILTIN` 里登记 `"lightfm"`
- [x] 用 `load_adapter("lightfm")` 能实例化并跑通 `evaluate_all()`

**调参（W3 上半周，只用 val）**
- [x] `--tune` 子命令，网格覆盖 `no_components ∈ {16,32,64,128}`、
      `loss ∈ {warp, bpr}`、`epochs`、`learning_rate`、`item_alpha`、`user_alpha`、`max_sampled`
- [x] 至少 10 组（实跑 **20 组**），结果写 `outputs/metrics/lightfm_tuning_val.csv`
- [x] 按 1-SE 规则选定配置（组 16：c=32·ms=100·identity），写入 `lightfm_params.json`

**评估（W3）**
- [x] `--eval`：**test 只跑一次**，输出 `outputs/metrics/lightfm_test.json`
- [x] `--analyze`：错误分析表 + 2 张图（沿用 itemcf 的样式）
- [x] 导出 `outputs/top10/lightfm.parquet`（60,400 行，零泄漏）

**文档（W3 末）**
- [x] 写 `LIGHTFM.md`，沿用 `ITEMCF.md` 的小节结构
- [x] 把结果行填进 `README.md` §6 的主结果表

### 4.2 已知的坑

| 坑 | 症状 | 对策 |
|---|---|---|
| **特征矩阵行序错位** | 不报错，但结果全错 | 行号直接用 `uid`/`iid`；用 `assert features.shape[0] == N_USERS/N_ITEMS` |
| 稀疏特征未补零行 | `ValueError` 或静默少一行 | `scipy.sparse.vstack` 前先建满 `(N,1)` |
| `num_threads` 默认过高 | 训练卡住或内存爆 | 显式设 `num_threads=4` |
| 把 `rating` 当权重 | 与其他模型口径不一致 | 本项目用隐式协议，`rating` 不参与 |
| 用 LogisticLoss 但数据是隐式 | 收敛慢、指标差 | 用 `warp`（或 `bpr`） |
| **lightfm 1.17 在 Windows 上 warp 必崩**（实测坑，原清单没写） | 进程直接消失，`exit=-1073741819` = `0xC0000005`；`python -X faulthandler` 定位到 `lightfm.py:696 fit_warp`；事件日志 `_lightfm_fast_no_openmp*.pyd 偏移 0x6708`。上游 issue #720，1.17 是回归 | 降到 **lightfm 1.16**（conda-forge win-64 构建）；或换 Linux/Colab |
| **特征只有人口统计时用户向量撞车**（实测坑，本轮最大发现） | 6040 用户只有 **241 种**不同特征向量（平均 25 人共享），val Recall@10 卡在 0.0151，低于爆款基线 0.0353 | 加身份列（`identity=True`，用户 6,070 / 物品 3,444 列）→ val 0.0919；这样模型是 **hybrid MF**，报告里别写成纯 content-based |

### 4.3 额外加分（可选，W4）

- [ ] **冷启动实验**：按 `README.md` / `DATA.md` 之外的约定，遮挡 20% 热门物品的评分后重训，
      对比 LightFM 与 Item-CF 在冷物品上的 Recall@10。这是本算法最大的卖点。
      > 现状：**未做**。但 §4.2 最后一行已经给出了它的动机与量化证据 ——
      > 当前特征设计在"有少量交互"的中长尾（100–1499 条）上比 Item-CF 高 36%~56%，
      > 而在 <100 条评分的物品上完全无改善，恰好说明"零交互"只能靠遮挡实验来单独衡量。

---

## 5. ③ LRURec（C）

### 5.1 任务清单

**环境（已在独立 `.venv-lrurec-cuda` 验证，PyTorch 2.7.1+cu128 / CUDA 12.8 / RTX 5060）**
- [x] 核对官方实现 `yueqirex/LRURec`，保留模型出处和基准提交
- [x] 确认 GPU 全量训练、检查点保存与重载一致性，记录逐 epoch 耗时
- [x] 官方模型在团队 canon 上跑通全量训练与评估（不运行官方独立重切分与评估）
- [x] ⚠️ **不要碰 Mamba4Rec**（需要 `cuda` + `causal-conv1d` + `mamba-ssm`，装不上概率高）
- 环境已成功，无需启用 ICSRec 备选。

**理解算法（W2 上半周）**
- [x] 核对 LRURec 官方实现与论文定义（WSDM 2024，DOI 10.1145/3616855.3635760）
- [x] 弄清：线性 RNN 的并行训练 + 用户序列建模

**实现（W2）**
- [x] 完整 20 组运行器 `src/adapters/lrurec_fresh20.py`；新目录、随机初始化、拒绝复用旧检查点
- [x] 新建 `src/adapters/lrurec.py`，继承 `AdapterBase`，`name = "LRURec"`
- [x] 直接从 canon 生成按时间排列的训练张量，无须中间 `.inter` 文件
      ⚠️ **item id 必须 1..N，`0` 留给 padding**（canon 是 0-based，需要 `+1`）
- [x] 序列按 `ts` 升序；train 用 train 段，val/test 各取一条作目标
- [x] **序列最大长度 `maxlen=50` 已定稿**（已同步 DATA.md D3；全新20组 val 联调选择）
- [x] `fit()`：训练到 early stopping，记录 epoch 数与耗时
- [x] `score_matrix()`：序列状态与 item embedding 点积，加官方输出偏置
- [x] 在 `base.py` 的 `_BUILTIN` 里登记 `"lrurec"`

**调参（W3 上半周，只用 val；已完成全新20组，旧10组不计，每组最多 5 epoch，patience=2）**
- [x] 网格覆盖 `hidden_size`、`num_layers`、`maxlen`、`dropout`、`learning_rate`
- [x] 已完成全新20组；[完整调参记录](results/lrurec/lrurec_tuning_val.csv)，旧结果计入数为 0

**评估（W3）**
- [x] 本次 20 组选参后评估 test 一次；[指标 JSON](results/lrurec/lrurec_test.json)
- [x] 错误分析表和 2 张图已发布到 [results/lrurec/](results/lrurec/README.md)
- [x] `outputs/lrurec_fresh20_cuda_20261109/top10/lrurec.parquet`

**验收**
- [x] 4 项算法测试 + 3 项新实验验收全部通过；[验证记录](results/lrurec/verification.txt)

**文档（W3 末）**
- [x] 按 ITEMCF 文档结构完成 [LRUREC.md](LRUREC.md)
- [x] 结果行填进 `README.md` §6

### 5.2 已知的坑

| 坑 | 症状 | 对策 |
|---|---|---|
| item id 不是从 1 开始 | embedding 越界 / 结果全错 | canon 的 `iid + 1`；`0` 留 padding |
| 序列长度不足 | 短序列用户被跳过 | 过滤长度 < 3 的用户；本项目最少 16 条，风险低 |
| 直接加载官方 checkpoint | 结果虚高、无法复现 | **必须自己训练** |
| `maxlen` 设太小 | 丢失历史 | 200 覆盖 74% 用户；见 DATA.md §5.4 |
| 用 test 调 early stopping | 泄漏 | early stopping 只看 val |

---


> 2026-11-09：按用户要求完整重跑 20 组，旧结果计入数为 0。新目录独立选参并评估一次 test；由于旧 test 已查看，重复实验事实已记录。详见 LRUREC.md。

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
