# MovieLens 电影推荐系统

> CS5481 Recommender Systems · 小组项目（方向一：Movie Recommendation）
> 在 MovieLens-1M 上实现并对比 **三个推荐算法**，提供可交互的推荐演示。

一句话：**给定一个用户，推荐他还没看过、且可能会喜欢的电影。**

---

## 1. 项目做什么

| 项 | 内容 |
|---|---|
| 数据集 | MovieLens-1M（6,040 用户 / 3,416 部电影 / 999,611 条评分） |
| 任务 | Top-N 推荐：为每个用户排出最可能喜欢的 10 部未看过的电影 |
| 算法 | ① Item-based CF（课内）② LightFM（Reference）③ LRURec（2024 新算法） |
| 评估 | Leave-One-Out 切分；Recall@10 / NDCG@10 / HR@10 |
| 交付 | 技术报告 + 源码 + 本 README + 课堂演示 |

评分权重：数据集信息 20% · EDA 与预处理 20% · 算法定义 30% · 推荐实现 20% · 演示 10%

---

## 2. 系统架构

```
ml-1m/*.dat
    │  src/build_canon.py   （一次性：解析 → 过滤 → 重映射 → LOO 切分）
    ▼
data/canon/*.parquet          ← 冻结的数据契约，所有人只读这里
    │
    ├── src/adapters/itemcf.py    Item-based CF
    ├── src/adapters/lightfm.py   LightFM
    └── src/adapters/lrurec.py    LRURec
              │  统一接口：AdapterBase.score_matrix() → (6040, 3416)
              ▼
    src/evaluate.py           ← 指标的唯一出口（full / candidates 两种模式）
              ▼
    outputs/{metrics,figures,top10}  →  app/ 演示界面
```

> 三个模型适配器均已实现；`app/` 是计划中的 Streamlit 界面。LRURec 的本机运行步骤见 [LRUREC.md](LRUREC.md)。

**三条设计约束**（保证三个模型的数字可比）：

1. 所有模型读写同一份 `data/canon/`，**不得自行切分数据**；
2. 模型只输出分数，**指标一律由 `evaluate.py` 计算**；
3. 任何特征/统计量**只能从 `train` 计算**，用 `val` 调参，`test` 只跑一次。

---

## 3. 快速开始

### 3.0 获取代码

```bash
git clone https://github.com/toulandeking/cs5481-movielens-recsys.git
cd cs5481-movielens-recsys
```

> 仓库**只含代码与文档，不含任何数据**（许可证禁止再分发）。
> 数据在 3.2–3.3 两步里由你自己准备，产物与其他人逐字节一致。

### 3.1 软件要求

| 依赖 | 版本 | 用途 |
|---|---|---|
| Python | ≥ 3.10（开发用 3.12.8） | — |
| pandas | 2.3.1 | 数据处理 |
| numpy | 2.1.3 | 数值计算 |
| scipy | 1.16.0 | 稀疏矩阵（Item-CF 相似度） |
| pyarrow | 26.0.0 | parquet 读写（缺失时自动降级为 pickle） |
| matplotlib | ≥ 3.7 | 结果图 |
| lightfm / torch | lightfm 1.16 / torch 2.7.1+cu128（LRURec CUDA 环境） | 分别使用独立环境，见算法文档 |

```bash
python -m pip install -r requirements.txt
```

> 上表是开发验证用的具体版本，`requirements.txt` 里写的是宽松下界，允许有差异。
> 若 `pyarrow` 装不上，canon 会自动降级为 pickle，其余功能不受影响。

### 3.2 数据准备

⚠️ **克隆下来的仓库里既没有 `ml-1m/`，也没有 `data/canon/`** —— 这是刻意的：
ML-1M 的许可证禁止再分发数据，而 `build_canon` 完全确定，每人本地生成的产物逐字节相同。

从 [GroupLens](https://grouplens.org/datasets/movielens/1m/) 下载 `ml-1m.zip`，解压到项目根目录，形成：

```
Recommender_System/
├── ml-1m/
│   ├── ratings.dat  users.dat  movies.dat  README
```

> ⚠️ 原始数据**不得提交到 git**（许可证禁止再分发）。

### 3.3 生成数据集

```bash
python -m src.build_canon          # 生成 data/canon/，约 6 秒
python -m src.build_canon --check  # 校验产物与预期一致
python -m src.contract             # 打印当前生效的全部配置
```

### 3.4 训练与评估

```bash
# Item-CF（已完成）
python -m src.adapters.itemcf --tune      # 在 val 上搜超参
python -m src.adapters.itemcf --eval      # test 评估 + 导出 top-10
python -m src.adapters.itemcf --analyze   # 错误分析表 + 两张图
python -m src.adapters.itemcf --top --uid 42   # 查看某个用户的推荐

# LightFM（已完成）
python -m src.adapters.lightfm --tune      # 在 val 上搜超参（20 组，约 35 分钟）
python -m src.adapters.lightfm --eval      # test 评估 + 导出 top-10（约 3 分钟）
python -m src.adapters.lightfm --analyze   # 错误分析表 + 两张图
python -m src.adapters.lightfm --top --uid 42   # 查看某个用户的推荐

# LRURec（独立 .venv-lrurec-cuda 环境，完整新20组，详见 LRUREC.md）
python -m src.adapters.lrurec_fresh20 --run-dir outputs/lrurec_run20 --workers 1 --device cuda
# 已完成目录的验收与推荐查看（不重新训练）
python -m src.adapters.lrurec_fresh20 --run-dir outputs/lrurec_run20 --check
python -m src.adapters.lrurec_fresh20 --run-dir outputs/lrurec_run20 --top --uid 42
```

> ⚠️ Windows 上 LightFM 要用独立的 CPython 3.9 解释器（原因见 [LIGHTFM.md](LIGHTFM.md) §9）：
> `& "<workspace>\.py39e\rt\python.exe" -m src.adapters.lightfm --eval`。Linux/Colab 直接 `python -m`。

### 3.5 按画像推荐（演示用）

模拟一个**新用户**，输入人口属性与偏好类型即可得到推荐：

```bash
python -m src.demo_profile --gender F --age 25-34 --genres Comedy,Romance --top 10 --explain
```

> 说明：Item-CF 无法直接消费性别/年龄/类型，该命令用「同群用户画像 → 伪历史 → 邻域扩散」
> 的近似方案实现，同时作为冷启动实验的对照组。原生支持用户特征的是 LightFM。

### 3.6 环境自检（新成员必做）

上面 3.1–3.3 跑完后，执行这四条命令；**全部通过说明环境已就绪**，可以开始写自己的算法：

| 命令 | 预期 |
|---|---|
| `python -m src.contract` | 打印 `N_ITEMS 3416`、`CANON_FORMAT parquet` |
| `python -m src.build_canon --check` | 每一行都是 ✓ |
| `python -m src.evaluate --random` | full ≈ 0.0029、candidates ≈ 0.0990 |
| `python -m src.adapters.itemcf --eval` | full Recall@10 = **0.0641** |

最后一条能复现 0.0641，说明**数据、切分、指标、模型**整条链路都与团队完全一致。

---

## 4. 当前状态

| 模块 | 状态 | 说明 |
|---|---|---|
| 数据管道（canon） | ✅ 完成 | 4 个 parquet + id_map.json，全量 `assert` 校验 |
| 评估框架 | ✅ 完成 | full / candidates 双模式，含随机与爆款基线自检 |
| 适配层基类 | ✅ 完成 | `AdapterBase` + 模型注册表 |
| **Item-CF** | ✅ 完成 | 调参、评估、错误分析、演示导出 |
| **LightFM** | ✅ 完成 | 调参、评估、错误分析、演示导出；test Recall@10 = 0.0775 |
| LRURec | ✅ 完成 | 全新 20 组，旧结果不计；GPU 每组最多 5 epoch；full test Recall@10=0.2272；见 LRUREC.md |
| Streamlit 演示 | ⬜ 待开始 | 依赖三个模型的 `outputs/top10/*.parquet` |
| 技术报告 | ⬜ 进行中 | 数据及三个算法章节素材已就绪，待整合 |

详细的任务拆分见 [TASKS.md](TASKS.md)。

---

## 5. 评估协议

**切分**：Leave-One-Out —— 每个用户按时间排序，最后一条作 test、倒数第二条作 val、其余作 train。
（train 987,531 / val 6,040 / test 6,040）

**相关性**：所有观测到的交互均为正样本（隐式反馈协议），`rating` 数值不参与训练与评估。

**两种评估模式**（必须来自同一分数矩阵）：

| 模式 | 做法 | 用途 |
|---|---|---|
| **full** | 对全部 3,416 部电影打分排序 | **主表**，无负采样歧义 |
| candidates | 1 正 + 100 固定负样本内排序 | 副表，对齐采用 1+99 协议的论文 |

> 两种模式的绝对值**不可跨比**，报告中同时给出是为了覆盖不同的文献口径。

---

## 6. 结果与图表位置

| 结果 | 路径 |
|---|---|
| Item-CF 指标（两模式） | `outputs/metrics/itemcf_test.json` |
| Item-CF 调参记录（20 组） | `outputs/metrics/itemcf_tuning_val.csv` |
| Item-CF 最终超参 | `outputs/metrics/itemcf_params.json` |
| 错误分析（按目标热度分档） | `outputs/metrics/itemcf_popularity_buckets.csv` |
| 图：超参敏感性 | `outputs/figures/itemcf_alpha_k_val.png` |
| 图：流行度偏置 | `outputs/figures/itemcf_popularity_bias.png` |
| 推荐结果（演示用） | `outputs/top10/itemcf.parquet` |
| LightFM 指标（两模式） | `outputs/metrics/lightfm_test.json` |
| LightFM 调参记录（20 组） | `outputs/metrics/lightfm_tuning_val.csv` |
| LightFM 最终超参 | `outputs/metrics/lightfm_params.json` |
| LightFM 错误分析（按目标热度分档） | `outputs/metrics/lightfm_popularity_buckets.csv` |
| LightFM 图：超参/特征敏感性 | `outputs/figures/lightfm_tuning_sensitivity.png` |
| LightFM 图：流行度偏置 | `outputs/figures/lightfm_popularity_bias.png` |
| LightFM 推荐结果（演示用） | `outputs/top10/lightfm.parquet` |
| LRURec 指标（两模式） | [results/lrurec/lrurec_test.json](results/lrurec/lrurec_test.json) |
| LRURec 调参记录（全新 20 组） | [results/lrurec/lrurec_tuning_val.csv](results/lrurec/lrurec_tuning_val.csv) |
| LRURec 最终超参 | [results/lrurec/lrurec_params.json](results/lrurec/lrurec_params.json) |
| LRURec 错误分析（按目标热度分档） | [results/lrurec/lrurec_popularity_buckets.csv](results/lrurec/lrurec_popularity_buckets.csv) |
| LRURec 图：20 组参数敏感性 | [results/lrurec/lrurec_tuning_sensitivity.png](results/lrurec/lrurec_tuning_sensitivity.png) |
| LRURec 图：流行度偏置 | [results/lrurec/lrurec_popularity_bias.png](results/lrurec/lrurec_popularity_bias.png) |
| LRURec 本地推荐 / 检查点 | `<run>/top10/lrurec.parquet`、`<run>/models/lrurec_best.pt`（运行命令生成） |

> 三个算法的全部细节见 [ITEMCF.md](ITEMCF.md)、[LIGHTFM.md](LIGHTFM.md) 与 [LRUREC.md](LRUREC.md)。

> 当前 LRURec 主结果来自 `outputs/lrurec_fresh20_cuda_20261109/`：20 组全部重新初始化和训练，旧 10 组不计。它是既有 test 已查看后的完整重复实验；每次实验内只在选参后计算 test 一次。新运行和验收命令见 [LRUREC.md](LRUREC.md)。

### 当前结果（test，K=10）

| 模型 | full Recall@10 | full NDCG@10 | candidates Recall@10 | candidates NDCG@10 |
|---|---:|---:|---:|---:|
| 随机打分 | 0.0033 | 0.0014 | 0.0990 | 0.0430 |
| 最流行基线 | 0.0353 | 0.0173 | 0.4159 | 0.2237 |
| Item-CF | 0.0641 | 0.0335 | 0.5560 | 0.3224 |
| LightFM | 0.0775 | 0.0398 | 0.6621 | 0.3895 |
| **LRURec（全新20组）** | **0.2272** | **0.1233** | **0.7531** | **0.5366** |

> **LRURec 相对 LightFM：full Recall@10 +193.1%、NDCG@10 +209.8%；candidates +13.8% / +37.8%。** 完整 20 组均重新训练，旧 10 组不计；本次使用此前已查看过的同一 test，属于重复实验。每组最多 5 epoch，不代表充分收敛上限。见 [LRUREC.md](LRUREC.md)。

> **LightFM 相对 Item-CF：full Recall@10 +20.9%、NDCG@10 +18.7%、candidates +19.1% / +20.8%。**
> 增益集中在训练期评分数 100–1499 的中长尾影片（+36%~+56%），热门档仅 +2.6%，评分数 <100 的物品无改善 ——
> 见 [LIGHTFM.md](LIGHTFM.md) §6 的分档对照表。

---

## 7. 可复现性

| 项 | 值 |
|---|---|
| 全局随机种子 | **42**（`src/contract.py` 的 `SEED`） |
| 子种子 | `negative_sampling` / `cold_start_split` / `model_init` 均为 42 |
| 确定性 | 切分按 `(timestamp, MovieID)` 稳定排序；相似度截断用 stable argsort |
| 环境 | Python 3.12.8 · pandas 2.3.1 · numpy 2.1.3 · scipy 1.16.0 · pyarrow 26.0.0 |
| LightFM 子环境 | 因 lightfm 在 Windows 无可用 wheel，单独用 CPython **3.9.13**（embeddable）· numpy 1.23.5 · scipy 1.10.1 · pandas 1.5.3 · pyarrow 12.0.1 · lightfm **1.16**；启动器 `<workspace>\.py39e\rt\python.exe`，详见 [LIGHTFM.md](LIGHTFM.md) §10 |
| LRURec CUDA 环境 | Python 3.13.12 · PyTorch 2.7.1+cu128 · CUDA 12.8 · RTX 5060 Laptop GPU；[精确依赖](requirements-lrurec-cuda.txt) |
| 数据切分 | 固定写入 `data/canon/`，任何人不得在脚本内重新切分 |

---

## 8. 数据集与引用

本数据集的使用须遵守 GroupLens 的许可：**仅限研究用途，不得再分发，不得商用。**

> F. Maxwell Harper and Joseph A. Konstan. 2015. **The MovieLens Datasets: History and Context.**
> *ACM Transactions on Interactive Intelligent Systems (TiiS)* 5, 4, Article 19 (December 2015), 19 pages.
> DOI: [10.1145/2827872](https://doi.org/10.1145/2827872)

- 数据来源：<https://grouplens.org/datasets/movielens/1m/>
- 目录中的 `ml-1m/README` 为官方说明，包含完整的字段字典与许可条款。

---

## 9. 文档索引

| 文档 | 内容 |
|---|---|
| [DATA.md](DATA.md) | 数据结构、7 个坑、四项关键决策的依据、canon 规格与读取纪律 |
| [ITEMCF.md](ITEMCF.md) | Item-CF 的算法定义、结果、三条关键发现、结果展示方案 |
| [LIGHTFM.md](LIGHTFM.md) | LightFM 的算法定义、WARP 动机、20 组调参、长尾分档对照、Windows 踩坑与复现清单 |
| [LRUREC.md](LRUREC.md) | LRURec 定义、完整新 20 组实验、结果、错误分析、GPU 环境与复现步骤 |
| [TASKS.md](TASKS.md) | 三人三算法的任务清单、验收标准、时间节点 |

> 新增算法文档时请沿用 `ITEMCF.md` 的小节结构（状态 → 模块清单 → 算法定义 → 运行 →
> 结果 → 发现 → 展示建议 → 局限 → 复现清单 → 变更记录）。

---

## 10. 已知限制

- LRURec 主表使用 train-only 推理历史以统一信息范围，未在 test 输入中追加 val；这与官方常用协议不同。每组最多 5 epoch，结果不代表充分训练上限。

- Item-CF 的命中高度集中在热门电影：目标评分数 < 100 时命中率不足 0.5%，
  长尾与冷物品需要内容侧信息（这正是引入 LightFM 的动机）。
- **LightFM 缓解但没有解决这个问题**：100–1499 条评分的中长尾比 Item-CF 高 36%~56%，
  但 < 100 条评分的物品命中率与 Item-CF 持平（0.43%），< 20 条为 0 —— 零交互物品仍不可荐。
- **LightFM 的 `identity` 列意味着它是 hybrid MF 而非纯 content-based**（纯侧信息只有 241 种用户向量），
  报告引用时不要把它当作纯冷启动方案的证据。
- `val` 与 `test` 的难度不同（`val` 系统性偏高约 0.015），**`val` 只用于选超参，
  报告中一律报 `test`**。
- 全量排序下 Recall@10 的绝对值不高（0.06–0.08），这是 3,416 部候选、单正样本的固有难度，
  不同论文采用的切分与候选集不同，**数值不可直接横向比较**。
