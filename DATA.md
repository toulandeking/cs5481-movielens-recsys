# 数据处理说明（ML-1M）

> **状态**：🟡 决策已定稿，代码尚未开始（见第 2 节）
> **最后更新**：2026-10-28
> **配套文档**：[FRAMEWORK.md](FRAMEWORK.md)（架构设计）

---

## 0. 30 秒速读

如果你只想记 6 句话，记这 6 句：

1. 数据在 `ml-1m/` 下，**3 个 `.dat` 文件，全部没有表头，分隔符是 `::`**。
2. **`movies.dat` 必须用 `latin-1` 打开**，用 utf-8 会直接报错。
3. 最终建模口径：**`N_USERS = 6040`，`N_ITEMS = 3416`**。
4. **原始 ID 不能直接用**：`MovieID` 1..3952 里有 69 个空洞，必须重映射成连续 id。
5. **物品过滤必须在 LOO 切分之前做**，顺序反了会丢用户。
6. **绝对不要自己切分数据**，等 `data/canon/` 生成后只读它（第 6 节）。

---

## 1. 数据文件在哪、长什么样

```
Recommender_System/
├── ml-1m.zip              # 官方压缩包 5.9 MB（原始下载物）
└── ml-1m/                 # 解压后
    ├── ratings.dat        # 24.6 MB   1,000,209 行
    ├── users.dat          # 134 KB    6,040 行
    ├── movies.dat         # 171 KB    3,883 行
    └── README             # 官方说明 170 行（无扩展名）
```

> 📌 **规划**：后续会移到 `data/raw/ml-1m/` 并加入 `.gitignore`（许可禁止再分发，见第 8 节）。
> 现在的位置不影响使用。

### 1.1 `ratings.dat` —— 核心交互表 ⭐ 最重要

```
UserID::MovieID::Rating::Timestamp
1::1193::5::978300760
1::661::3::978302109
1::914::3::978301968
```

| 字段 | 类型 | 原始范围 | 含义 |
|---|---|---|---|
| `UserID` | int | 1 – 6040（**6040 个全在，连续无空洞**） | 用户 id |
| `MovieID` | int | 1 – 3952，**但只有 3706 个不同值** | 电影 id（**稀疏！见坑 #3**） |
| `Rating` | int | `{1,2,3,4,5}`，只取整 | 星级评分 |
| `Timestamp` | int | `2000-04-25` → `2003-02-28` | Unix 秒 |

### 1.2 `users.dat` —— 用户人口统计

```
UserID::Gender::Age::Occupation::Zip-code
1::F::1::10::48067
2::M::56::16::70072
```

| 字段 | 取值 | 实测分布 |
|---|---|---|
| `Gender` | `M` / `F` | M 4,331（71.7%）/ F 1,709（28.3%） |
| `Age` | **7 个分桶码** | `1:222, 18:1103, 25:2096, 35:1193, 45:550, 50:496, 56:380` |
| `Occupation` | **21 个类别码** 0–20 | 最多的：`4 student 759`、`0 other 711`、`7 exec 679`、`1 educator 528` |
| `Zip-code` | **字符串** | 3,439 个唯一值，662 个有前导零，66 个带连字符 |

**`Age` 是分桶码，`1` 表示 "Under 18"，不是 1 岁！**

| 码 | 含义 | | 码 | 含义 |
|---:|---|---|---:|---|
| 1 | **Under 18** | | 45 | 45–49 |
| 18 | 18–24 | | 50 | 50–55 |
| 25 | 25–34 | | 56 | 56+ |
| 35 | 35–44 | | | |

**`Occupation` 完整映射**（来自官方 README）

| 码 | 含义 | | 码 | 含义 | | 码 | 含义 |
|---:|---|---|---:|---|---|---:|---|
| 0 | other / 未填 | | 7 | executive/managerial | | 14 | sales/marketing |
| 1 | academic/educator | | 8 | farmer | | 15 | scientist |
| 2 | artist | | 9 | homemaker | | 16 | self-employed |
| 3 | clerical/admin | | 10 | K-12 student | | 17 | technician/engineer |
| 4 | college/grad student | | 11 | lawyer | | 18 | tradesman/craftsman |
| 5 | customer service | | 12 | programmer | | 19 | unemployed |
| 6 | doctor/health care | | 13 | retired | | 20 | writer |

### 1.3 `movies.dat` —— 电影元数据

```
MovieID::Title::Genres
1::Toy Story (1995)::Animation|Children's|Comedy
73::Misérables, Les (1995)::Drama|Musical
260::Star Wars: Episode IV - A New Hope (1977)::Action|Adventure|Fantasy|Sci-Fi
```

| 字段 | 说明 |
|---|---|
| `MovieID` | 1 – 3952，**有 69 个 id 根本不存在** |
| `Title` | 片名**末尾带年份**，正则 `\((\d{4})\)\s*$` 可 100% 解析，年份跨度 **1919 – 2000** |
| `Genres` | **多标签**，用竖线分隔，共 **18 类** |

**18 个类型及电影数**

`Drama 1603` · `Comedy 1200` · `Action 503` · `Thriller 492` · `Romance 471` · `Horror 343` ·
`Adventure 283` · `Sci-Fi 276` · `Children's 251` · `Crime 211` · `War 143` · `Documentary 127` ·
`Musical 114` · `Mystery 106` · `Animation 105` · `Fantasy 68` · `Western 68` · `Film-Noir 44`

- **52.2% 的电影只有 1 个类型**，最多的有 6 个类型。
- 这 18 个类型 + 上映年份，是 LightFM 的 `item_features` 来源。

---

## 2. 处理进度与最终口径

### 2.1 进度状态

| 阶段 | 状态 | 说明 |
|---|---|---|
| 文件格式侦察 | ✅ 完成 | 编码、分隔符、有无表头、行数，全部实测确认 |
| 字段字典 | ✅ 完成 | 从官方 README 提取 Age / Occupation 完整映射 |
| 数据质量检查 | ✅ 完成 | 缺失、重复、外键完整性、ID 连续性，全部跑过 |
| 分布统计 | ✅ 完成 | 评分分布、用户/物品活跃度、时间分布、稀疏度 |
| **关键决策** | ✅ 已定稿 | 见 §2.2 |
| `src/contract.py` | ✅ 完成 | 常量、路径、字段字典、canon 读写入口 |
| `src/build_canon.py` | ✅ 完成 | 7 步流水线，25 处 `assert` |
| **`data/canon/` 产物** | ✅ **已生成（parquet）** | 5 个文件，共 **6.0 MB** |
| 质量校验 | ✅ 通过 | `python -m src.build_canon --check` 全绿 |

> ✅ **数据已就绪**，`data/canon/` 可直接使用。读法见 §6，纪律见 §6.3–§6.4。
> 总产物体积 6.0 MB = ratings 4.69 + eval_candidates 0.96 + movies 0.11 + users 0.08 + id_map 0.17。

### 2.2 已定稿的四个决策

| # | 决策 | 结论 | 依据 |
|---|---|---|---|
| **D1** | 相关性标签 | **全部交互为正**（隐式反馈协议） | 见 §5.1：用 `≥4` 会让 41.84% 的测试目标失去意义 |
| **D2** | 物品过滤 | **剔除评分 <5 的物品** | 见 §5.2：只损失 0.06% 的数据，换来相似度稳定性 |
| **D3** | 序列最大长度 | **50**（LRURec 全新20组 val 联调选定） | 只影响序列模型，详见 [LRUREC.md](LRUREC.md) |
| **D4** | 评估协议 | **全量排序为主 + 101 候选为辅** | 见 §5.3：全量排序成本仅 82.5 MB，无负采样歧义 |

### 2.3 最终口径（canon 必须产出的数字）

| 量 | 值 |
|---|---|
| `N_USERS` | **6,040** |
| `N_ITEMS` | **3,416** |
| ratings 行数 | **999,611** |
| train / val / test | **987,531** / **6,040** / **6,040** |
| 稀疏度 | **95.16%** |
| 平均分 | 3.5820 |
| 每用户评分数 | min **18** / 中位 96 / max **2,277** |
| 每电影评分数 | min **5** / 中位 146 / max **3,428** |

---

## 3. 数据质量检查结果

| 检查项 | 结果 | 结论 |
|---|---|---|
| `ratings` 重复 `(user, movie)` 对 | **0 条** | ✅ 无需去重 |
| 空值 / NaN | **0 个** | ✅ 三个文件都无缺失值 |
| `users.dat` 的 uid == `ratings.dat` 的 uid | ✅ **完全一致**（各 6040） | ✅ 可直接 join |
| `ratings` 中出现但 `movies` 没有的 MovieID | **0 个** | ✅ 无悬挂外键 |
| `UserID` 是否连续 1..6040 | ✅ **是** | 减 1 即可 0-based |
| `MovieID` 是否连续 | ❌ **否**，缺 69 个 | ⚠️ **必须重映射** |

> 💡 **重点**：这份数据**没有缺失值、没有重复**。所以"处理空值"在 ML-1M 上几乎不是问题。
> 唯一的"缺失"是那 177 部零评分电影的**整列空缺**，而那不是空值问题，
> 而是"该不该存在于物品空间"的问题（见 §5.2）。

---

## 4. 用之前必须知道的 7 个坑

| # | 坑 | 你看到的症状 | 正确做法 |
|---|---|---|---|
| **1** | `movies.dat` 是 **latin-1**，不是 utf-8 | `UnicodeDecodeError: invalid continuation byte`（byte 3114） | `encoding='latin-1'` |
| **2** | **三个文件都没有表头** | 少了一行数据；`users.dat` 少一个用户，你完全察觉不到 | `header=None` |
| **3** | **`MovieID` 稀疏**（1..3952 只有 3883 个） | `IndexError`，或 69 个空槽位浪费内存 | 重映射成 `0..N-1` |
| **4** | `Age` 码 `1` = "Under 18" | 当数值标准化 → 语义完全错误 | 映射成类别（或区间中点 17） |
| **5** | `Zip-code` 前导零 | `02460` → `246`，信息丢失 | `dtype={'Zip-code': str}` |
| **6** | 时间分布**极度偏斜** | 全局时间切分后测试期几乎没人 | 用 **LOO**，不用时间切分 |
| **7** | **物品过滤顺序** | 先切分后过滤 → 丢 6 个用户的测试目标 | **先过滤，后切分**（见 §5.2） |

### 坑 #6 的实测证据

```
2000-11  290,793  ████████████████████████████████████████████████████████████  ← 单月占 29%
2000-08  182,109  ███████████████████████████████████████████
2001-01   18,004  ████
2002-06    1,643  ▌
2003-02    1,496  ▌
```

**若用"最后 10% 时间"当测试期，约半数用户在测试期一条评分都没有** → 无法评估。
这是 LOO 决策的**数据侧证据**，报告里可以引用。

---

## 5. 四个决策的完整依据

### 5.1 D1：为什么用"全部交互为正"，而不是 `rating ≥ 4`

原始数据是 **1–5 星的显式评分**，但隐式反馈模型（LightFM、LRURec）只吃"**有交互 / 无交互**"的 0/1 信号。所以要先定义：什么算"用户喜欢"？

| 定义 | 正样本占比 | 语义 |
|---|---:|---|
| **A. 全部交互为正** ✅ | **100%** | "用户主动看了并评分" = 感兴趣 |
| B. `rating ≥ 4` 为正 | 57.52% | "4 星以上" = 喜欢 |
| C. `rating == 5` 为正 | 22.63% | "5 星" = 非常喜欢 |

**决定性证据**：我们用 Leave-One-Out，每个用户**最后一条评分**当作测试目标。它的星级分布是：

| test target 星级 | 用户数 | 占比 |
|---:|---:|---:|
| 1 星 | 403 | 6.67% |
| 2 星 | 663 | 10.98% |
| 3 星 | 1,460 | **24.17%** |
| 4 星 | 1,992 | 32.98% |
| 5 星 | 1,522 | 25.20% |

> **有 41.84% 的用户，其测试目标评分 ≤ 3 星。**

若按 `≥4` 才算"相关"，评估就变成："**模型有没有把这部 2 星烂片排进 top-10？**"
正确答案应该是"没有"，但指标却会惩罚"没排进去"的模型。**指标与目标反向。**

因此选定 **协议 A（全部交互为正）**：

1. 这是 ML-1M 上 top-N 评估的**行业标准**，NCF / BPR / LRURec / ICSRec 官方实现全都如此。
2. 语义站得住：隐式反馈框架下，"**用户选择观看并评分**"这个行为本身代表兴趣，与评分高低无关。
3. 结构最简单：交互矩阵是纯 0/1，没有"中间态"物品。
4. 完全避开上述 41.84% 的陷阱。

> 📌 **报告写法**（对应 rubric「Explain how ratings are converted into relevance labels」）：
> "We follow the standard implicit-feedback protocol for top-N recommendation: every observed
> (user, movie) rating is treated as a positive interaction, since the act of watching and rating
> reflects user interest. Unobserved pairs are treated as unknown, not negative."
>
> **补充分析建议**：把测试目标按星级分组，比较各模型在 `target=5星` 与 `target=2星` 上的命中率差异。
> 这既保留"高分更该推荐"的直觉，又不破坏评估定义，成本极低。

### 5.2 D2：为什么剔除评分 <5 的物品

物品要过**两级过滤**，且**必须在 LOO 切分之前**执行：

```
目录 3,883 部
  │  ① 剔除 177 部零评分          → 3,706 部
  │  ② 剔除评分 <5 的 290 部长尾   → 3,416 部   ← N_ITEMS
  ▼
LOO 切分（此时每人仍有 ≥18 条，无人被淘汰）
```

| 过滤方式 | 保留物品 | 丢弃评分行 | 影响的用户 |
|---|---:|---:|---|
| 不过滤（只要被评过） | 3,706 | 0 | 0 |
| **`≥5`** ✅ | **3,416** | **598（0.06%）** | **0** |
| `≥10` | 3,260 | — | — |
| `≥20` | 3,043 | — | — |

**为什么第①级（剔除 177 部零评分）**：它们是**"完全没有观测"**，不是"评分为 0"。
ML-1M 量表是 1–5，**0 是非法值**。

| 若填 0 或填均值 | 后果 |
|---|---|
| 需伪造 `6040 × 177 = 1,069,080` 条记录 | 比真实数据（1,000,209 条）**还多** |
| 若填均值 3.58，每部获得 6,040 条评分 | **热度超过真实冠军 `American Beauty (1999)` 的 3,428**，冷门片秒变"全站最热" |
| 均值从全体评分算出 | **数值泄漏**（含验证/测试期数据） |
| 6040 个用户填同一常数 | 零信息量，只增加偏置 |

**为什么第②级（剔除 `<5`）**：只剩 1–4 条评分的物品，共现结构近乎随机，
会严重污染 Item-CF 的相似度矩阵。代价只有 598 行（0.06%），换来的是稳定的相似度。

**⚠️ 顺序陷阱**：必须"**先过滤、后 LOO**"。
若反过来（先 LOO 再过滤），会有 **6 个用户**的测试目标被删掉；
先过滤后 LOO 则 **6040 个用户全部保留**。

**⚠️ 候选池污染**：被过滤的物品**绝对不能出现在候选池里**，
否则它们是"免费负样本"，会系统性**抬高** Recall@10。`build_canon` 里要 `assert` 拦住。

> 📌 **报告写法**：写出链路"目录 3,883 → 剔除零评分 3,706 → 剔除长尾 3,416"并说明理由。
> 这一句本身就是「数据集信息 20%」的加分项。

### 5.3 D4：为什么"全量排序为主 + 101 候选为辅"

**全量排序**：给每个用户对全部 **3,416** 部电影打分，在全部物品里排 top-10。

实测成本：

```
6040 × 3416 = 20,632,640 个分数
一次矩阵乘 ≈ 1.3 GFLOPs
float32 分数矩阵 = 82.5 MB
```

**三个候选算法（Item-CF、LightFM、LRURec）都能在 1 秒内算完。**

所以全量排序是更好的选择：无负采样歧义、更接近真实场景、可与 full-ranking 论文直接对比。

**101 候选**（1 正 + 100 固定种子负样本）仍然生成，但**只作辅助表**，
用于与采用 1+99 负采样的 top-N 论文口径对齐。

**关键约束**：主表与副表**必须来自同一个分数矩阵** —— 算一次全量分数，取子集得 101 候选结果。
禁止两套独立实现。

> **诚实说明**：LOO 下每用户只有 1 个正样本，因此 `Recall@10 ≡ HR@10`（命中即 1）。
> 全量排序下这个值有真实区分度，比 1+100 时的虚高值更可信。
> 报告里要写清"我们采用 full-ranking 评估，数值与 1+99 负采样的论文不可直接比较"。

### 5.4 D3：序列最大长度（已定稿：50）

只影响序列模型（LRURec），不影响 Item-CF 和 LightFM。
LRURec 已在全新20组 val 联合调参后选定 `maxlen=50`；旧10组结果不计。训练窗口覆盖全部 train 相邻转换，推理保留最后 50 条 train 交互。主表 test 不追加 val，具体协议见 [LRUREC.md](LRUREC.md)。参考数据：

| maxlen | 会被截断的用户 | 保留的信息量 |
|---:|---:|---:|
| 50 | 4,166（69.0%） | 27.1% |
| 100 | 2,857（47.3%） | 44.6% |
| **200** | 1,561（25.8%） | 66.0% |
| 500 | 393（6.5%） | 90.9% |

---

## 6. 数据契约：canon 规格 ⭐

`build_canon.py` 已跑完，以下文件已生成在 `data/canon/`。**这是唯一允许的数据来源。**

| 文件 | 内容 | 规模 | 体积 |
|---|---|---:|---:|
| `ratings` | `uid, iid, rating, ts, split` | 999,611 行 | 4.69 MB |
| `users` | `uid, gender, age_code, age_group, occupation_code, occupation, zip` | 6,040 行 | 0.08 MB |
| `movies` | `iid, raw_movie_id, title, year, genres(list[str])` | **3,416 行** | 0.11 MB |
| `eval_candidates` | `uid, iid, is_pos` — 每用户 101 行 | 610,040 行（辅助表） | 0.96 MB |
| `id_map.json` | 原始 ID ↔ 新 ID 映射 + 生成参数 | — | 0.17 MB |

### 6.0 存储格式与读取方式

扩展名由 `contract.CANON_FORMAT` **自动决定**：装好 `pyarrow` 或 `fastparquet` 时写 `.parquet`，
否则自动退化为 `.pkl`。当前环境为 **parquet**（pickle 下 ratings 为 20.0 MB，parquet 为 4.69 MB）。

**读取必须走统一入口，不要直接 `pd.read_parquet`：**

```python
from src.contract import read_canon_table
ratings = read_canon_table("ratings")
movies  = read_canon_table("movies")
```

原因见 §6.6 —— 直接读会拿到与 pickle 不一致的列类型。

### 6.1 id 约定

```
uid : 0 .. 6039      ← 原始 UserID 减 1
iid : 0 .. 3415      ← 原始 MovieID 重新编号（因为 MovieID 有 69 个空洞）
```

- `uid` 与 `iid` 都是 **0-based 连续**。
- 序列模型使用时需要 `iid + 1`（LRURec 要求 id 从 1 开始、`0` 留给 padding），
  **在自己的 adapter 里做这个变换，不要改 canon 文件**。
- `movies.parquet` 保留 `raw_movie_id`，方便和 IMDB / `movies.dat` 对照。

### 6.2 切分结果（LOO）

| split | 行数 | 说明 |
|---|---:|---|
| `train` | **987,531** | 每用户除最后两条以外 |
| `val` | **6,040** | 每人倒数第二条 |
| `test` | **6,040** | 每人最后一条 |

### 6.3 三条纪律

| ❌ 不要 | ✅ 应该 |
|---|---|
| 自己 `read_csv` 原始 `.dat` 然后切分 | `read_parquet('data/canon/ratings.parquet')` 后按 `split` 列过滤 |
| 自己写 Recall@10 函数 | 统一调 `src/evaluate.py`，它是指标**唯一出口** |
| 改 canon 文件（增删列、改 id） | 在 adapter 里做局部转换 |

> 原因：三个模型的指标必须**可比**。一旦用了不同的切分或不同的指标实现，
> 对比表就失效了，而「对比三算法」占了 20% 的分。

### 6.4 训练时只能用 `train`

**铁律**：任何特征、任何统计量（流行度、均值、用户活跃度、物品被评次数）**只能从 `train` 算**。

谁用了 `val`/`test`，就是**数据泄漏**。建议在每个特征函数里加断言：

```python
assert df["split"].isin(["train"]).all(), "泄漏了！只能用 train 算特征"
```

### 6.5 ⚠️ val 与 test 的难度差异（影响**所有**模型的调参）

LOO 下 `val` 是每人倒数第二条、`test` 是最后一条。**两者难度不同**，
实测（同一模型、同一历史、同一排除集合，只换目标物品）：

| 目标 | Recall@10 |
|---|---:|
| `val` 目标 | 0.0786 |
| `test` 目标 | 0.0631 |
| `test` 正式口径（排除 train∪val） | 0.0641 |

差异约 **0.015，相当于 4 个标准误**，不是噪声。
原因是「用户的最后一条评分」更可能是探索性的新片，而倒数第二条更接近其已有口味。

**因此：**
1. 超参只能在 `val` 上选，**绝不能看 test**。
2. **`val` 的绝对值系统性偏高约 0.015**，不能把它外推成 test 的预期结果。
3. 报告里只报 `test` 数字；`val` 只用于「哪个配置更好」的相对比较。

> 对序列模型和 LightFM 同样成立——这是切分方式的属性，不是某个模型的问题。

### 6.6 ⚠️ 不要绕过 `read_canon_table()` 直接读 parquet

`movies.genres` 在 parquet 里是 `list<string>`，pyarrow 读回来是 **`numpy.ndarray`** 而不是 Python `list`。
`read_canon_table()` 已统一转回 `list`。

绕过它直接用 `pd.read_parquet()` 的后果：`isinstance(g, list)` 返回 `False`、
`g.append(...)` 报错、两种存储格式下同一份代码行为不一致。

**案例**：这个 bug 是在切换 pickle → parquet 时被发现的，见 §9 变更记录。

---

## 7. 常见问题 FAQ

**Q：我能直接用 `pd.read_csv('ml-1m/ratings.dat')` 吗？**
不能。三个坑同时踩：分隔符是 `::`、没有表头、`movies.dat` 编码是 latin-1。正确写法：

```python
ratings = pd.read_csv('ml-1m/ratings.dat', sep='::', header=None, engine='python',
                      names=['UserID','MovieID','Rating','Timestamp'])
users   = pd.read_csv('ml-1m/users.dat',   sep='::', header=None, engine='python',
                      names=['UserID','Gender','Age','Occupation','Zip'],
                      dtype={'Zip': str})
movies  = pd.read_csv('ml-1m/movies.dat',  sep='::', header=None, engine='python',
                      names=['MovieID','Title','Genres'], encoding='latin-1')
```

**Q：`sep='::'` 会不会有问题？**
`engine='python'` 必加，否则 C engine 会警告或行为不一致。

**Q：`ratings.dat` 里有重复吗？**
没有，实测 0 条重复 `(user, movie)` 对。不用去重。

**Q：要不要做 5-core 过滤？**
✅ 已定：用户侧**不**过滤（过滤后每人仍有 ≥18 条）；物品侧**过滤 `≥5`**，剩 3,416 部。

**Q：评分要标准化吗？**
评分预测（RMSE）需要 `(r - 1) / 4` 到 [0,1]。**排序任务（Recall / NDCG）不需要**，
当前协议下 `rating` 数值不参与训练和评估，只作报告期分组解读。

**Q：那 177 部零评分电影能拿来做冷启动实验吗？**
**不能。** 它们是"绝对冷启动"（永远没有任何交互），任何算法都推不出来，
实验结果会是"所有模型都是 0 分"，没有信息量。
正确做法是"遮挡一部分**有**评分的物品"来模拟冷启动（见 [FRAMEWORK.md](FRAMEWORK.md) §5.1）。

**Q：`ml-1m.zip` 要不要提交到 git？**
**不要。** 官方许可禁止再分发。`data/raw/` 和 `ml-1m.zip` 都要进 `.gitignore`。

**Q：`rating` 列还有用吗？**
有，但**不参与训练和评估**。用途：① 报告里分析"模型在 5 星测试目标 vs 2 星测试目标上的差异"
② 解释为什么选择隐式反馈协议。

---

## 8. 许可与引用（提交前必查）

- ❌ **不得再分发原始数据** → `ml-1m.zip` 与 `ml-1m/` 必须 gitignore
- ❌ 不得暗示 University of Minnesota / GroupLens 背书
- ❌ 不得商用
- ✅ **必须致谢引用**，格式如下（直接抄）：

> F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context.
> *ACM Transactions on Interactive Intelligent Systems (TiiS)* 5, 4, Article 19 (December 2015), 19 pages.
> DOI: [10.1145/2827872](https://doi.org/10.1145/2827872)

---

## 9. 变更记录

| 日期 | 变更 |
|---|---|
| 2026-10-28 | 建立文档；完成数据侦察与四项决策（D1 全部交互为正 / D2 物品 `≥5` 过滤 / D4 全量排序为主 / D3 暂缓）；`contract.py` + `build_canon.py` 完成并生成 canon（`N_ITEMS=3,416`，train/val/test = 987,531 / 6,040 / 6,040）；canon 存储切换为 parquet；修复 `genres` 被读成 ndarray 的问题；记录 val/test 难度差异（§6.5） |
| 2026-11-09 | LRURec 接入并完成**全新 20 组**验证集调参（旧 10 组不计）；D3 定稿 **`maxlen=50`**，训练/推理协议见 [LRUREC.md](LRUREC.md)；不修改 canon 切分。 |
