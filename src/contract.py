"""
cs5481 MovieLens-1M recommender
==============================

src/contract.py  —  单一真相源 (Single Source of Truth)
--------------------------------------------------------
本文件是整个项目**唯一**允许定义常量的地方。其他脚本一律 `from src.contract import ...`，
不允许在任何地方重复硬编码数字、路径、字段名或映射表。

设计约束
~~~~~~~~
1. **纯标准库**。不 import pandas / numpy / torch，因此任何脚本都能无副作用地导入它。
2. **不产生副作用**。除了 `ensure_dirs()` 被显式调用外，导入本模块不读写磁盘。
3. **数字全部来自实测**。每个常量都注明出处，改之前先看 DATA.md。

配套文档
~~~~~~~~
- `DATA.md`      数据处理说明（口径、坑、决策依据、canon 规格）
- `FRAMEWORK.md` 架构与分工

快速自检
~~~~~~~~
    python -m src.contract        # 打印当前生效的配置
"""

from __future__ import annotations

import re
from pathlib import Path

__all__ = [
    # 路径
    "REPO_ROOT", "RAW_DIR", "DATA_DIR", "CANON_DIR", "OUTPUT_DIR",
    "RAW_RATINGS", "RAW_USERS", "RAW_MOVIES",
    "CANON_RATINGS", "CANON_USERS", "CANON_MOVIES", "CANON_ID_MAP",
    "CANON_EVAL_CANDIDATES",
    # 原始文件格式
    "SEP", "MOVIES_ENCODING", "TITLE_YEAR_RE",
    "RATING_MIN", "RATING_MAX",
    # 规模
    "N_USERS", "N_ITEMS", "CATALOG_N_ITEMS", "RATED_N_ITEMS",
    "MIN_ITEM_FREQ", "DROP_ZERO_RATED_ITEMS",
    # 期望基线
    "EXPECTED",
    # 切分
    "SPLIT_TRAIN", "SPLIT_VAL", "SPLIT_TEST", "SPLITS", "EXPECTED_SPLIT_ROWS",
    # 标签
    "RELEVANCE_RULE", "relevance_label",
    # 评估
    "K", "N_NEG_CANDIDATES", "CANDIDATES_PER_USER",
    "EVAL_MODE_FULL", "EVAL_MODE_CANDIDATES", "EVAL_MODES", "METRICS",
    "random_baseline",
    # 随机种子
    "SEED", "SEEDS",
    # 字段字典
    "AGE_GROUP", "OCCUPATION", "GENRES", "GENRE_INDEX",
    "RATINGS_COLUMNS", "USERS_COLUMNS", "MOVIES_COLUMNS",
    # 工具
    "ensure_dirs", "describe", "canon_path", "write_canon_table", "read_canon_table",
    "CANON_FORMAT", "CANON_EXT",
]


# =============================================================================
# 1. 路径
# =============================================================================
# 约定：本文件位于 <repo>/src/contract.py，故 parents[1] 是仓库根目录。
REPO_ROOT: Path = Path(__file__).resolve().parents[1]

# 原始数据的候选位置。按顺序探测，取第一个含 ratings.dat 的目录。
# 目前数据在 <repo>/ml-1m/；DATA.md §1 规划迁到 <repo>/data/raw/ml-1m/。
_RAW_DIR_CANDIDATES: tuple[Path, ...] = (
    REPO_ROOT / "data" / "raw" / "ml-1m",
    REPO_ROOT / "data" / "raw",
    REPO_ROOT / "ml-1m",
)


def _find_raw_dir() -> Path:
    """探测原始数据目录。找不到时不抛异常，交给下游在需要读取时明确报错。"""
    for candidate in _RAW_DIR_CANDIDATES:
        if (candidate / "ratings.dat").is_file():
            return candidate
    return _RAW_DIR_CANDIDATES[-1]


RAW_DIR: Path = _find_raw_dir()

DATA_DIR: Path = REPO_ROOT / "data"
CANON_DIR: Path = DATA_DIR / "canon"
OUTPUT_DIR: Path = REPO_ROOT / "outputs"

# ---- 原始文件（官方 .dat，无表头，`::` 分隔）----
RAW_RATINGS: Path = RAW_DIR / "ratings.dat"
RAW_USERS: Path = RAW_DIR / "users.dat"
RAW_MOVIES: Path = RAW_DIR / "movies.dat"
RAW_README: Path = RAW_DIR / "README"

# ---- canon 存储格式（自适应）----
# parquet 需要 pyarrow 或 fastparquet；两者都没有时自动退化为 pickle，
# 保证流水线在最小依赖（pandas + numpy）下也能跑通。
# 装好 pyarrow 后无需改任何代码，重新运行 build_canon 即自动产出 .parquet。
def _parquet_engine_available() -> bool:
    import importlib.util
    return any(
        importlib.util.find_spec(m) is not None
        for m in ("pyarrow", "fastparquet")
    )


CANON_FORMAT: str = "parquet" if _parquet_engine_available() else "pickle"
CANON_EXT: str = ".parquet" if CANON_FORMAT == "parquet" else ".pkl"


def canon_path(stem: str) -> Path:
    """canon 表文件的路径（扩展名随 CANON_FORMAT 变化）。"""
    return CANON_DIR / f"{stem}{CANON_EXT}"


# ---- canon 派生物（冻结，所有模型只读这些）----
CANON_RATINGS: Path = canon_path("ratings")
CANON_USERS: Path = canon_path("users")
CANON_MOVIES: Path = canon_path("movies")
CANON_EVAL_CANDIDATES: Path = canon_path("eval_candidates")
CANON_ID_MAP: Path = CANON_DIR / "id_map.json"   # 元数据始终是 JSON


# =============================================================================
# 2. 原始文件格式
# =============================================================================
SEP: str = "::"                     # 三个 .dat 的分隔符（不是逗号！）
MOVIES_ENCODING: str = "latin-1"    # utf-8 会在 byte 3114 崩（DATA.md §4 坑 1）
TITLE_YEAR_RE: re.Pattern[str] = re.compile(r"\((\d{4})\)\s*$")  # 片名尾部年份

# 三个 .dat 均无表头，列名在此统一声明（DATA.md §4 坑 2）
RATINGS_COLUMNS: tuple[str, ...] = ("UserID", "MovieID", "Rating", "Timestamp")
USERS_COLUMNS: tuple[str, ...] = ("UserID", "Gender", "Age", "Occupation", "Zip")
MOVIES_COLUMNS: tuple[str, ...] = ("MovieID", "Title", "Genres")

RATING_MIN: int = 1
RATING_MAX: int = 5


# =============================================================================
# 3. 规模与过滤规则
# =============================================================================
# ---- 原始的实测规模 ----
CATALOG_N_ITEMS: int = 3_883   # movies.dat 行数（目录大小）
RATED_N_ITEMS: int = 3_706     # 至少被评分过 1 次的电影数 → 177 部零评分
N_USERS: int = 6_040           # users.dat 行数，也是 ratings 里的用户数

# ---- 两级物品过滤（见 DATA.md §5.2）----
DROP_ZERO_RATED_ITEMS: bool = True   # 第一级：剔除 177 部零评分
MIN_ITEM_FREQ: int = 5               # 第二级：剔除评分 < 5 的物品（290 部）

# ---- 过滤后的建模口径 ----
N_ITEMS: int = 3_416   # 3_883 - 177 - 290

# 注意：物品过滤**必须**在 LOO 切分之前执行。
# 顺序反了会有 6 个用户的 test target 被删掉；正序则 6040 个用户全部保留。
FILTER_BEFORE_SPLIT: bool = True


# =============================================================================
# 4. 期望基线（build_canon / 测试用 assert 校验）
# =============================================================================
# 全部来自对 ml-1m 的实测，见 DATA.md §2.3。
EXPECTED: dict[str, int | float] = {
    # 原始
    "raw_ratings":            1_000_209,
    "raw_users":                  6_040,
    "catalog_items":              3_883,
    "rated_items":                3_706,
    "zero_rated_items":             177,
    "duplicate_user_item_pairs":      0,
    # 过滤
    "dropped_low_freq_items":       290,
    "dropped_low_freq_ratings":     598,   # 被 ≥5 过滤丢弃的评分行
    # canon
    "canon_ratings":            999_611,
    "canon_users":                6_040,
    "canon_items":                3_416,
    "canon_mean_rating":          3.5820,
    # 切分
    "train_rows":               987_531,
    "val_rows":                   6_040,
    "test_rows":                  6_040,
    # 派生
    "candidates_rows":          610_040,   # 6040 × 101
}


# =============================================================================
# 5. 切分（Leave-One-Out）
# =============================================================================
SPLIT_TRAIN: str = "train"
SPLIT_VAL: str = "val"
SPLIT_TEST: str = "test"
SPLITS: tuple[str, str, str] = (SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST)

# LOO：每人最后一条 → test，倒数第二条 → val，其余 → train。
EXPECTED_SPLIT_ROWS: dict[str, int] = {
    SPLIT_TRAIN: EXPECTED["train_rows"],
    SPLIT_VAL: EXPECTED["val_rows"],
    SPLIT_TEST: EXPECTED["test_rows"],
}


# =============================================================================
# 6. 相关性标签
# =============================================================================
# 隐式反馈协议：**所有观测到的交互都是正样本**，rating 数值不参与正负判定。
# 未观测对 = 未知 (unknown)，**不是负样本**。
#
# 为什么不用 `rating >= 4`：LOO 的 test target 里有 41.84% 的评分 ≤ 3 星，
# 用 ≥4 当"相关"会让这部分测试目标变成"不该被推荐的东西"，指标与目标反向。
# 详见 DATA.md §5.1。
RELEVANCE_RULE: str = "all_observed"


def relevance_label(rating: float) -> int:
    """把一条评分映射成正负标签。

    在 `RELEVANCE_RULE == "all_observed"` 下恒返回 1；保留 `rating` 参数是为了
    ① 校验取值合法性，② 将来若要切换到 "rating>=4" 协议，接口不用改。
    """
    if not (RATING_MIN <= float(rating) <= RATING_MAX):
        raise ValueError(
            f"评分 {rating!r} 超出合法量表 [{RATING_MIN}, {RATING_MAX}]。"
            "ML-1M 没有 0 星，0 是非法值（DATA.md §5.2）。"
        )
    return 1


# =============================================================================
# 7. 评估协议
# =============================================================================
K: int = 10                     # Recall@10 / NDCG@10 / HR@10
N_NEG_CANDIDATES: int = 100     # 辅表：每用户 100 个固定种子负样本
CANDIDATES_PER_USER: int = 1 + N_NEG_CANDIDATES  # = 101（1 正 + 100 负）

EVAL_MODE_FULL: str = "full"              # 主表：全量排序（3,416 物品全排）
EVAL_MODE_CANDIDATES: str = "candidates"  # 副表：101 候选
EVAL_MODES: tuple[str, str] = (EVAL_MODE_FULL, EVAL_MODE_CANDIDATES)

METRICS: tuple[str, str, str] = ("recall", "ndcg", "hr")

# 评估时是否排除该用户 train 里已交互过的物品（top-N 的标准做法）
EXCLUDE_TRAIN_ITEMS: bool = True


def random_baseline(mode: str) -> float:
    """随机打分模型的理论 Recall@K / HR@K 期望值，用于自检指标实现是否正确。

    LOO 下每用户只有 1 个正样本，故 Recall@K ≡ HR@K。
    - full 模式：从 N_ITEMS 里随便排，命中率 ≈ K / N_ITEMS
    - candidates 模式：从 101 个候选里排，命中率 ≈ K / 101
    """
    if mode == EVAL_MODE_FULL:
        return K / N_ITEMS
    if mode == EVAL_MODE_CANDIDATES:
        return K / CANDIDATES_PER_USER
    raise ValueError(f"未知评估模式 {mode!r}，应为 {EVAL_MODES}")


# =============================================================================
# 8. 随机种子
# =============================================================================
SEED: int = 42
SEEDS: dict[str, int] = {
    "global": SEED,
    "negative_sampling": SEED,     # 101 候选的负样本采样
    "cold_start_split": SEED,      # 冷启动实验的物品遮挡（FRAMEWORK.md §5.1）
    "model_init": SEED,            # LightFM / LRURec 初始化
}


# =============================================================================
# 9. 字段字典（来自官方 README 第 103–135 行）
# =============================================================================
AGE_GROUP: dict[int, str] = {
    1: "Under 18",
    18: "18-24",
    25: "25-34",
    35: "35-44",
    45: "45-49",
    50: "50-55",
    56: "56+",
}

OCCUPATION: dict[int, str] = {
    0: "other",
    1: "academic/educator",
    2: "artist",
    3: "clerical/admin",
    4: "college/grad student",
    5: "customer service",
    6: "doctor/health care",
    7: "executive/managerial",
    8: "farmer",
    9: "homemaker",
    10: "K-12 student",
    11: "lawyer",
    12: "programmer",
    13: "retired",
    14: "sales/marketing",
    15: "scientist",
    16: "self-employed",
    17: "technician/engineer",
    18: "tradesman/craftsman",
    19: "unemployed",
    20: "writer",
}

# 18 个类型，顺序固定（LightFM 的 item_features 列序依赖它）
GENRES: tuple[str, ...] = (
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
)
GENRE_INDEX: dict[str, int] = {g: i for i, g in enumerate(GENRES)}


# =============================================================================
# 10. 工具
# =============================================================================
def ensure_dirs() -> None:
    """创建所有输出目录。仅在需要写盘的脚本里显式调用。"""
    for path in (CANON_DIR, OUTPUT_DIR, OUTPUT_DIR / "metrics", OUTPUT_DIR / "scores", OUTPUT_DIR / "top10"):
        path.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 11. canon 读写（格式随 CANON_FORMAT 自适应）
# =============================================================================
# 这两个函数是**唯一**允许读写 canon 表的入口。下游脚本不要直接调
# `pd.read_parquet` / `pd.read_pickle`，否则格式切换时会全线崩。
def write_canon_table(df, stem: str) -> Path:
    """把 DataFrame 写进 canon。`stem` 是不带扩展名的表名，如 "ratings"。"""
    import pandas as pd  # 延迟导入：保持模块导入时零重依赖

    path = canon_path(stem)
    path.parent.mkdir(parents=True, exist_ok=True)
    if CANON_FORMAT == "parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_pickle(path)
    return path


def _normalize_list_columns(df):
    """把 parquet 读回来的 `list<string>` 列（实际是 numpy.ndarray）还原成 Python list。

    为什么必须做：pickle 保留 `list`，parquet（pyarrow）返回 `np.ndarray`。
    两者在 `isinstance(x, list)`、`.append()`、序列化等场景行为不同，
    会导致"同一份 canon 在两种格式下跑出不同结果"。统一在读取出口抹平差异。
    """
    for col in df.columns:
        if df[col].dtype != "object":
            continue
        head = df[col].iloc[0] if len(df) else None
        if head is None or isinstance(head, list):
            continue
        if hasattr(head, "tolist") and not isinstance(head, (str, bytes)):
            df[col] = df[col].map(
                lambda v: v.tolist() if hasattr(v, "tolist") else v
            )
    return df


def read_canon_table(stem: str):
    """读 canon 表。缺文件时给出可操作的报错，而不是 pandas 的底层异常。

    返回的 DataFrame 在 parquet / pickle 两种格式下**完全一致**（含 list 列）。
    """
    import pandas as pd  # 延迟导入

    path = canon_path(stem)
    if not path.is_file():
        raise FileNotFoundError(
            f"缺少 canon 文件 {path}。请先运行：python -m src.build_canon"
        )
    df = pd.read_parquet(path) if CANON_FORMAT == "parquet" else pd.read_pickle(path)
    return _normalize_list_columns(df)


def describe() -> str:
    """返回人类可读的配置摘要。"""
    lines = [
        "=" * 68,
        "CS5481 MovieLens-1M · 当前生效配置",
        "=" * 68,
        f"REPO_ROOT      {REPO_ROOT}",
        f"RAW_DIR        {RAW_DIR}   (exists={RAW_DIR.is_dir()})",
        f"CANON_DIR      {CANON_DIR}   (exists={CANON_DIR.is_dir()})",
        f"CANON_FORMAT   {CANON_FORMAT}   (ext={CANON_EXT}"
        f"{'' if CANON_FORMAT == 'parquet' else ', 未检测到 pyarrow/fastparquet 已降级为 pickle'})",
        "",
        f"N_USERS        {N_USERS}",
        f"N_ITEMS        {N_ITEMS}   (catalog {CATALOG_N_ITEMS} - zero-rated "
        f"{EXPECTED['zero_rated_items']} - low-freq {EXPECTED['dropped_low_freq_items']})",
        f"物品过滤       zero-rated={DROP_ZERO_RATED_ITEMS}, min_freq>={MIN_ITEM_FREQ}",
        f"过滤顺序       filter before split = {FILTER_BEFORE_SPLIT}",
        "",
        f"切分           Leave-One-Out  train={EXPECTED_SPLIT_ROWS[SPLIT_TRAIN]:,}  "
        f"val={EXPECTED_SPLIT_ROWS[SPLIT_VAL]:,}  test={EXPECTED_SPLIT_ROWS[SPLIT_TEST]:,}",
        f"标签协议       {RELEVANCE_RULE}  (所有观测交互均为正，rating 不参与判定)",
        "",
        f"评估 K         {K}",
        f"评估模式       primary = {EVAL_MODE_FULL} (全量 {N_ITEMS} 物品)",
        f"               secondary = {EVAL_MODE_CANDIDATES} ({CANDIDATES_PER_USER} 候选/用户)",
        f"随机基线       full ≈ {random_baseline(EVAL_MODE_FULL):.6f}   "
        f"candidates ≈ {random_baseline(EVAL_MODE_CANDIDATES):.4f}",
        "",
        f"SEED           {SEED}",
        f"类型数         {len(GENRES)}     职业数 {len(OCCUPATION)}     年龄档 {len(AGE_GROUP)}",
        "=" * 68,
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
