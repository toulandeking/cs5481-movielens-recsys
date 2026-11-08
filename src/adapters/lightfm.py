"""
cs5481 MovieLens-1M recommender
==============================

src/adapters/lightfm.py  —  课外算法：LightFM（hybrid matrix factorisation）
--------------------------------------------------------------------------

算法定义
~~~~~~~~
LightFM 把"用户 i"和"物品 j"都表示成**特征嵌入之和**：

    u_i = Σ_{f ∈ U_i} e_f      i_j = Σ_{g ∈ I_j} e_g
    score(i, j) = u_i · i_j + b_i + b_j

其中 `U_i` 是用户 i 命中的用户特征（性别 / 年龄档 / 职业），`I_j` 是物品 j 命中的
物品特征（18 个类型多热 + 10 年一个的年份桶）。相比纯 MF，多出来的这层让模型具备
**冷启动 / 跨用户泛化**能力：特征相同的用户共享同一批嵌入，长尾片靠类型与年代获得
合理的先验。

损失函数用 `warp`（Bayesian Personalised Ranking 的加权近似），本项目是隐式反馈
协议（`contract.RELEVANCE_RULE = "all_observed"`）：交互矩阵 data 全 1，``rating``
数值**不参与**训练，也不做 `sample_weight`。

特征矩阵（见 `build_user_features` / `build_item_features`）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
::

    user_features  (6040, 30) = [gender 2] + [age_group 7] + [occupation_code 21]
    item_features  (3416, 28) = [genres 18] + [year 的 10 年桶 10]

行号 == uid / iid（直接拿 id 数组当 COO 行号，不做 reindex），数据只来自 canon 的
`users` / `movies` 两张静态画像表，与 split 无关，不可能泄漏 val/test。

打分：为什么用表示法而不是 `model.predict`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
`model.predict` 每次调用都要重算一遍特征嵌入之和，逐用户铺满 3416 个物品会把
6010 万次点积重复做一遍。直接取表示做一次矩阵乘法：

    ub, uw = model.get_user_representations(user_features)   # 注意顺序是 (bias, emb)
    ib, iw = model.get_item_representations(item_features)
    S = uw @ iw.T + ub[:, None] + ib[None, :]

**bias 必须加**，否则与 `predict` 不一致（实测差异集中在偏置项上）。
`get_*_representations` 返回的是 `(bias, embeddings)` 二元组、且参数名是 `features=`，
不接受 `user_features=` —— 这两点都与 `predict` 的签名不同，容易写错。

已知限制（Windows）
~~~~~~~~~~~~~~~~~~
1. `loss="warp"` / `"bpr"` 在 Windows 上会让进程直接消失（退出码 `0xC0000005`，
   栈落在 `fit_warp`），见 lightfm issue #720；`loss="logistic"` 不会崩，但在 LOO
   协议下把留出正例当负样本压制，test full Recall@10 ≈ 0.003（比随机还差），
   不可作为交付结果。**正式的 warp 结果请在 Linux（Colab / WSL）上跑。**
2. Windows 的官方构建**不含 OpenMP**（`setup.py` 里 `not sys.platform.startswith("win")`），
   `num_threads` 会被忽略，只有单线程 —— 不是本项目的 bug。

用法
~~~~
    python -m src.adapters.lightfm --eval                    # 训练 + test 评估 + 导出 top10
    python -m src.adapters.lightfm --top --uid 42            # 看某个用户的前 10 推荐
    python -m src.adapters.lightfm --eval --loss logistic    # Windows 上冒烟用
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import sparse

from src.adapters.base import AdapterBase, _check_matrix
from src.contract import (
    AGE_GROUP,
    GENRES,
    GENRE_INDEX,
    K,
    N_ITEMS,
    N_USERS,
    OCCUPATION,
    OUTPUT_DIR,
    SEED,
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VAL,
    read_canon_table,
)

__all__ = [
    "LightFMAdapter",
    "USER_FEATURE_NAMES",
    "ITEM_FEATURE_NAMES",
    "YEAR_BUCKETS",
]

# =============================================================================
# 列布局
# =============================================================================
# gender：2 列。ML-1M 只有 M/F 两种取值（DATA.md §1.2：M 4331 / F 1709）。
GENDER_CODES: tuple[str, ...] = ("M", "F")

# age_group：7 列，列序 = `contract.AGE_GROUP` 的键序（1,18,25,35,45,50,56）。
# 用**标签**匹配（"Under 18" / "18-24" / …），不用 age_code 的数值。
AGE_LABELS: tuple[str, ...] = tuple(AGE_GROUP[k] for k in AGE_GROUP)

# occupation：21 列，列序 = `contract.OCCUPATION` 的键序（0–20），按类别码匹配。
OCCUPATION_CODES: tuple[int, ...] = tuple(OCCUPATION)

# year：10 年一桶，1910s … 2000s 共 10 桶（canon 实测 1919–2000，DATA.md §1.3）。
YEAR_MIN: int = 1910
YEAR_MAX: int = 2000
YEAR_BUCKETS: tuple[int, ...] = tuple(range(YEAR_MIN, YEAR_MAX + 1, 10))

USER_FEATURE_NAMES: tuple[str, ...] = (
    *(f"gender={g}" for g in GENDER_CODES),
    *(f"age_group={a}" for a in AGE_LABELS),
    *(f"occupation_code={c}" for c in OCCUPATION_CODES),
)

ITEM_FEATURE_NAMES: tuple[str, ...] = (
    *(f"genre={g}" for g in GENRES),
    *(f"year={d}s" for d in YEAR_BUCKETS),
)

N_USER_FEATURES: int = len(USER_FEATURE_NAMES)   # 2 + 7 + 21 = 30
N_ITEM_FEATURES: int = len(ITEM_FEATURE_NAMES)   # 18 + 10   = 28

# 支持的损失。warp / bpr 是排序损失（Windows 上会崩，见模块 docstring）；
# logistic 是点二分类损失，只在 Linux 冒烟或对照实验里用。
LOSSES: tuple[str, ...] = ("warp", "bpr", "logistic")


# =============================================================================
# 小工具
# =============================================================================
def _user_indicator(rows: np.ndarray) -> sparse.csr_matrix:
    """把一组 uid 压成 (N_USERS, 1) 的 0/1 指示列。

    `rows` 里的重复行号会被**累加**（同一 uid 重复出现会得到 2），所以调用方
    必须先保证源表主键唯一（下面两个 build 函数都有 assert）。
    """
    rows = np.asarray(rows, dtype=np.int64)
    data = np.ones(rows.size, dtype=np.float32)
    return sparse.csr_matrix(
        (data, (rows, np.zeros(rows.size, dtype=np.int64))),
        shape=(N_USERS, 1),
        dtype=np.float32,
    )


def _check_ids(ids: np.ndarray, n_rows: int, table: str, id_col: str) -> np.ndarray:
    """校验主键：唯一、非空、落在 0..n_rows-1 的连续区段内。"""
    ids = np.asarray(ids, dtype=np.int64)
    if ids.size == 0:
        raise ValueError(f"canon 表 {table!r} 为空，无法构造特征矩阵")
    if np.unique(ids).size != ids.size:
        raise ValueError(f"canon 表 {table!r} 的 {id_col} 有重复值，行号会错位")
    lo, hi = int(ids.min()), int(ids.max())
    if lo < 0 or hi >= n_rows:
        raise ValueError(
            f"canon 表 {table!r} 的 {id_col} 越界：[{lo}, {hi}] 超出 [0, {n_rows - 1}]"
        )
    return ids


def _presence_mask(ids: np.ndarray, n_rows: int) -> np.ndarray:
    """ids 里出现过的行号置 True。用于区分"缺特征的全零行"和"映射写错的全零行"。"""
    mask = np.zeros(n_rows, dtype=bool)
    mask[ids] = True
    return mask


# =============================================================================
# 适配器
# =============================================================================
class LightFMAdapter(AdapterBase):
    """LightFM（WARP 隐式反馈）适配器。

    所有超参都能在构造时覆盖，CLI 与 `load_adapter("lightfm", ...)` 共用同一套。
    """

    name: str = "LightFM"

    def __init__(
        self,
        no_components: int = 32,
        loss: str = "warp",
        epochs: int = 20,
        learning_rate: float = 0.05,
        item_alpha: float = 0.0,
        user_alpha: float = 0.0,
        max_sampled: int = 10,
        num_threads: int = 4,
        identity: bool = False,
    ) -> None:
        if loss not in LOSSES:
            raise ValueError(f"未知损失 {loss!r}，应为 {LOSSES}")
        self.no_components = int(no_components)
        self.loss = str(loss)
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.item_alpha = float(item_alpha)
        self.user_alpha = float(user_alpha)
        self.max_sampled = int(max_sampled)
        # 可选：在侧信息前拼上身份指示列（见 build_user_features 的说明）。
        # 默认 False —— 与轮 2 定下的 30 / 28 列口径一致。
        self.identity = bool(identity)
        # 显式 4 线程（TASKS.md §4.2 坑 3：默认取满核数，会和别的进程抢 CPU）。
        # 注意 Windows 官方构建不含 OpenMP，这个值会被忽略。
        self.num_threads = int(num_threads)

        # 运行时状态（先置空，fit() 后填充）
        self.model = None
        self.user_features = None
        self.item_features = None
        self.fit_seconds = None
        self.interactions_nnz = None
        self._scores = None

    # -------------------------------------------------------------------------
    # 特征矩阵
    # -------------------------------------------------------------------------
    @staticmethod
    def build_user_features(
        strict: bool = True, include_identity: bool = False
    ) -> sparse.csr_matrix:
        """构造用户特征矩阵，行号 == uid。

        列布局（侧信息共 30 列）：:

            0..1    gender=M, gender=F                     （2 列）
            2..8    age_group  ∈ AGE_GROUP 的 7 个标签      （7 列）
            9..29   occupation_code ∈ 0..20                （21 列）

        Args:
            strict: True 时校验"源表里存在的 uid 必须至少有一个非零特征，源表里
                不存在的 uid 必须全零"。canon 的 users 表覆盖全部 6040 个 uid，
                默认严格模式能在映射写错时立刻炸掉。
            include_identity: True 时在侧信息**前面**再拼 N_USERS 列身份指示
                （第 i 行第 i 列为 1）。这是 LightFM 官方 `build_user_features`
                的做法：只有侧信息时，u_i = e_gender + e_age + e_occ，6040 个用户
                只落成 241 种不同向量（平均 25 人共享一个表示），模型无法区分
                同一画像里的人；加上身份列后每个用户才有一个可训练的自由向量。
                列数从 30 变成 30 + 6040 = 6070，稀疏度几乎不变。

        Returns:
            `scipy.sparse.csr_matrix`，形状 `(N_USERS, 30)`（或 `(N_USERS, 6070)`），
            dtype float32。
        """
        users = read_canon_table("users")
        if len(users) == 0:
            raise ValueError("canon 表 'users' 为空，无法构造用户特征")

        uid = _check_ids(users["uid"].to_numpy(), N_USERS, "users", "uid")

        # astype("string") 而不是 str：NaN/None 会变成 <NA>，不会误命中任何标签。
        gender = users["gender"].astype("string").fillna("").str.upper().to_numpy()
        age_label = users["age_group"].astype("string").fillna("").to_numpy()
        occ_code = users["occupation_code"].to_numpy()

        blocks = [
            _user_indicator(uid[gender == "M"]),
            _user_indicator(uid[gender == "F"]),
        ]
        blocks += [_user_indicator(uid[age_label == label]) for label in AGE_LABELS]
        blocks += [_user_indicator(uid[occ_code == code]) for code in OCCUPATION_CODES]

        side = sparse.hstack(blocks, format="csr", dtype=np.float32)

        assert side.shape == (N_USERS, N_USER_FEATURES), (
            f"用户特征矩阵形状应为 {(N_USERS, N_USER_FEATURES)}，实际 {side.shape}"
        )
        assert side.nnz > 0, "用户特征矩阵全零，映射一定写错了"

        if strict:
            present = _presence_mask(uid, N_USERS)
            per_row = np.diff(side.indptr)
            missing = np.flatnonzero(present & (per_row == 0))
            if missing.size:
                raise ValueError(
                    f"{missing.size} 个用户一个特征都没命中（前几个 uid={missing[:5].tolist()}）。"
                    f"检查 age_group 标签是否与 contract.AGE_GROUP 取值一致、"
                    f"occupation_code 是否落在 0..20。"
                )
            stray = np.flatnonzero(~present & (per_row > 0))
            if stray.size:
                raise ValueError(
                    f"{stray.size} 个不在 users 表里的 uid 却有特征（前几个 {stray[:5].tolist()}）"
                )

        if not include_identity:
            return side

        # 身份列放最前面：第 i 行第 i 列为 1，行号仍然等于 uid。
        ident = sparse.identity(N_USERS, dtype=np.float32, format="csr")
        features = sparse.hstack([ident, side], format="csr", dtype=np.float32)
        assert features.shape == (N_USERS, N_USER_FEATURES + N_USERS), (
            f"带身份列的用户特征形状应为 {(N_USERS, N_USER_FEATURES + N_USERS)}，"
            f"实际 {features.shape}"
        )
        # 每行必然多一个身份非零值。
        assert (np.diff(features.indptr) == np.diff(side.indptr) + 1).all(), (
            "身份列没有给每个用户恰好加 1 个非零，行号可能错位"
        )
        return features

    @staticmethod
    def build_item_features(
        strict: bool = True, include_identity: bool = False
    ) -> sparse.csr_matrix:
        """构造物品特征矩阵，行号 == iid。

        列布局（侧信息共 28 列）：:

            0..17   genres 18 个类型多热（列序 = contract.GENRES）  （18 列）
            18..27  year 的 10 年桶，1910s … 2000s                （10 列）

        Args:
            strict: True 时校验"源表里存在的 iid 至少有 1 个类型 + 1 个年份桶"，
                源表里不存在的 iid 必须全零。
            include_identity: True 时在侧信息前面拼 N_ITEMS 列身份指示，
                列数变成 28 + 3416 = 3444（理由同 `build_user_features`）。

        Returns:
            `scipy.sparse.csr_matrix`，形状 `(N_ITEMS, 28)`（或 `(N_ITEMS, 3444)`），
            dtype float32。
        """
        movies = read_canon_table("movies")
        if len(movies) == 0:
            raise ValueError("canon 表 'movies' 为空，无法构造物品特征")

        iid = _check_ids(movies["iid"].to_numpy(), N_ITEMS, "movies", "iid")

        # --- genres：多热 ---------------------------------------------------
        genre_lists = movies["genres"].to_numpy()
        lengths = np.fromiter((len(g) for g in genre_lists), dtype=np.int64, count=iid.size)
        if (lengths == 0).any():
            bad = iid[lengths == 0][:5].tolist()
            raise ValueError(f"{int((lengths == 0).sum())} 部电影没有任何类型，前几个 iid={bad}")

        rows = np.repeat(iid, lengths)
        cols = np.fromiter(
            (GENRE_INDEX[g] for gl in genre_lists for g in gl),
            dtype=np.int64,
            count=int(lengths.sum()),
        )
        genre_mat = sparse.csr_matrix(
            (np.ones(rows.size, dtype=np.float32), (rows, cols)),
            shape=(N_ITEMS, len(GENRES)),
            dtype=np.float32,
        )
        # GENRE_INDEX 缺键会直接 KeyError，比静默丢列好；这里再确认一遍覆盖情况。
        unknown = {g for gl in genre_lists for g in gl if g not in GENRE_INDEX}
        if unknown:
            raise ValueError(f"movies.genres 里有未知类型 {sorted(unknown)}，未在 contract.GENRES 中")

        # --- year：10 年桶 one-hot ------------------------------------------
        year = movies["year"].to_numpy()
        if np.isnan(year.astype(np.float64)).any():
            raise ValueError(f"{int(np.isnan(year.astype(np.float64)).sum())} 部电影缺 year")
        year = year.astype(np.int64)
        decade = (year // 10) * 10
        if ((decade < YEAR_MIN) | (decade > YEAR_MAX)).any():
            bad = year[(decade < YEAR_MIN) | (decade > YEAR_MAX)][:5].tolist()
            raise ValueError(
                f"{int(((decade < YEAR_MIN) | (decade > YEAR_MAX)).sum())} 部电影年份落在 "
                f"{YEAR_MIN}s–{YEAR_MAX}s 之外，前几个 year={bad}；请同步扩展 YEAR_BUCKETS"
            )
        year_col = (decade - YEAR_MIN) // 10
        year_mat = sparse.csr_matrix(
            (np.ones(iid.size, dtype=np.float32), (iid, year_col)),
            shape=(N_ITEMS, len(YEAR_BUCKETS)),
            dtype=np.float32,
        )

        side = sparse.hstack([genre_mat, year_mat], format="csr", dtype=np.float32)

        assert side.shape == (N_ITEMS, N_ITEM_FEATURES), (
            f"物品特征矩阵形状应为 {(N_ITEMS, N_ITEM_FEATURES)}，实际 {side.shape}"
        )
        assert side.nnz > 0, "物品特征矩阵全零，映射一定写错了"

        if strict:
            present = _presence_mask(iid, N_ITEMS)
            per_row = np.diff(side.indptr)
            # 每部片至少 1 个类型 + 1 个年份桶 = 2 个非零。
            thin = np.flatnonzero(present & (per_row < 2))
            if thin.size:
                raise ValueError(
                    f"{thin.size} 部电影的物品特征少于 2 个非零（前几个 iid={thin[:5].tolist()}）"
                )
            stray = np.flatnonzero(~present & (per_row > 0))
            if stray.size:
                raise ValueError(
                    f"{stray.size} 个不在 movies 表里的 iid 却有特征（前几个 {stray[:5].tolist()}）"
                )

        if not include_identity:
            return side

        ident = sparse.identity(N_ITEMS, dtype=np.float32, format="csr")
        features = sparse.hstack([ident, side], format="csr", dtype=np.float32)
        assert features.shape == (N_ITEMS, N_ITEM_FEATURES + N_ITEMS), (
            f"带身份列的物品特征形状应为 {(N_ITEMS, N_ITEM_FEATURES + N_ITEMS)}，"
            f"实际 {features.shape}"
        )
        assert (np.diff(features.indptr) == np.diff(side.indptr) + 1).all(), (
            "身份列没有给每部片恰好加 1 个非零，行号可能错位"
        )
        return features

    # -------------------------------------------------------------------------
    # 训练
    # -------------------------------------------------------------------------
    def fit(self) -> None:
        """只用 `split == "train"` 训练。

        交互矩阵按隐式反馈构造：`data` 全 1，`rating` 不参与（也不作为
        `sample_weight`）—— 见 `contract.RELEVANCE_RULE = "all_observed"`。
        """
        from lightfm import LightFM

        ratings = read_canon_table("ratings")
        train = ratings.loc[ratings["split"] == SPLIT_TRAIN, ["uid", "iid"]]
        if len(train) == 0:
            raise ValueError("ratings 表里没有 split == 'train' 的行")

        uid = train["uid"].to_numpy(dtype=np.int64)
        iid = train["iid"].to_numpy(dtype=np.int64)
        interactions = sparse.csr_matrix(
            (np.ones(uid.size, dtype=np.float32), (uid, iid)),
            shape=(N_USERS, N_ITEMS),
            dtype=np.float32,
        )
        # 构造式里 (uid, iid) 重复会被 csr 悄悄求和成 2，导致"隐式权重"偏离 1。
        assert interactions.nnz == uid.size, (
            f"train 里出现重复 (uid, iid)：{uid.size} 行 → nnz={interactions.nnz}"
        )
        assert interactions.shape == (N_USERS, N_ITEMS)

        self.user_features = self.build_user_features(include_identity=self.identity)
        self.item_features = self.build_item_features(include_identity=self.identity)
        self.interactions_nnz = int(interactions.nnz)

        self.model = LightFM(
            no_components=self.no_components,
            loss=self.loss,
            learning_schedule="adagrad",
            learning_rate=self.learning_rate,
            max_sampled=self.max_sampled,
            user_alpha=self.user_alpha,
            item_alpha=self.item_alpha,
            random_state=SEED,
        )
        # num_threads 只存在于 fit()/predict() 的签名里，不是构造参数（1.17）。
        t0 = time.perf_counter()
        self.model.fit(
            interactions,
            user_features=self.user_features,
            item_features=self.item_features,
            epochs=self.epochs,
            num_threads=self.num_threads,
            verbose=False,
        )
        self.fit_seconds = time.perf_counter() - t0
        self._scores = None

    def _ensure_fit(self) -> None:
        if self.model is None:
            raise RuntimeError("LightFMAdapter 还没训练，请先调用 fit()")

    # -------------------------------------------------------------------------
    # 打分
    # -------------------------------------------------------------------------
    def score_matrix(self) -> np.ndarray:
        """(N_USERS, N_ITEMS) 完整分数矩阵，用表示法一次算出。

        注意 `get_user_representations` / `get_item_representations` 返回的是
        `(bias, embeddings)`——bias 在前；参数名是 `features=`，不是 `user_features=`。
        """
        self._ensure_fit()
        if self._scores is None:
            user_biases, user_embs = self.model.get_user_representations(self.user_features)
            item_biases, item_embs = self.model.get_item_representations(self.item_features)

            scores = (user_embs @ item_embs.T).astype(np.float32, copy=False)
            scores += user_biases.astype(np.float32)[:, None]
            scores += item_biases.astype(np.float32)[None, :]
            self._scores = _check_matrix(scores)
        return self._scores

    def predict_scores(self, uid: np.ndarray, iid: np.ndarray) -> np.ndarray:
        """与 `(uid, iid)` 等长的分数；走缓存的分数矩阵，避免逐对调 `model.predict`。"""
        return self.score_matrix()[uid, iid]


# =============================================================================
# CLI（沿用 itemcf.py 的写法：手写 flag 解析，不用 argparse）
# =============================================================================
USAGE = """用法:
  python -m src.adapters.lightfm --eval [超参...]        训练 + test 评估 + 导出 top10
  python -m src.adapters.lightfm --top --uid N [超参...]  某个用户的前 10 推荐
  python -m src.adapters.lightfm --tune [--resume]        在 val 上扫超参网格
  python -m src.adapters.lightfm --analyze                错误分析表 + 两张图（重训同一模型）

超参（默认值来自 outputs/metrics/lightfm_params.json，显式 flag 优先）:
  --components INT    隐向量维度 (默认 32)
  --loss STR          warp | bpr | logistic (默认 warp)
  --epochs INT        训练轮数 (默认 20)
  --lr FLOAT          learning_rate (默认 0.05)
  --item-alpha FLOAT  物品嵌入 L2 (默认 0.0)
  --user-alpha FLOAT  用户嵌入 L2 (默认 0.0)
  --max-sampled INT   WARP 每个正样本最多采几个负样本 (默认 100)
  --threads INT       num_threads (默认 4；Windows 构建无 OpenMP，会被忽略)
  --identity          特征里拼身份指示列 (6040+30 / 3416+28 列；默认按 params.json)
  --from-params       强制要求 outputs/metrics/lightfm_params.json 存在
"""


def _flag(argv: list[str], name: str) -> bool:
    """开关型 flag（无值）。"""
    return name in argv


def _opt(argv: list[str], flag: str, cast, default=None):
    """读 `--flag value`。与 itemcf.py 的解析风格保持一致。"""
    if flag not in argv:
        return default
    i = argv.index(flag)
    if i + 1 >= len(argv):
        raise SystemExit(f"{flag} 需要一个值\n\n{USAGE}")
    try:
        return cast(argv[i + 1])
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"{flag} 的取值 {argv[i + 1]!r} 无法转成 {cast.__name__}: {exc}")


PARAMS_PATH = OUTPUT_DIR / "metrics" / "lightfm_params.json"


def _lightfm_version() -> str:
    try:
        from importlib.metadata import version

        return version("lightfm")
    except Exception:                                    # pragma: no cover
        try:
            import lightfm

            return str(getattr(lightfm, "__version__", "unknown"))
        except Exception:
            return "unknown"


def _load_params_defaults(required: bool = True) -> dict:
    """把落盘的超参当作默认值。

    `--eval` / `--top` 在 `lightfm_params.json` 存在时**默认**读它：轮 4 已经把配置
    定稿（组 16：c=32 / warp / ep20 / ms100 / identity），轮 5 的 test 只能跑一次，
    不能再因为忘记加 flag 而用错超参。`--from-params` 退化成"强制要求该文件"。
    """
    if not PARAMS_PATH.exists():
        if not required:
            return {}
        raise SystemExit(f"{PARAMS_PATH} 不存在，无法从定稿配置读超参（先跑 --tune）")
    payload = json.loads(PARAMS_PATH.read_text(encoding="utf-8"))
    return dict(payload.get("params", {}))


def _build_model(argv: list[str]) -> LightFMAdapter:
    force = _flag(argv, "--from-params")
    d = _load_params_defaults() if (force or PARAMS_PATH.exists()) else {}
    return LightFMAdapter(
        no_components=_opt(argv, "--components", int, d.get("no_components", 32)),
        loss=_opt(argv, "--loss", str, d.get("loss", "warp")),
        epochs=_opt(argv, "--epochs", int, d.get("epochs", 20)),
        learning_rate=_opt(argv, "--lr", float, d.get("learning_rate", 0.05)),
        item_alpha=_opt(argv, "--item-alpha", float, d.get("item_alpha", 0.0)),
        user_alpha=_opt(argv, "--user-alpha", float, d.get("user_alpha", 0.0)),
        max_sampled=_opt(argv, "--max-sampled", int, d.get("max_sampled", 10)),
        num_threads=_opt(argv, "--threads", int, d.get("num_threads", 4)),
        identity=_flag(argv, "--identity") or bool(d.get("identity", False)),
    )


def _describe(model: LightFMAdapter) -> str:
    return (
        f"c={model.no_components}  loss={model.loss}  epochs={model.epochs}  "
        f"lr={model.learning_rate}  item_alpha={model.item_alpha}  "
        f"user_alpha={model.user_alpha}  max_sampled={model.max_sampled}  "
        f"threads={model.num_threads}"
    )


def _params_payload(model: LightFMAdapter, extra: dict | None = None) -> dict:
    n_uf = (
        int(model.user_features.shape[1])
        if model.user_features is not None
        else N_USER_FEATURES + (N_USERS if model.identity else 0)
    )
    n_if = (
        int(model.item_features.shape[1])
        if model.item_features is not None
        else N_ITEM_FEATURES + (N_ITEMS if model.identity else 0)
    )
    payload = {
        "model": model.name,
        "seed": SEED,
        "params": {
            "no_components": model.no_components,
            "loss": model.loss,
            "epochs": model.epochs,
            "learning_rate": model.learning_rate,
            "item_alpha": model.item_alpha,
            "user_alpha": model.user_alpha,
            "max_sampled": model.max_sampled,
            "num_threads": model.num_threads,
            "identity": model.identity,
        },
        "features": {
            "n_user_features": n_uf,
            "n_item_features": n_if,
        },
        "fit_seconds": round(model.fit_seconds or 0.0, 2),
        "interactions_nnz": model.interactions_nnz,
        "n_users": N_USERS,
        "n_items": N_ITEMS,
        "lightfm_version": _lightfm_version(),
    }
    if extra:
        payload.update(extra)
    return payload


def _fit_and_report(model: LightFMAdapter) -> None:
    print(f"\n[训练] {_describe(model)}")
    model.fit()
    print(f"  交互矩阵 nnz={model.interactions_nnz:,}（只用 train）")
    print(f"  用户特征 {model.user_features.shape}  nnz={model.user_features.nnz:,}")
    print(f"  物品特征 {model.item_features.shape}  nnz={model.item_features.nnz:,}")
    print(f"  训练耗时 {model.fit_seconds:.1f}s  num_threads={model.num_threads}")


def _write_params(model: LightFMAdapter, extra: dict | None = None) -> None:
    out = OUTPUT_DIR / "metrics"
    out.mkdir(parents=True, exist_ok=True)
    p = out / "lightfm_params.json"

    # `--tune` 写进去的 selection / tuning 是选配置的依据，`--eval` / `--top` /
    # `--analyze` 只是复用配置，不该把这份出处擦掉（保留旧值，新值优先）。
    keep = {}
    if p.is_file():
        try:
            old = json.loads(p.read_text(encoding="utf-8"))
            keep = {k: old[k] for k in ("selection", "tuning") if k in old}
        except (OSError, json.JSONDecodeError):
            keep = {}
    keep.update(extra or {})

    p.write_text(
        json.dumps(_params_payload(model, keep or None), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"  超参 → {p.relative_to(OUTPUT_DIR.parent)}")


def _export_top10(model: LightFMAdapter) -> None:
    df = model.export_top10()        # 内部已排除该用户 train ∪ val ∪ test 的影片
    d = OUTPUT_DIR / "top10"
    d.mkdir(parents=True, exist_ok=True)
    p = d / "lightfm.parquet"
    df.to_parquet(p, index=False)
    print(f"  top-10 → {p.relative_to(OUTPUT_DIR.parent)}  ({len(df):,} 行)")


# =============================================================================
# 调参（只在 val 上跑，test 一次都不碰）
# =============================================================================
TUNING_STEM = "lightfm_tuning_val"
TUNING_COLUMNS = (
    "group",
    "phase",
    "no_components",
    "loss",
    "epochs",
    "learning_rate",
    "item_alpha",
    "user_alpha",
    "max_sampled",
    "num_threads",
    "identity",
    "val_recall@10",
    "val_ndcg@10",
    "val_hr@10",
    "fit_seconds",
)

_TUNING_BASE: dict = {
    "no_components": 32,
    "loss": "warp",
    "epochs": 20,
    "learning_rate": 0.05,
    "item_alpha": 0.0,
    "user_alpha": 0.0,
    "max_sampled": 10,
    "num_threads": 4,
    "identity": False,
}


def _cfg(**over) -> dict:
    cfg = dict(_TUNING_BASE)
    cfg.update(over)
    return cfg


def _grid_phase_a() -> list[dict]:
    """第一步：固定 lr=0.05 / epochs=20 / max_sampled=10，扫 `no_components`。"""
    return [("A", _cfg(no_components=c)) for c in (16, 32, 64, 128)]


def _grid_phase_b(best_components: int) -> list[dict]:
    """第二步：把 components 固定在 A 阶段最好的值，逐个扫其余轴。"""
    c = int(best_components)
    return [
        ("B", _cfg(no_components=c, learning_rate=0.01)),
        ("B", _cfg(no_components=c, learning_rate=0.1)),
        ("C", _cfg(no_components=c, loss="bpr")),
        ("C", _cfg(no_components=c, loss="bpr", max_sampled=100)),
        ("D", _cfg(no_components=c, max_sampled=100)),
        ("E", _cfg(no_components=c, epochs=10)),
        ("E", _cfg(no_components=c, epochs=40)),
        ("F", _cfg(no_components=c, item_alpha=1e-4)),
        ("F", _cfg(no_components=c, user_alpha=1e-4)),
        # G/H：特征设计对照。侧信息只有 241 种不同用户向量（6040 人共享），
        # 加上身份列后每个用户才有可训练的自由向量，val 会跳一个量级；
        # 因此给这一族单独补一小段网格，否则最终配置只能靠单点偶然性。
        ("G", _cfg(no_components=c, identity=True)),
        ("H", _cfg(no_components=c, identity=True, max_sampled=100)),
        ("H", _cfg(no_components=32, identity=True, max_sampled=100)),
        ("H", _cfg(no_components=64, identity=True, max_sampled=100)),
        ("H", _cfg(no_components=c, identity=True, max_sampled=100, epochs=10)),
        ("H", _cfg(no_components=c, identity=True, max_sampled=100, epochs=40)),
        ("H", _cfg(no_components=c, identity=True, max_sampled=100, item_alpha=1e-5)),
    ]


def _select_config(rows: list[dict]) -> tuple[dict, dict]:
    """按 1-SE 规则选配置：在最优值 1 个标准误内，选最省的（epochs 少、维度小）。"""
    recalls = np.array([float(r["val_recall@10"]) for r in rows], dtype=np.float64)
    best_i = int(recalls.argmax())
    n = recalls.size
    se = float(recalls.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    threshold = float(recalls[best_i] - se)
    eligible = [r for r in rows if float(r["val_recall@10"]) >= threshold]
    # 省 = epochs 少 → 维度小 → 训练快
    chosen = min(
        eligible,
        key=lambda r: (int(r["epochs"]), int(r["no_components"]), float(r["fit_seconds"])),
    )
    info = {
        "rule": "1se",
        "n_configs": n,
        "best_group": int(rows[best_i]["group"]),
        "best_val_recall@10": round(float(recalls[best_i]), 6),
        "best_val_ndcg@10": round(float(rows[best_i]["val_ndcg@10"]), 6),
        "std_err": round(se, 6),
        "threshold": round(threshold, 6),
        "n_eligible": len(eligible),
        "chosen_group": int(chosen["group"]),
        "chosen_val_recall@10": round(float(chosen["val_recall@10"]), 6),
        "chosen_val_ndcg@10": round(float(chosen["val_ndcg@10"]), 6),
    }
    return chosen, info


def cmd_tune(argv: list[str]) -> int:
    """在 val 上扫网格 → CSV；再按 1-SE 选配置写 params json。test 不参与。"""
    from src.evaluate import EVAL_MODE_FULL, evaluate_matrix, load_ground_truth

    metrics_dir = OUTPUT_DIR / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    csv_path = metrics_dir / f"{TUNING_STEM}.csv"

    rows: list[dict] = []
    if _flag(argv, "--resume") and csv_path.exists():
        rows = pd.read_csv(csv_path).to_dict("records")
        print(f"[续跑] {csv_path.name} 已有 {len(rows)} 组，跳过已跑的组")
    done = {int(r["group"]) for r in rows}

    gt_val = load_ground_truth(SPLIT_VAL)
    n_grid = len(_grid_phase_a()) + len(_grid_phase_b(32))
    print(f"[调参] 只在 val 上评估（n={N_USERS}，K={K}），test 不参与")
    print(f"  网格：A 扫 components {len(_grid_phase_a())} 组 + B~H 扫其余轴 "
          f"{len(_grid_phase_b(32))} 组 = {n_grid} 组")

    def run(group: int, phase: str, cfg: dict) -> dict:
        model = LightFMAdapter(**cfg)
        t0 = time.perf_counter()
        model.fit()
        S = model.score_matrix()
        rep = evaluate_matrix(S, EVAL_MODE_FULL, split=SPLIT_VAL, model=model.name, gt=gt_val)
        row = {
            "group": group,
            "phase": phase,
            **{k: cfg[k] for k in ("no_components", "loss", "epochs", "learning_rate",
                                   "item_alpha", "user_alpha", "max_sampled",
                                   "num_threads", "identity")},
            "val_recall@10": round(float(rep.recall), 6),
            "val_ndcg@10": round(float(rep.ndcg), 6),
            "val_hr@10": round(float(rep.hr), 6),
            "fit_seconds": round(float(model.fit_seconds), 2),
        }
        rows.append(row)
        pd.DataFrame(rows, columns=list(TUNING_COLUMNS)).to_csv(csv_path, index=False)
        print(
            f"  [组 {group:>2}/{phase}] c={cfg['no_components']:<3} {cfg['loss']:<5} "
            f"ep={cfg['epochs']:<3} lr={cfg['learning_rate']:<5} "
            f"ia={cfg['item_alpha']:<6} ua={cfg['user_alpha']:<6} "
            f"ms={cfg['max_sampled']:<3} id={int(cfg['identity'])} → "
            f"val R@10={rep.recall:.4f}  NDCG@10={rep.ndcg:.4f}  "
            f"({model.fit_seconds:.0f}s, 累计 {time.perf_counter() - t0:.0f}s)",
            flush=True,
        )
        return row

    group = 0
    phase_a_rows: list[dict] = []
    for phase, cfg in _grid_phase_a():
        group += 1
        if group in done:
            phase_a_rows.append(next(r for r in rows if int(r["group"]) == group))
            continue
        phase_a_rows.append(run(group, phase, cfg))

    best_c = max(phase_a_rows, key=lambda r: float(r["val_recall@10"]))
    best_c_value = int(best_c["no_components"])
    print(f"\n  A 阶段最优 components = {best_c_value}"
          f"（val R@10={float(best_c['val_recall@10']):.4f}），后续阶段固定它\n")

    for phase, cfg in _grid_phase_b(best_c_value):
        group += 1
        if group in done:
            continue
        run(group, phase, cfg)

    chosen, info = _select_config(rows)
    print("\n[选择] 1-SE 规则")
    print(f"  最优组 {info['best_group']}：val R@10={info['best_val_recall@10']:.4f}"
          f"  NDCG@10={info['best_val_ndcg@10']:.4f}")
    print(f"  标准误 {info['std_err']:.4f} → 阈值 {info['threshold']:.4f}"
          f"（{info['n_eligible']}/{info['n_configs']} 组在 1 SE 内）")
    print(f"  选定组 {info['chosen_group']}：c={chosen['no_components']} loss={chosen['loss']} "
          f"ep={chosen['epochs']} lr={chosen['learning_rate']} "
          f"ia={chosen['item_alpha']} ua={chosen['user_alpha']} ms={chosen['max_sampled']} "
          f"identity={int(chosen['identity'])} → val R@10={info['chosen_val_recall@10']:.4f}")

    model = LightFMAdapter(
        no_components=int(chosen["no_components"]),
        loss=str(chosen["loss"]),
        epochs=int(chosen["epochs"]),
        learning_rate=float(chosen["learning_rate"]),
        item_alpha=float(chosen["item_alpha"]),
        user_alpha=float(chosen["user_alpha"]),
        max_sampled=int(chosen["max_sampled"]),
        num_threads=int(chosen["num_threads"]),
        identity=bool(chosen["identity"]),
    )
    model.fit_seconds = float(chosen["fit_seconds"])
    # 选中的配置不再重训一次（省 3 分钟），但 nnz 是可确定的常量，照实填。
    _train = read_canon_table("ratings")
    model.interactions_nnz = int((_train["split"] == SPLIT_TRAIN).sum())
    _write_params(
        model,
        {
            "selection": info,
            "tuning": {
                "csv": str(csv_path.relative_to(OUTPUT_DIR.parent)),
                "split": SPLIT_VAL,
                "note": "val 绝对值比 test 系统性偏高约 0.015（DATA.md §6.5），只用于相对排序",
            },
        },
    )
    print(f"  调参表 → {csv_path.relative_to(OUTPUT_DIR.parent)}  ({len(rows)} 组)")
    return 0


def cmd_eval(model: LightFMAdapter) -> int:
    from src.evaluate import evaluate_all, print_reports, save_report

    _fit_and_report(model)

    t0 = time.perf_counter()
    S = model.score_matrix()
    print(f"  分数矩阵 {S.shape} {S.dtype}  耗时 {time.perf_counter() - t0:.1f}s"
          f"  区间 [{S.min():.4f}, {S.max():.4f}]")

    reports = evaluate_all(S, model=model.name)
    print_reports(reports, f"LightFM 评估（test，K={K}）")

    path = save_report(reports, "lightfm_test")
    print(f"\n  指标 → {path.relative_to(OUTPUT_DIR.parent)}")

    _export_top10(model)
    _write_params(model)
    return 0


def cmd_top(model: LightFMAdapter, uid: int) -> int:
    if not (0 <= uid < N_USERS):
        raise SystemExit(f"--uid 必须在 [0, {N_USERS - 1}]，收到 {uid}")

    _fit_and_report(model)
    _export_top10(model)

    df = model.export_top10()
    mine = df[df["uid"] == uid].sort_values("rank")
    movies = read_canon_table("movies").set_index("iid")

    print(f"\n  uid={uid} 的前 {K} 推荐（已排除该用户 train ∪ val ∪ test 的所有影片）")
    rows = []
    for r in mine.itertuples(index=False):
        m = movies.loc[r.iid]
        rows.append({
            "rank": int(r.rank),
            "iid": int(r.iid),
            "title": m["title"],
            "year": int(m["year"]),
            "genres": "/".join(m["genres"][:3]),
            "score": round(float(r.score), 4),
        })
    import pandas as pd

    print(pd.DataFrame(rows).to_string(index=False))
    _write_params(model)
    return 0


# =============================================================================
# 错误分析：按目标物品训练期热度分档 + 两张图（口径与 itemcf.py 完全一致）
# =============================================================================
BUCKET_EDGES = [0, 5, 20, 100, 500, 1500, np.inf]
BUCKET_LABELS = ["<5", "5-19", "20-99", "100-499", "500-1499", ">=1500"]

# 参考线：README §6 的爆款基线（full Recall@10）与 DATA.md §6.5 的 Item-CF val 值。
POPULARITY_BASELINE = 0.0353
ITEMCF_VAL_RECALL = 0.0786


def _hit_by_bucket(
    S: np.ndarray,
    targets: np.ndarray,
    seen: np.ndarray,
    pop: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """在给定排除集合下，按目标物品的训练期热度分档统计命中率。

    名次口径与 `evaluate._rank_metrics` 一致：分数更高者在前，同分则**列下标更小**者在前
    （所以这里用 `(M > tgt).sum() + 1`，而不是 `>=`）。
    """
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
    return rates, counts, hit


def cmd_analyze(model: LightFMAdapter) -> int:
    """产出报告用的两张图 + 一张错误分析表。

    test 的命中率会在最后与 `lightfm_test.json` 的 full Recall@10 对账：两次 fit 用的是
    同一个 `SEED`（warp 的负采样与 shuffle 都由 `random_state` 驱动），分数矩阵逐位可复现，
    所以这里把 test 再算一遍**不是**"多跑一次 test"，而是同一模型的确定性重放。
    """
    # matplotlib 默认往 ~/.matplotlib 写缓存；沙箱下那里不可写（WinError 5），
    # 顺手把缓存指到 outputs 旁边，用户直接跑也不用先设环境变量。
    os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR.parent / ".mplcache"))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.evaluate import load_ground_truth

    fig_dir = OUTPUT_DIR / "figures"
    met_dir = OUTPUT_DIR / "metrics"
    fig_dir.mkdir(parents=True, exist_ok=True)

    _fit_and_report(model)
    S = model.score_matrix()

    r = read_canon_table("ratings")
    tr = r[r["split"] == SPLIT_TRAIN]
    pop = np.bincount(tr["iid"].to_numpy(), minlength=N_ITEMS).astype(np.int64)

    def targets_of(split: str) -> np.ndarray:
        t = r[r["split"] == split]
        arr = np.full(N_USERS, -1, np.int32)
        arr[t["uid"].to_numpy()] = t["iid"].to_numpy()
        return arr

    seen_train = np.zeros((N_USERS, N_ITEMS), dtype=bool)
    seen_train[tr["uid"].to_numpy(), tr["iid"].to_numpy()] = True
    seen_test = load_ground_truth(SPLIT_TEST)["seen"]

    v_rate, v_cnt, _ = _hit_by_bucket(S, targets_of(SPLIT_VAL), seen_train, pop)
    t_rate, t_cnt, t_hit = _hit_by_bucket(S, targets_of(SPLIT_TEST), seen_test, pop)

    tbl = pd.DataFrame({
        "target_popularity": BUCKET_LABELS,
        "val_n": v_cnt, "val_hit@10": np.round(v_rate, 4),
        "test_n": t_cnt, "test_hit@10": np.round(t_rate, 4),
    })
    csv = met_dir / "lightfm_popularity_buckets.csv"
    tbl.to_csv(csv, index=False)
    print("\n错误分析：按目标物品训练期热度分档的命中率")
    print(tbl.to_string(index=False))
    print(f"  表格 → {csv.relative_to(OUTPUT_DIR.parent)}")

    # 与 --eval 的 test 指标对账（同一 seed 的确定性重放）
    rep_path = met_dir / "lightfm_test.json"
    if rep_path.is_file():
        saved = json.loads(rep_path.read_text(encoding="utf-8"))
        full = (saved.get("reports") or {}).get("full", {})
        saved_r = full.get("recall")
        if saved_r is not None:
            mine = float(t_hit.mean())
            print(f"  对账：本函数重放得 full Recall@10={mine:.6f}，"
                  f"lightfm_test.json 记录 {float(saved_r):.6f}"
                  f"（差 {abs(mine - float(saved_r)):.2e}）")

    # ---- 图 1：超参敏感性（20 组 val R@10 + components 曲线） ----
    tun = met_dir / f"{TUNING_STEM}.csv"
    if tun.is_file():
        d = pd.read_csv(tun)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))

        d = d.sort_values("group")
        colors = ["#d95f02" if v else "#4c78a8" for v in d["identity"]]
        ax1.bar(d["group"], d["val_recall@10"], color=colors)
        ax1.axhline(POPULARITY_BASELINE, ls="--", c="grey", lw=1,
                    label=f"popularity baseline {POPULARITY_BASELINE}")
        ax1.axhline(ITEMCF_VAL_RECALL, ls=":", c="green", lw=1,
                    label=f"Item-CF val {ITEMCF_VAL_RECALL}")
        ax1.set_xlabel("grid group")
        ax1.set_ylabel("val Recall@10")
        ax1.set_title("all 20 configs (blue: side-info only, orange: +identity)")
        ax1.legend(fontsize=7, loc="upper left")
        ax1.grid(alpha=0.3, axis="y")

        for label, sub, style in (
            ("warp, identity=0", d[(d["identity"] == 0) & (d["loss"] == "warp")
                                   & (d["max_sampled"] == 10) & (d["epochs"] == 20)
                                   & (d["learning_rate"] == 0.05)
                                   & (d["item_alpha"] == 0) & (d["user_alpha"] == 0)], "o-"),
            ("warp, identity=1, ms=100", d[(d["identity"] == 1) & (d["loss"] == "warp")
                                           & (d["max_sampled"] == 100) & (d["epochs"] == 20)
                                           & (d["learning_rate"] == 0.05)
                                           & (d["item_alpha"] == 0)], "s--"),
        ):
            if not sub.empty:
                ax2.plot(sub["no_components"], sub["val_recall@10"], style, label=label)
        ax2.axhline(POPULARITY_BASELINE, ls="--", c="grey", lw=1)
        ax2.set_xlabel("no_components")
        ax2.set_ylabel("val Recall@10")
        ax2.set_title("components sweep: with vs without identity block")
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.3)
        fig.tight_layout()
        p1 = fig_dir / "lightfm_tuning_sensitivity.png"
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
    ax.set_title("LightFM: hits still skew to popular targets (popularity bias)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    p2 = fig_dir / "lightfm_popularity_bias.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    print(f"  图 2 → {p2.relative_to(OUTPUT_DIR.parent)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or not ({"--eval", "--top", "--tune", "--analyze"} & set(argv)):
        print(USAGE)
        return 2

    if "--tune" in argv:
        return cmd_tune(argv)

    model = _build_model(argv)
    if "--eval" in argv:
        return cmd_eval(model)
    if "--analyze" in argv:
        return cmd_analyze(model)
    uid = _opt(argv, "--uid", int, None)
    if uid is None:
        raise SystemExit(f"--top 需要 --uid N\n\n{USAGE}")
    return cmd_top(model, uid)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
