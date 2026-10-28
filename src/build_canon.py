"""
cs5481 MovieLens-1M recommender
==============================

src/build_canon.py  —  把原始 .dat 变成冻结的 canon 数据集
----------------------------------------------------------
一次性运行，产出 5 个文件到 `data/canon/`：

    ratings.<ext>            uid, iid, rating, ts, split
    users.<ext>              uid, gender, age_code, age_group,
                             occupation_code, occupation, zip
    movies.<ext>             iid, raw_movie_id, title, year, genres
    id_map.json              原始 ID ↔ 新 ID 映射 + 生成参数
    eval_candidates.<ext>    uid, iid, is_pos     (每用户 101 行，辅助表)

`<ext>` 由 `contract.CANON_FORMAT` 决定（有 pyarrow 时 .parquet，否则 .pkl）。

流水线顺序（**不可调换**）
~~~~~~~~~~~~~~~~~~~~~~~~~~
    读原始 → 物品过滤 → 全局 id 重映射 → LOO 切分 → 候选集

为什么 **过滤必须在切分之前**：
    - 先过滤后 LOO → 6040 个用户全部保留（过滤后每人仍有 ≥18 条评分）
    - 先 LOO 后过滤 → 6 个用户的 test target 被删掉，评估用户数变成 6034
详见 DATA.md §5.2。

为什么 **候选集只能从 3416 个保留物品里采**：
    被过滤的物品若混进候选池，会变成"免费负样本"，系统性抬高 Recall@10。
本文件的 `_assert_no_leakage()` 会拦住这种情况。

用法
~~~~
    python -m src.build_canon            # 生成（会覆盖旧文件）
    python -m src.build_canon --check    # 只校验已有 canon 是否与 EXPECTED 一致
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.contract import (
    AGE_GROUP, CANON_DIR, CANON_EVAL_CANDIDATES, CANON_ID_MAP,
    CANON_RATINGS, CANON_USERS, CANON_MOVIES,
    CANDIDATES_PER_USER, CATALOG_N_ITEMS, EVAL_MODE_FULL, EXPECTED,
    EXPECTED_SPLIT_ROWS, FILTER_BEFORE_SPLIT, GENRES, MIN_ITEM_FREQ,
    MOVIES_COLUMNS, MOVIES_ENCODING, N_ITEMS, N_NEG_CANDIDATES, N_USERS,
    OCCUPATION, RATINGS_COLUMNS, RAW_MOVIES, RAW_RATINGS, RAW_USERS,
    RATING_MAX, RATING_MIN, SEED, SEEDS, SEP, SPLIT_TEST, SPLIT_TRAIN,
    SPLIT_VAL, SPLITS, TITLE_YEAR_RE, USERS_COLUMNS, ensure_dirs,
    read_canon_table, write_canon_table,
)

# =============================================================================
# 小工具
# =============================================================================
_T0 = time.perf_counter()


def log(msg: str) -> None:
    print(f"[{time.perf_counter() - _T0:6.2f}s] {msg}", flush=True)


def _banner(title: str) -> None:
    print(f"\n{'─' * 70}\n{title}\n{'─' * 70}", flush=True)


# =============================================================================
# 步骤 1：读原始文件
# =============================================================================
def read_raw() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """读 3 个 .dat。注意三件事：`::` 分隔、无表头、movies 必须 latin-1。"""
    _banner("步骤 1 · 读取原始文件")

    for path in (RAW_RATINGS, RAW_USERS, RAW_MOVIES):
        if not path.is_file():
            raise FileNotFoundError(
                f"找不到 {path}。请先解压官方 ml-1m.zip，"
                f"或把它放到 data/raw/ml-1m/（见 DATA.md §1）。"
            )

    ratings = pd.read_csv(
        RAW_RATINGS, sep=SEP, header=None, engine="python",
        names=list(RATINGS_COLUMNS),
        dtype={"UserID": "int32", "MovieID": "int32", "Rating": "int8", "Timestamp": "int64"},
    )
    users = pd.read_csv(
        RAW_USERS, sep=SEP, header=None, engine="python",
        names=list(USERS_COLUMNS),
        dtype={"UserID": "int32", "Gender": "str", "Age": "int16",
               "Occupation": "int16", "Zip": "str"},   # Zip 必须是 str，否则丢前导零
    )
    movies = pd.read_csv(
        RAW_MOVIES, sep=SEP, header=None, engine="python",
        names=list(MOVIES_COLUMNS),
        dtype={"MovieID": "int32", "Title": "str", "Genres": "str"},
        encoding=MOVIES_ENCODING,   # utf-8 会在 byte 3114 崩
    )

    assert len(ratings) == EXPECTED["raw_ratings"], f"ratings 行数 {len(ratings)}"
    assert len(users) == EXPECTED["raw_users"], f"users 行数 {len(users)}"
    assert len(movies) == EXPECTED["catalog_items"], f"movies 行数 {len(movies)}"

    # 评分量表必须是 1..5；出现 0 说明读错了（ML-1M 没有 0 星）
    bad = ratings[~ratings["Rating"].between(RATING_MIN, RATING_MAX)]
    assert bad.empty, f"评分越界 {len(bad)} 行，量表应为 [{RATING_MIN}, {RATING_MAX}]"

    dup = ratings.duplicated(subset=["UserID", "MovieID"]).sum()
    assert dup == EXPECTED["duplicate_user_item_pairs"], f"重复 (user,item) {dup} 条"

    log(f"ratings {len(ratings):>9,} 行   users {len(users):>6,}   movies {len(movies):>5,}")
    log(f"评分量表 [{RATING_MIN}, {RATING_MAX}] 无越界；重复 (user,item) 对 = 0")
    return ratings, users, movies


# =============================================================================
# 步骤 2：物品过滤（必须在切分之前）
# =============================================================================
def filter_items(ratings: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """两级过滤：① 剔除零评分物品 ② 剔除评分 < MIN_ITEM_FREQ 的物品。"""
    _banner("步骤 2 · 物品过滤（在切分之前）")

    freq = ratings["MovieID"].value_counts()
    rated_items = set(freq.index)

    zero_rated = CATALOG_N_ITEMS - len(rated_items)
    assert zero_rated == EXPECTED["zero_rated_items"], f"零评分物品 {zero_rated}"

    kept_items = {m for m, c in freq.items() if c >= MIN_ITEM_FREQ}
    dropped = len(rated_items) - len(kept_items)
    assert dropped == EXPECTED["dropped_low_freq_items"], f"低频物品 {dropped}"

    n_before = len(ratings)
    filtered = ratings[ratings["MovieID"].isin(kept_items)].reset_index(drop=True)
    n_dropped_rows = n_before - len(filtered)
    assert n_dropped_rows == EXPECTED["dropped_low_freq_ratings"], f"丢弃评分 {n_dropped_rows}"

    assert len(kept_items) == N_ITEMS, f"保留物品 {len(kept_items)} != N_ITEMS={N_ITEMS}"

    log(f"① 零评分物品   {zero_rated:>4} 部  ({CATALOG_N_ITEMS:,} → {len(rated_items):,})")
    log(f"② 评分 <{MIN_ITEM_FREQ} 物品 {dropped:>4} 部  ({len(rated_items):,} → {N_ITEMS:,})")
    log(f"   随之丢弃评分 {n_dropped_rows:,} 行  ({n_before:,} → {len(filtered):,})")

    stats = {
        "catalog_items": CATALOG_N_ITEMS,
        "rated_items": len(rated_items),
        "zero_rated_items": zero_rated,
        "dropped_low_freq_items": dropped,
        "dropped_low_freq_ratings": n_dropped_rows,
    }
    return filtered, stats


# =============================================================================
# 步骤 3：全局 id 重映射
# =============================================================================
def build_id_maps(
    ratings: pd.DataFrame, users: pd.DataFrame, movies: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, dict[int, int]]]:
    """原始 ID → 0-based 连续整数。MovieID 有 69 个空洞，必须重映射。"""
    _banner("步骤 3 · 全局 id 重映射")

    kept_movie_ids = sorted(ratings["MovieID"].unique().tolist())
    user_ids = sorted(ratings["UserID"].unique().tolist())

    user_map = {raw: new for new, raw in enumerate(user_ids)}
    item_map = {raw: new for new, raw in enumerate(kept_movie_ids)}

    ratings = ratings.assign(
        uid=ratings["UserID"].map(user_map).astype("int32"),
        iid=ratings["MovieID"].map(item_map).astype("int32"),
    )

    # users：只保留出现在 ratings 里的用户（本数据集三者完全一致）
    users = users[users["UserID"].isin(user_map)].copy()
    users = users.assign(uid=users["UserID"].map(user_map).astype("int32"))
    users = users.assign(
        age_group=users["Age"].map(AGE_GROUP),
        occupation=users["Occupation"].map(OCCUPATION),
    )
    assert users["age_group"].notna().all(), "存在未映射的 Age 码"
    assert users["occupation"].notna().all(), "存在未映射的 Occupation 码"

    # movies：只保留保留下来的物品
    movies = movies[movies["MovieID"].isin(item_map)].copy()
    movies = movies.assign(
        iid=movies["MovieID"].map(item_map).astype("int32"),
        year=movies["Title"].str.extract(TITLE_YEAR_RE)[0].astype("int16"),
        genres=movies["Genres"].str.split("|"),
    )
    assert movies["year"].notna().all(), "有片名解析不出年份"
    unknown = {g for gs in movies["genres"] for g in gs} - set(GENRES)
    assert not unknown, f"出现未知类型 {unknown}"

    # 连续性
    assert sorted(item_map.values()) == list(range(N_ITEMS)), "iid 不是 0..N-1 连续"
    assert sorted(user_map.values()) == list(range(N_USERS)), "uid 不是 0..N-1 连续"

    log(f"uid : {len(user_map):,} 个，0..{N_USERS - 1}")
    log(f"iid : {len(item_map):,} 个，0..{N_ITEMS - 1}（原始 MovieID 最大 3952，有 69 个空洞）")
    log(f"年份解析成功 {movies['year'].notna().sum():,}/{len(movies):,}")
    return ratings, users, movies, {"user": user_map, "item": item_map}


# =============================================================================
# 步骤 4：Leave-One-Out 切分
# =============================================================================
def loo_split(ratings: pd.DataFrame) -> pd.DataFrame:
    """每人按 (ts, raw_iid) 升序排；最后一条 → test，倒数第二条 → val，其余 → train。

    用 (ts, raw_iid) 排序是为了打破时间戳并列，保证**完全确定性**：
    同一份数据无论跑多少次、什么 pandas 版本，切分结果都一致。
    """
    _banner("步骤 4 · Leave-One-Out 切分")

    df = (
        ratings.sort_values(["uid", "Timestamp", "MovieID"], kind="mergesort")
        .reset_index(drop=True)
    )
    rev_rank = df.groupby("uid", sort=False).cumcount(ascending=False)
    split = np.where(rev_rank == 0, SPLIT_TEST, np.where(rev_rank == 1, SPLIT_VAL, SPLIT_TRAIN))

    df = df.assign(
        rating=df["Rating"].astype("float32"),
        ts=df["Timestamp"].astype("int64"),
        split=pd.Categorical(split, categories=list(SPLITS), ordered=False),
    )

    counts = df["split"].value_counts().to_dict()
    for name in SPLITS:
        assert counts.get(name, 0) == EXPECTED_SPLIT_ROWS[name], (
            f"{name} 行数 {counts.get(name, 0)} != {EXPECTED_SPLIT_ROWS[name]}"
        )

    # 每个用户恰好 1 条 val + 1 条 test
    n_val = df[df["split"] == SPLIT_VAL].groupby("uid").size()
    n_test = df[df["split"] == SPLIT_TEST].groupby("uid").size()
    assert len(n_val) == N_USERS and (n_val == 1).all(), "有用户不是恰好 1 条 val"
    assert len(n_test) == N_USERS and (n_test == 1).all(), "有用户不是恰好 1 条 test"

    # train 至少 1 条（否则该用户无法训练）
    n_train = df[df["split"] == SPLIT_TRAIN].groupby("uid").size()
    assert len(n_train) == N_USERS and n_train.min() >= 1, "有用户 train 为空"

    log(f"{SPLIT_TRAIN:<6} {counts[SPLIT_TRAIN]:>9,}")
    log(f"{SPLIT_VAL:<6} {counts[SPLIT_VAL]:>9,}   (= N_USERS，每人 1 条)")
    log(f"{SPLIT_TEST:<6} {counts[SPLIT_TEST]:>9,}   (= N_USERS，每人 1 条)")
    log(f"train 长度  min={n_train.min()}  median={n_train.median():.0f}  max={n_train.max()}")
    return df


# =============================================================================
# 步骤 5：101 候选集（辅助表）
# =============================================================================
def build_eval_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """每用户 = 1 个 test 正样本 + N_NEG_CANDIDATES 个未交互负样本。

    - 负样本从**全部 3416 个保留物品**中采，排除该用户 train ∪ val 里出现过的，
      再排除 test 本身。因此候选池里**不可能**出现被过滤的物品。
    - 每用户用 `default_rng([SEED, uid])` 独立播种：结果不依赖用户遍历顺序，
      也不依赖样本总数，任何人重跑都得到同一份候选集。
    - 101 行会做一次打乱，避免"第 0 行永远是正样本"这种位置泄漏。
    """
    _banner("步骤 5 · 生成 101 候选集（辅助表）")

    train = df[df["split"] == SPLIT_TRAIN]
    val = df[df["split"] == SPLIT_VAL]
    test = df[df["split"] == SPLIT_TEST].set_index("uid")["iid"]

    seen = np.zeros((N_USERS, N_ITEMS), dtype=bool)
    seen[train["uid"].to_numpy(), train["iid"].to_numpy()] = True
    seen[val["uid"].to_numpy(), val["iid"].to_numpy()] = True

    test_iid = test.reindex(np.arange(N_USERS)).to_numpy()
    assert not np.isnan(test_iid).any(), "有用户没有 test 目标"

    n_rows = N_USERS * CANDIDATES_PER_USER
    out_iid = np.empty(n_rows, dtype=np.int32)
    out_lab = np.empty(n_rows, dtype=np.int8)
    rng_seed = SEEDS["negative_sampling"]

    min_eligible = N_ITEMS
    for uid in range(N_USERS):
        eligible = np.flatnonzero(~seen[uid])
        eligible = eligible[eligible != test_iid[uid]]
        min_eligible = min(min_eligible, eligible.size)
        if eligible.size < N_NEG_CANDIDATES:
            raise RuntimeError(
                f"用户 {uid} 只有 {eligible.size} 个可选负样本 "
                f"(< {N_NEG_CANDIDATES})，候选集无法构造"
            )

        # 每用户独立播种 → 与遍历顺序无关
        rng = np.random.default_rng([rng_seed, uid])
        neg = rng.choice(eligible, size=N_NEG_CANDIDATES, replace=False)

        cand = np.empty(CANDIDATES_PER_USER, dtype=np.int32)
        lab = np.zeros(CANDIDATES_PER_USER, dtype=np.int8)
        cand[0] = test_iid[uid]
        lab[0] = 1
        cand[1:] = neg

        perm = rng.permutation(CANDIDATES_PER_USER)   # 防位置泄漏
        lo = uid * CANDIDATES_PER_USER
        out_iid[lo:lo + CANDIDATES_PER_USER] = cand[perm]
        out_lab[lo:lo + CANDIDATES_PER_USER] = lab[perm]

    cands = pd.DataFrame({
        "uid": np.repeat(np.arange(N_USERS, dtype=np.int32), CANDIDATES_PER_USER),
        "iid": out_iid,
        "is_pos": out_lab,
    })

    assert len(cands) == EXPECTED["candidates_rows"], f"候选行数 {len(cands)}"
    assert int(cands["is_pos"].sum()) == N_USERS, "正样本数应恰好等于用户数"
    assert (cands.groupby("uid").size() == CANDIDATES_PER_USER).all(), "有用户候选数不对"
    assert (cands.groupby("uid")["is_pos"].sum() == 1).all(), "有用户正样本数不是 1"
    assert not cands.duplicated(subset=["uid", "iid"]).any(), "候选内有重复 (uid,iid)"

    log(f"行数 {len(cands):,}  =  {N_USERS:,} 用户 × {CANDIDATES_PER_USER}")
    log(f"正样本 {int(cands['is_pos'].sum()):,}  负样本 {int((1 - cands['is_pos']).sum()):,}")
    log(f"最少可选负样本的用户有 {min_eligible} 个候选（远大于 {N_NEG_CANDIDATES}）")
    return cands


# =============================================================================
# 步骤 6：写盘
# =============================================================================
def write_canon(
    df: pd.DataFrame,
    users: pd.DataFrame,
    movies: pd.DataFrame,
    cands: pd.DataFrame,
    id_maps: dict[str, dict[int, int]],
    filter_stats: dict[str, int],
) -> None:
    _banner("步骤 6 · 写入 data/canon/")

    ratings_out = df[[c for c in ("uid", "iid", "rating", "ts", "split")]].copy()
    users_out = users[[
        "uid", "Gender", "Age", "Occupation", "Zip", "age_group", "occupation",
    ]].rename(columns={
        "Gender": "gender", "Age": "age_code", "Occupation": "occupation_code", "Zip": "zip",
    })
    users_out = users_out.assign(
        gender=users_out["gender"].astype("category"),
        age_group=users_out["age_group"].astype("category"),
        occupation=users_out["occupation"].astype("category"),
    ).sort_values("uid").reset_index(drop=True)

    movies_out = (
        movies[["iid", "MovieID", "Title", "year", "genres"]]
        .rename(columns={"MovieID": "raw_movie_id", "Title": "title"})
        .sort_values("iid")
        .reset_index(drop=True)
    )

    p1 = write_canon_table(ratings_out, "ratings")
    p2 = write_canon_table(users_out, "users")
    p3 = write_canon_table(movies_out, "movies")
    p4 = write_canon_table(cands, "eval_candidates")

    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generated_by": "src/build_canon.py",
        "n_users": N_USERS,
        "n_items": N_ITEMS,
        "seed": SEED,
        "params": {
            "min_item_freq": MIN_ITEM_FREQ,
            "filter_before_split": FILTER_BEFORE_SPLIT,
            "split": "leave-one-out by (timestamp, raw_movie_id)",
            "relevance_rule": "all_observed",
            "n_neg_candidates": N_NEG_CANDIDATES,
            "sep": SEP,
            "movies_encoding": MOVIES_ENCODING,
        },
        "filter_stats": filter_stats,
        "user": {str(k): v for k, v in id_maps["user"].items()},
        "item": {str(k): v for k, v in id_maps["item"].items()},
    }
    CANON_DIR.mkdir(parents=True, exist_ok=True)
    CANON_ID_MAP.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    for path in (p1, p2, p3, p4, CANON_ID_MAP):
        log(f"{path.stat().st_size / 1e6:8.2f} MB  {path.relative_to(CANON_DIR.parent.parent)}")


# =============================================================================
# 步骤 7：校验
# =============================================================================
def validate() -> None:
    """从磁盘重新读回 canon，逐条对照 EXPECTED。"""
    _banner("步骤 7 · 校验 canon")

    r = read_canon_table("ratings")
    u = read_canon_table("users")
    m = read_canon_table("movies")
    c = read_canon_table("eval_candidates")

    checks: list[tuple[str, object, object]] = [
        ("ratings 行数", len(r), EXPECTED["canon_ratings"]),
        ("users 行数", len(u), EXPECTED["canon_users"]),
        ("movies 行数", len(m), EXPECTED["canon_items"]),
        ("candidates 行数", len(c), EXPECTED["candidates_rows"]),
        ("uid 最大值", int(r["uid"].max()), N_USERS - 1),
        ("iid 最大值", int(r["iid"].max()), N_ITEMS - 1),
        ("唯一用户数", int(r["uid"].nunique()), N_USERS),
        ("物品数", int(r["iid"].nunique()), N_ITEMS),
        ("train 行数", int((r["split"] == SPLIT_TRAIN).sum()), EXPECTED_SPLIT_ROWS[SPLIT_TRAIN]),
        ("val 行数", int((r["split"] == SPLIT_VAL).sum()), EXPECTED_SPLIT_ROWS[SPLIT_VAL]),
        ("test 行数", int((r["split"] == SPLIT_TEST).sum()), EXPECTED_SPLIT_ROWS[SPLIT_TEST]),
        ("平均分(4位)", round(float(r["rating"].mean()), 4), EXPECTED["canon_mean_rating"]),
    ]
    for name, got, want in checks:
        assert got == want, f"{name}: 得到 {got}，期望 {want}"
        log(f"✓ {name:<14} {got}")

    # 无 NaN
    for name, table in (("ratings", r), ("users", u), ("movies", m)):
        assert not table.isna().any().any(), f"{name} 含 NaN"
    log("✓ 三张表均无 NaN")

    # 每个用户恰好 1 val + 1 test
    g = r.groupby(["uid", "split"], observed=False).size().unstack(fill_value=0)
    assert (g[SPLIT_VAL] == 1).all() and (g[SPLIT_TEST] == 1).all()
    log("✓ 每用户恰好 1 条 val + 1 条 test")

    # test 目标不在该用户 train 里
    test_pairs = r.loc[r["split"] == SPLIT_TEST, ["uid", "iid"]]
    train_pairs = set(map(tuple, r.loc[r["split"] == SPLIT_TRAIN, ["uid", "iid"]].to_numpy()))
    overlap = sum((a, b) in train_pairs for a, b in test_pairs.to_numpy())
    assert overlap == 0, f"{overlap} 个 test 目标泄漏进 train"
    log("✓ test 目标与 train 无重叠（无泄漏）")

    # 候选集不含被过滤物品（全部 iid < N_ITEMS）
    assert int(c["iid"].max()) < N_ITEMS
    log(f"✓ 候选 iid 全部 < {N_ITEMS}（无被过滤物品混入）")

    # 候选集负样本不得是该用户 train/val 里出现过的
    seen = set(map(tuple, r.loc[r["split"].isin([SPLIT_TRAIN, SPLIT_VAL]),
                               ["uid", "iid"]].to_numpy()))
    neg = c.loc[c["is_pos"] == 0, ["uid", "iid"]].to_numpy()
    leaked = sum((a, b) in seen for a, b in neg)
    assert leaked == 0, f"{leaked} 个负样本其实是该用户交互过的物品"
    log("✓ 候选集负样本与用户 train/val 无重叠")


# =============================================================================
# 展示样例（方便人工确认）
# =============================================================================
def show_examples(n_users: int = 2) -> None:
    r = read_canon_table("ratings")
    m = read_canon_table("movies")
    c = read_canon_table("eval_candidates")
    title = dict(zip(m["iid"], m["title"]))
    genre = dict(zip(m["iid"], m["genres"]))

    _banner("样例（人工确认用）")
    for uid in range(n_users):
        sub = r[r["uid"] == uid]
        print(f"\n用户 uid={uid}   总评分 {len(sub)} 条")
        for split in SPLITS:
            part = sub[sub["split"] == split].sort_values("ts")
            head, tail = part.iloc[0], part.iloc[-1]
            print(f"  {split:<5} {len(part):>4} 条 | 最早: {title[head['iid']]:<42.42} "
                  f"({head['rating']:.0f}★)  | 最晚: {title[tail['iid']]:<42.42} ({tail['rating']:.0f}★)")
        cc = c[c["uid"] == uid]
        pos = cc[cc["is_pos"] == 1].iloc[0]
        print(f"  候选集 101 行，正样本 iid={pos['iid']} → {title[pos['iid']]} "
              f"[{','.join(genre[pos['iid']])}]")


# =============================================================================
# main
# =============================================================================
def build() -> None:
    ensure_dirs()
    log(f"输出目录 {CANON_DIR}")

    ratings, users, movies = read_raw()
    filtered, filter_stats = filter_items(ratings)
    filtered, users, movies, id_maps = build_id_maps(filtered, users, movies)
    df = loo_split(filtered)
    cands = build_eval_candidates(df)
    write_canon(df, users, movies, cands, id_maps, filter_stats)
    validate()
    show_examples()

    print(f"\n{'=' * 70}")
    print(f"完成，用时 {time.perf_counter() - _T0:.1f}s")
    print(f"  N_USERS={N_USERS}  N_ITEMS={N_ITEMS}")
    print(f"  train={EXPECTED_SPLIT_ROWS[SPLIT_TRAIN]:,}  "
          f"val={EXPECTED_SPLIT_ROWS[SPLIT_VAL]:,}  test={EXPECTED_SPLIT_ROWS[SPLIT_TEST]:,}")
    print(f"  随机基线 Recall@10：{EVAL_MODE_FULL} ≈ {10 / N_ITEMS:.6f}")
    print(f"{'=' * 70}")


def main(argv: list[str]) -> int:
    if "--check" in argv:
        validate()
        return 0
    build()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
