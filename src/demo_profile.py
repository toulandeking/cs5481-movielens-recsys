"""
cs5481 MovieLens-1M recommender
==============================

src/demo_profile.py  —  按"人群画像"做冷启动推荐（演示用，不参与评估）
--------------------------------------------------------------------
用途：让演示者输入一组**新用户**的属性（性别 / 年龄段 / 喜欢的类型），
输出一批电影。这是 5 分钟展示里"现场演示"的那一环。

⚠️ 重要：**Item-CF 本身无法消费性别 / 年龄 / 类型**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Item-CF 的输入只有一张 user × item 的交互矩阵。它没有任何侧信息通道，
所以对"一个零交互的新用户"完全无能为力 —— 这正是**冷启动**。

本模块用一条**两步桥接**让邻域方法也能回答：

    ① 人群画像：从 canon 里找出性别 + 年龄段相同、且喜欢所选类型的一批真实用户
                 → 用他们的评分聚合出一份"这个画像爱看什么"的偏好向量
    ② 伪历史 + Item-CF：取偏好最高的 k 部片当作"该假想用户看过的东西"
                 → 用 Item-CF 的相似度矩阵向外扩散，得到推荐

所以它输出的推荐**大部分是 Item-CF 算出来的**，但"历史"是借来的。

诚实标注
~~~~~~~~
- 这是**冷启动的近似方案**，不是 Item-CF 的原生能力。报告里必须这样写。
- 真正的解法是 **LightFM**：它的用户表示是"属性 embedding 之和"，
  一个新用户只要有性别/年龄就能直接算分，不需要借历史。见 FRAMEWORK.md §5.1。
- 本模块因此还兼任**冷启动实验的对照组**：LightFM 应当在同一画像上显著优于它。

数据纪律
~~~~~~~~
所有统计量只从 `split == "train"` 计算（与 DATA.md §6.4 一致），不使用 val/test。

用法
~~~~
    python -m src.demo_profile --gender F --age 25-34 --genres Comedy,Romance
    python -m src.demo_profile --genres "Sci-Fi" --top 10
    python -m src.demo_profile --gender M --age 18-24 --explain
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.contract import (
    AGE_GROUP, GENRES, K, N_ITEMS, N_USERS, SPLIT_TRAIN, read_canon_table,
)
from src.adapters.base import build_binary_matrix
from src.adapters.itemcf import cosine_similarity, truncate_topk, weight_users

__all__ = ["Profile", "build_cohort", "cohort_affinity", "recommend_for_profile"]


# =============================================================================
# 参数
# =============================================================================
SHRINK_LAMBDA = 10.0     # 人群偏好收缩：score = Σ(r-3)⁺ / (n_raters + λ)
MIN_SUPPORT = 15         # 一部片至少要有多少同群用户评过，才纳入画像
MIN_COHORT = 100         # 人群少于这个数就放宽条件
PSEUDO_HISTORY_K = 30    # 用偏好最高的多少部片当作"伪历史"


@dataclass
class Profile:
    gender: str | None = None          # "M" / "F" / None
    age_group: str | None = None       # "25-34" / None
    genres: tuple[str, ...] = ()       # ("Comedy", "Romance")

    def describe(self) -> str:
        g = {"M": "男", "F": "女"}.get(self.gender or "", "不限")
        a = self.age_group or "不限年龄段"
        s = "、".join(self.genres) if self.genres else "不限类型"
        return f"性别={g}  年龄={a}  偏好类型={s}"


# =============================================================================
# ① 人群画像
# =============================================================================
def _genre_mask(genres: tuple[str, ...]) -> np.ndarray:
    """(N_ITEMS,) bool：该片是否属于所选类型中的任意一个。"""
    if not genres:
        return np.ones(N_ITEMS, dtype=bool)
    movies = read_canon_table("movies").sort_values("iid")
    want = set(genres)
    return np.array([bool(want & set(gs)) for gs in movies["genres"]], dtype=bool)


def build_cohort(profile: Profile, min_size: int = MIN_COHORT) -> tuple[np.ndarray, list[str]]:
    """筛出符合画像的真实用户。命中太少时按序放宽条件，并记录放宽了什么。"""
    users = read_canon_table("users")
    ratings = read_canon_table("ratings")
    train = ratings[ratings["split"] == SPLIT_TRAIN]

    mask = np.ones(len(users), dtype=bool)
    if profile.gender:
        mask &= (users["gender"].astype(str) == profile.gender).to_numpy()
    if profile.age_group:
        mask &= (users["age_group"].astype(str) == profile.age_group).to_numpy()

    notes: list[str] = []
    if profile.gender:
        notes.append(f"性别={profile.gender}")
    if profile.age_group:
        notes.append(f"年龄段={profile.age_group}")

    # 类型条件：至少在所选类型上有 1 条 ≥4 星评分
    if profile.genres:
        gmask = _genre_mask(profile.genres)
        liked = train[train["rating"] >= 4]
        liked = liked[gmask[liked["iid"].to_numpy()]]
        genre_uids = set(liked["uid"].unique().tolist())
        mask &= users["uid"].map(lambda u: u in genre_uids).to_numpy()
        notes.append("喜欢 " + "/".join(profile.genres))

    cohort = users.loc[mask, "uid"].to_numpy()
    if len(cohort) >= min_size:
        return cohort, notes

    # 放宽：先丢类型，再丢年龄，最后只剩性别
    if profile.genres:
        relaxed = Profile(profile.gender, profile.age_group, ())
        c, _ = build_cohort(relaxed, min_size)
        notes.append(f"⚠️ 类型条件过严（仅 {len(cohort)} 人），已放宽为不限类型")
        return c, notes
    if profile.age_group:
        relaxed = Profile(profile.gender, None, ())
        c, _ = build_cohort(relaxed, min_size)
        notes.append(f"⚠️ 年龄段过严（仅 {len(cohort)} 人），已放宽为不限年龄")
        return c, notes
    return cohort, notes


def cohort_affinity(cohort: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """把人群评分聚合成逐物品偏好。

    score_i = Σ_{u∈cohort} max(r_ui - 3, 0) / (n_raters_i + λ)

    减去 3 表示"只有 4、5 星才算喜欢"；分母的 λ 让只有零星几人评过的片
    不会因为运气好而排到最前（和 Item-CF 里 shrinkage 的思路一致）。
    """
    ratings = read_canon_table("ratings")
    sub = ratings[(ratings["split"] == SPLIT_TRAIN) & (ratings["uid"].isin(cohort))]
    pos = np.maximum(sub["rating"].to_numpy(dtype=np.float32) - 3.0, 0.0)
    agg = pd.DataFrame({"iid": sub["iid"].to_numpy(), "pos": pos}) \
        .groupby("iid")["pos"].agg(["sum", "count"])
    score = np.zeros(N_ITEMS, dtype=np.float32)
    n_raters = np.zeros(N_ITEMS, dtype=np.int32)
    ids = agg.index.to_numpy()
    score[ids] = agg["sum"].to_numpy() / (agg["count"].to_numpy() + SHRINK_LAMBDA)
    n_raters[ids] = agg["count"].to_numpy().astype(np.int32)
    score[n_raters < MIN_SUPPORT] = 0.0      # 支撑不足，不纳入画像
    return score, n_raters


# =============================================================================
# ② 伪历史 + Item-CF 扩散
# =============================================================================
def recommend_for_profile(
    profile: Profile,
    top_n: int = K,
    k_neighbors: int = 20,
    alpha: float = 0.5,
    blend: float = 0.0,
) -> dict:
    """完整流程：画像 → 人群 → 伪历史 → Item-CF 扩散 → 推荐。"""
    cohort, notes = build_cohort(profile)
    if len(cohort) == 0:
        raise RuntimeError("该画像下没有匹配到任何用户，请放宽条件")

    affinity, n_raters = cohort_affinity(cohort)
    gmask = _genre_mask(profile.genres)

    # 伪历史：偏好最高、且（若指定类型）类型匹配的 k 部
    cand = np.where(gmask & (affinity > 0), affinity, -np.inf)
    k_hist = min(PSEUDO_HISTORY_K, int((cand > -np.inf).sum()))
    pseudo = np.argpartition(-cand, k_hist - 1)[:k_hist]
    pseudo = pseudo[np.argsort(-cand[pseudo])]

    # Item-CF：用同一套相似度（α 衰减 + top-k 截断）
    R = build_binary_matrix(SPLIT_TRAIN)
    sim = truncate_topk(cosine_similarity(weight_users(R, alpha)), k_neighbors)
    cf_score = sim[pseudo].sum(axis=0)
    cf_score[pseudo] = -np.inf                     # 别推荐"伪历史"本身

    # 可选：与人群偏好混合（0 = 纯 CF，1 = 纯人群偏好）
    if blend <= 0:
        final = cf_score.copy()                    # 保留原始量纲，与 ItemCF 可比
    else:
        pop_norm = affinity / max(affinity.max(), 1e-9)
        cf_norm = cf_score / max(cf_score[cf_score > -np.inf].max(), 1e-9)
        final = (1 - blend) * cf_norm + blend * pop_norm
    final[pseudo] = -np.inf
    if profile.genres:
        # 只展示类型匹配的结果，符合"我喜欢喜剧就给我喜剧"的直觉
        final = np.where(gmask, final, -np.inf)

    order = np.argsort(-final, kind="stable")[:top_n]
    return {
        "cohort_size": len(cohort),
        "notes": notes,
        "pseudo": pseudo,
        "affinity": affinity,
        "n_raters": n_raters,
        "scores": final,
        "top": order,
    }


# =============================================================================
# CLI
# =============================================================================
def _parse_age(v: str) -> str:
    """接受 '25-34' 或代码 '25'，统一成 age_group 字符串。"""
    v = v.strip()
    if v in AGE_GROUP.values():
        return v
    try:
        code = int(v)
    except ValueError:
        raise SystemExit(f"无法识别的年龄段 {v!r}，可选：{list(AGE_GROUP.values())}")
    if code not in AGE_GROUP:
        raise SystemExit(f"无法识别的年龄代码 {code}，可选：{sorted(AGE_GROUP)}")
    return AGE_GROUP[code]


def main(argv: list[str]) -> int:
    def opt(flag, default=None):
        return argv[argv.index(flag) + 1] if flag in argv else default

    gender = (opt("--gender") or "").strip().upper() or None
    if gender not in (None, "M", "F"):
        raise SystemExit("--gender 只能是 M 或 F")
    age_raw = opt("--age")
    age = _parse_age(age_raw) if age_raw else None
    genres_raw = opt("--genres") or ""
    genres = tuple(g.strip() for g in genres_raw.split(",") if g.strip())
    bad = [g for g in genres if g not in GENRES]
    if bad:
        raise SystemExit(f"未知类型 {bad}，可选：{list(GENRES)}")

    top_n = int(opt("--top", K))
    blend = float(opt("--blend", 0.0))
    profile = Profile(gender, age, genres)

    print("=" * 72)
    print("按人群画像推荐（演示）")
    print("=" * 72)
    print(f"  输入：{profile.describe()}")

    res = recommend_for_profile(profile, top_n=top_n, blend=blend)
    print(f"  匹配人群：{res['cohort_size']:,} 人  ({' / '.join(res['notes'])})")

    movies = read_canon_table("movies").sort_values("iid")
    title = movies["title"].to_numpy()
    year = movies["year"].to_numpy()
    genre_list = movies["genres"].to_numpy()

    if "--explain" in argv:
        print(f"\n  伪历史（借来的 {len(res['pseudo'])} 部，作为该画像的'已看'）：")
        for i, iid in enumerate(res["pseudo"][:10], 1):
            print(f"    {i:>2}. {title[iid]:<44.44} ({year[iid]})  "
                  f"人群偏好={res['affinity'][iid]:.3f}  评过={res['n_raters'][iid]}人")
        if len(res["pseudo"]) > 10:
            print(f"    ... 共 {len(res['pseudo'])} 部")

    print(f"\n  推荐前 {len(res['top'])} 部：")
    print(f"  {'#':>2}  {'片名':<44} {'年份':<5} {'类型':<34} 分数")
    print("  " + "-" * 100)
    for rank, iid in enumerate(res["top"], 1):
        iid = int(iid)
        gs = ", ".join(genre_list[iid])
        print(f"  {rank:>2}. {title[iid]:<44.44} {year[iid]:<5} {gs[:34]:<34} "
              f"{res['scores'][iid]:.4f}")

    print("\n  说明：分数来自 Item-CF 相似度扩散（伪历史 × 物品相似度之和），"
          "已排除伪历史自身。")
    print("  这是冷启动的**近似**方案；原生支持该输入的是 LightFM（用户特征直连）。")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
