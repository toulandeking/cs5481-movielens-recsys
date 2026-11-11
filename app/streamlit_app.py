"""
cs5481 MovieLens-1M recommender
==============================

app/streamlit_app.py  —  Interactive demo (English UI)
------------------------------------------------------
Satisfies the presentation requirement:

  * "Present your movie recommendation system in the form of a web or mobile
     application, notebook, or equivalent interactive interface."
  * "The interface should accept **a user or sample profile**."
  * "Display movie titles and genres and **compare the recommendations generated
     by the different algorithms**."

Two tabs:
  Tab 1 "Pick a user"    -> a real user id, three models side by side
  Tab 2 "Describe a user" -> demographics + preferred genres (a sample profile)

Design
~~~~~~
1. **Reads result files only, never loads a model.** So the demo still works on a
   machine that cannot install lightfm or torch.
2. **Degrades gracefully.** A missing model shows a placeholder; nothing breaks.
3. Looks for artifacts in both `results/<model>/` and `outputs/`, because team
   members exported to slightly different places.

Run
~~~
    python -m pip install streamlit
    streamlit run app/streamlit_app.py

Start it from the repository root (or pass an absolute path), because the app
imports `src.*`.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

# Streamlit only puts the script's own folder on sys.path; add the repo root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from src.contract import K, N_ITEMS, N_USERS, read_canon_table

# =============================================================================
# Constants and artifact lookup
# =============================================================================
MODELS = ("Item-CF", "LightFM", "LRURec")
MODEL_SLUG = {"Item-CF": "itemcf", "LightFM": "lightfm", "LRURec": "lrurec"}
RESULTS_DIR = ROOT / "results"
OUTPUTS_DIR = ROOT / "outputs"


def _first_existing(candidates: list[Path]) -> Path | None:
    for p in candidates:
        if p.is_file():
            return p
    return None


@st.cache_data(show_spinner=False)
def find_metrics(slug: str) -> Path | None:
    # 先看 results/：那是进仓库的版本，所有人一致；outputs/ 只是本地临时产物
    return _first_existing([
        RESULTS_DIR / slug / f"{slug}_test.json",
        RESULTS_DIR / slug / "test.json",
        OUTPUTS_DIR / "metrics" / f"{slug}_test.json",
    ])


@st.cache_data(show_spinner=False)
def find_top10(slug: str) -> Path | None:
    # 同上：results/ 优先，保证不同机器上看到的是同一份推荐
    return _first_existing([
        RESULTS_DIR / slug / "top10.parquet",
        RESULTS_DIR / slug / f"{slug}_top10.parquet",
        OUTPUTS_DIR / "top10" / f"{slug}.parquet",
        OUTPUTS_DIR / f"{slug}_top10.parquet",
    ])


# =============================================================================
# Loaders (all reads go through the canonical contract entry point)
# =============================================================================
@st.cache_data(show_spinner="Loading catalogue…")
def load_catalog():
    movies = read_canon_table("movies").sort_values("iid").reset_index(drop=True)
    users = read_canon_table("users").sort_values("uid").reset_index(drop=True)
    return movies, users


@st.cache_data(show_spinner="Loading rating history…")
def load_ratings():
    return read_canon_table("ratings")


@st.cache_data(show_spinner="Loading recommendations…")
def load_top10(slug: str):
    path = find_top10(slug)
    if path is None:
        return None
    df = pd.read_parquet(path)
    if not {"uid", "rank", "iid"}.issubset(df.columns):
        return None
    if "score" not in df.columns:
        df = df.assign(score=float("nan"))
    return df[["uid", "rank", "iid", "score"]]


@st.cache_data(show_spinner=False)
def load_metrics(slug: str) -> dict | None:
    path = find_metrics(slug)
    if path is None:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("reports", {})
    except Exception:  # noqa: BLE001
        return None


# =============================================================================
# Rendering helpers
# =============================================================================
def _fmt_genres(gs) -> str:
    return " / ".join(gs) if isinstance(gs, (list, tuple)) else str(gs)


def recommend_table(df_top10: pd.DataFrame, uid: int, movies: pd.DataFrame) -> pd.DataFrame:
    sub = df_top10[df_top10["uid"] == uid].sort_values("rank").head(K)
    if sub.empty:
        return pd.DataFrame()
    m = movies.set_index("iid")
    rows = []
    for r in sub.itertuples(index=False):
        iid = int(r.iid)
        if iid not in m.index:
            continue
        rows.append({
            "Rank": int(r.rank),
            "Title": m.at[iid, "title"],
            "Year": int(m.at[iid, "year"]),
            "Genres": _fmt_genres(m.at[iid, "genres"]),
            "Score": round(float(r.score), 4) if pd.notna(r.score) else None,
        })
    return pd.DataFrame(rows)


def metrics_table() -> pd.DataFrame:
    rows = []
    for name in MODELS:
        rep = load_metrics(MODEL_SLUG[name])
        if not rep:
            rows.append({"Model": name, "Full R@10": None, "Full NDCG@10": None,
                         "Cand R@10": None, "Cand NDCG@10": None})
            continue
        f, c = rep.get("full", {}), rep.get("candidates", {})
        rows.append({
            "Model": name,
            "Full R@10": round(f.get("recall", float("nan")), 4),
            "Full NDCG@10": round(f.get("ndcg", float("nan")), 4),
            "Cand R@10": round(c.get("recall", float("nan")), 4),
            "Cand NDCG@10": round(c.get("ndcg", float("nan")), 4),
        })
    df = pd.DataFrame(rows).set_index("Model")
    return df.astype(object).where(df.notna(), "—")


def history_block(uid: int, ratings: pd.DataFrame, movies: pd.DataFrame) -> pd.DataFrame:
    m = movies.set_index("iid")
    hist = ratings[ratings["uid"] == uid].sort_values("ts").tail(10)
    rows = []
    for r in hist.itertuples(index=False):
        iid = int(r.iid)
        rows.append({
            "Title": m.at[iid, "title"] if iid in m.index else f"iid={iid}",
            "Genres": _fmt_genres(m.at[iid, "genres"]) if iid in m.index else "",
            "Rating": int(r.rating),
            "Split": str(r.split),
        })
    return pd.DataFrame(rows)


# =============================================================================
# Page
# =============================================================================
def main() -> None:
    st.set_page_config(page_title="MovieLens Recommender — Demo", page_icon="🎬", layout="wide")

    st.title("🎬 MovieLens Movie Recommender")
    st.caption(
        f"Three models under one frozen protocol · "
        f"**{N_USERS:,} users / {N_ITEMS:,} movies / K = {K}** · "
        "the interface reads recommendation files only, it does not load any model"
    )

    movies, users = load_catalog()
    ratings = load_ratings()

    # ---- headline results ----
    with st.expander("📊 Main results (held-out test set, K = 10)", expanded=False):
        st.dataframe(metrics_table(), use_container_width=True)
        st.caption(
            "**Full** ranks against all 3,416 movies (primary view). "
            "**Cand** ranks inside 1 positive + 100 negatives (secondary view). "
            "The two views are not comparable in absolute value. "
            "`Recall@10 == HR@10` follows from leave-one-out with a single positive."
        )
        st.caption(
            "Validation was used only to choose configurations; every number above is test."
        )

    # ---- artifact status ----
    status = {n: find_top10(MODEL_SLUG[n]) is not None for n in MODELS}
    cols = st.columns(len(MODELS))
    for col, name in zip(cols, MODELS):
        col.metric(label=f"{name} recommendations",
                   value="✅ ready" if status[name] else "⬜ missing")
    missing = [n for n, ok in status.items() if not ok]
    if missing:
        st.warning(
            f"No `top10.parquet` for **{', '.join(missing)}**, so those columns show a placeholder. "
            "Drop the file in and refresh — no code change needed.",
            icon="⚠️",
        )

    tab_user, tab_profile = st.tabs(["1. Pick a user", "2. Describe a user (sample profile)"])

    # =====================================================================
    # Tab 1 — a real user, three models side by side
    # =====================================================================
    with tab_user:
        st.markdown("Pick a user and see **what each of the three models recommends**.")

        c1, c2 = st.columns([2, 1])
        with c1:
            uid = st.number_input(
                f"User id (0 – {N_USERS - 1}; this is the internal id, not the raw UserID)",
                min_value=0, max_value=N_USERS - 1, step=1, key="uid",
            )
        with c2:
            st.write("")
            st.write("")

            def _randomize() -> None:
                # on_click runs before the widget is rendered, so this is safe
                st.session_state["uid"] = int(random.randrange(N_USERS))

            st.button("🎲 Random user", use_container_width=True, on_click=_randomize)

        uid = int(uid)
        if uid in users["uid"].values:
            urow = users[users["uid"] == uid].iloc[0]
            st.markdown(
                f"**uid {uid}** · {urow['gender']} · {urow['age_group']} · "
                f"{urow['occupation']} · {int((ratings['uid'] == uid).sum())} ratings"
            )

        with st.expander("Last 10 ratings of this user (models only ever saw the train split)"):
            st.dataframe(history_block(uid, ratings, movies),
                         use_container_width=True, hide_index=True)

        st.divider()
        cols = st.columns(len(MODELS))
        for col, name in zip(cols, MODELS):
            slug = MODEL_SLUG[name]
            with col:
                st.subheader(name)
                df_top10 = load_top10(slug)
                if df_top10 is None:
                    st.info(
                        f"No recommendations for `{slug}` yet.\n\n"
                        f"Expected `results/{slug}/top10.parquet` "
                        f"or `outputs/top10/{slug}.parquet`.",
                        icon="⬜",
                    )
                    continue
                tbl = recommend_table(df_top10, uid, movies)
                if tbl.empty:
                    st.warning("This model has no recommendation for this user.")
                else:
                    st.dataframe(tbl, use_container_width=True, hide_index=True)

        st.caption(
            "Recommendations exclude every movie the user has rated "
            "(train ∪ val ∪ test), matching the brief: "
            "“recommend movies that the user has not rated”."
        )

    # =====================================================================
    # Tab 2 — a sample profile (cold start approximation)
    # =====================================================================
    with tab_profile:
        st.markdown(
            "Describe a **new user** with zero interactions.\n\n"
            "> Item-CF cannot consume gender, age or genres — that is the cold-start problem. "
            "This tab uses an approximation: find a matching cohort, borrow their taste as a "
            "pseudo-history, then let Item-CF expand it."
        )

        p1, p2, p3 = st.columns([1, 1, 3])
        with p1:
            gender_label = st.radio("Gender", ["Any", "Male", "Female"], horizontal=True)
        with p2:
            age_options = ["Any"] + list(dict.fromkeys(users["age_group"].astype(str)))
            age_label = st.selectbox("Age group", age_options)
        with p3:
            genre_options = sorted({g for gs in movies["genres"] for g in gs})
            genres = st.multiselect("Preferred genres", genre_options,
                                    default=["Comedy", "Romance"])

        top_n = st.slider("Number of recommendations", 5, 20, K)

        if st.button("🎯 Generate recommendations", type="primary"):
            gender = {"Any": None, "Male": "M", "Female": "F"}[gender_label]
            age = None if age_label == "Any" else age_label
            with st.spinner("Building the cohort and expanding through Item-CF…"):
                try:
                    from src.demo_profile import Profile, recommend_for_profile
                    res = recommend_for_profile(Profile(gender, age, tuple(genres)),
                                                top_n=top_n)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Could not generate recommendations: {exc}")
                    st.stop()

            st.success(f"Matched cohort: **{res['cohort_size']:,}** users "
                       f"({' / '.join(res['notes'])})")

            m = movies.set_index("iid")
            rows = []
            for i, iid in enumerate(res["top"], 1):
                iid = int(iid)
                rows.append({
                    "Rank": i,
                    "Title": m.at[iid, "title"],
                    "Year": int(m.at[iid, "year"]),
                    "Genres": _fmt_genres(m.at[iid, "genres"]),
                    "Score": round(float(res["scores"][iid]), 4),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            with st.expander("How this was computed (the borrowed pseudo-history)"):
                st.markdown(
                    "The 30 highest-affinity movies of the cohort are treated as the "
                    "'history' of this hypothetical user, then spread through the "
                    "item-item similarity matrix."
                )
                prow = []
                for i, iid in enumerate(res["pseudo"][:15], 1):
                    iid = int(iid)
                    prow.append({
                        "#": i,
                        "Title": m.at[iid, "title"],
                        "Genres": _fmt_genres(m.at[iid, "genres"]),
                        "Cohort affinity": round(float(res["affinity"][iid]), 3),
                        "Rated by": int(res["n_raters"][iid]),
                    })
                st.dataframe(pd.DataFrame(prow), use_container_width=True, hide_index=True)

            st.caption(
                "This is an **approximation** of cold start and doubles as a control baseline. "
                "The model that natively consumes user features is LightFM, which represents a "
                "user as a sum of attribute embeddings."
            )

    st.divider()
    st.caption(
        "Data: MovieLens-1M (GroupLens) · cite Harper & Konstan, ACM TiiS 5(4), 2015, "
        "DOI 10.1145/2827872 · the raw data may not be redistributed"
    )


if __name__ == "__main__":
    main()
