"""
slides/build_figures.py  —  Generate all figures used by the slide deck.
======================================================================
All labels are English. Outputs go to slides/figs/.

Data sources (do not invent numbers):
  * full Recall@10 / NDCG@10   -> outputs/metrics/itemcf_test.json
                                  LIGHTFM.md  (LightFM; results/lightfm/ not in repo)
                                  results/lrurec/lrurec_test.json
  * popularity buckets (test)  -> outputs/metrics/itemcf_popularity_buckets.csv
                                  LIGHTFM.md  section 6
                                  results/lrurec/lrurec_popularity_buckets.csv
  * LightFM 20-config sweep    -> LIGHTFM.md  section 5

Usage:
    python slides/build_figures.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "figs"
OUT.mkdir(parents=True, exist_ok=True)

# Consistent palette across all slides
C_ITEMCF, C_LIGHTFM, C_LRUREC = "#4C78A8", "#F58518", "#54A24B"
MODELS = ["Item-CF", "LightFM", "LRURec"]
COLORS = [C_ITEMCF, C_LIGHTFM, C_LRUREC]

plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


# ---------------------------------------------------------------------------
# 1. Main results (full-ranking, test)
# ---------------------------------------------------------------------------
def fig_main_results() -> None:
    # ItemCF from its own JSON; LRURec from results/; LightFM from LIGHTFM.md section 4.1
    itemcf = json.loads((ROOT / "outputs/metrics/itemcf_test.json").read_text(encoding="utf-8"))["reports"]["full"]
    lrurec = json.loads((ROOT / "results/lrurec/lrurec_test.json").read_text(encoding="utf-8"))["reports"]["full"]

    recall = [itemcf["recall"], 0.0775, lrurec["recall"]]
    ndcg = [itemcf["ndcg"], 0.0398, lrurec["ndcg"]]

    x = np.arange(len(MODELS))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    b1 = ax.bar(x - w / 2, recall, w, label="Recall@10", color=COLORS, alpha=1.0)
    b2 = ax.bar(x + w / 2, ndcg, w, label="NDCG@10", color=COLORS, alpha=0.55)

    for bars in (b1, b2):
        for r in bars:
            ax.annotate(f"{r.get_height():.4f}",
                        xy=(r.get_x() + r.get_width() / 2, r.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=9)

    ax.axhline(0.0353, ls="--", lw=1, c="grey", label="popularity baseline (0.0353)")

    ax.set_xticks(x)
    ax.set_xticklabels(MODELS)
    ax.set_ylabel("score (full ranking, test)")
    ax.set_title("Full-ranking results: LRURec is ~2.9x LightFM")
    ax.legend(frameon=False, loc="upper left")
    ax.set_ylim(0, 0.26)
    fig.tight_layout()
    fig.savefig(OUT / "results_full.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Where each model wins: hit rate by target popularity (test)
# ---------------------------------------------------------------------------
def fig_popularity() -> None:
    """Bucket edges aligned to <20 / 20-99 / 100-499 / 500-1499 / >=1500.

    ItemCF and LightFM report <5 and 5-19 separately; both are 0.0000 on test,
    so merging them into <20 is exact (weights 2 and 65 users).
    """
    with open(ROOT / "outputs/metrics/itemcf_popularity_buckets.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    itemcf_test = {r["target_popularity"]: float(r["test_hit@10"]) for r in rows}

    with open(ROOT / "results/lrurec/lrurec_popularity_buckets.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    lrurec_test = {r["target_popularity"]: float(r["test_hit@10"]) for r in rows}

    # LightFM: LIGHTFM.md section 6 (test column)
    lightfm_test = {"<5": 0.0, "5-19": 0.0, "20-99": 0.0043,
                    "100-499": 0.0185, "500-1499": 0.0901, ">=1500": 0.2874}

    labels = ["<20", "20-99", "100-499", "500-1499", ">=1500"]
    counts = [67, 469, 2276, 2553, 675]

    itemcf = [itemcf_test["<5"], itemcf_test["20-99"], itemcf_test["100-499"],
              itemcf_test["500-1499"], itemcf_test[">=1500"]]
    lightfm = [lightfm_test["<5"], lightfm_test["20-99"], lightfm_test["100-499"],
               lightfm_test["500-1499"], lightfm_test[">=1500"]]
    lrurec = [lrurec_test["<20"], lrurec_test["20-99"], lrurec_test["100-499"],
              lrurec_test["500-1499"], lrurec_test[">=1500"]]

    x = np.arange(len(labels))
    w = 0.26
    fig, ax = plt.subplots(figsize=(7.6, 4.1))
    ax.bar(x - w, itemcf, w, label="Item-CF", color=C_ITEMCF)
    ax.bar(x, lightfm, w, label="LightFM", color=C_LIGHTFM)
    ax.bar(x + w, lrurec, w, label="LRURec", color=C_LRUREC)

    for xi, v in zip(x - w, itemcf):
        ax.annotate(f"{v:.3f}", xy=(xi, v), xytext=(0, 2), textcoords="offset points",
                    ha="center", fontsize=7, rotation=90)
    for xi, v in zip(x, lightfm):
        ax.annotate(f"{v:.3f}", xy=(xi, v), xytext=(0, 2), textcoords="offset points",
                    ha="center", fontsize=7, rotation=90)
    for xi, v in zip(x + w, lrurec):
        ax.annotate(f"{v:.3f}", xy=(xi, v), xytext=(0, 2), textcoords="offset points",
                    ha="center", fontsize=7, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\n(n={c:,})" for l, c in zip(labels, counts)], fontsize=9)
    ax.set_xlabel("target popularity in train (# ratings)")
    ax.set_ylabel("hit rate @10 (test)")
    ax.set_title("Cold targets stay unsolved: every model is ~0 below 100 ratings")
    ax.legend(frameon=False)
    ax.set_ylim(0, 0.42)
    fig.tight_layout()
    fig.savefig(OUT / "popularity_by_model.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. LightFM: hyperparameters are flat, features are not
# ---------------------------------------------------------------------------
def fig_lightfm_sweep() -> None:
    """LIGHTFM.md section 5: 20 configs, val Recall@10.

    Groups 1-13 = side information only, 14-20 = + identity features.
    """
    plain = [(1, 0.0151), (2, 0.0141), (3, 0.0149), (4, 0.0146),
             (5, 0.0151), (6, 0.0151), (7, 0.0119), (8, 0.0119),
             (9, 0.0159), (10, 0.0157), (11, 0.0149),
             (12, 0.0154), (13, 0.0144)]
    ident = [(14, 0.0773), (15, 0.0805), (16, 0.0919),
             (17, 0.0929), (18, 0.0781), (19, 0.0844), (20, 0.0783)]

    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    px, py = zip(*plain)
    ix, iy = zip(*ident)
    ax.scatter(px, py, s=55, color=C_ITEMCF, label="metadata only (13 configs)", zorder=3)
    ax.scatter(ix, iy, s=55, color=C_LIGHTFM, label="+ user/item identity (7 configs)", zorder=3)

    ax.axhspan(0.0119, 0.0159, color=C_ITEMCF, alpha=0.10, zorder=0)
    ax.annotate("all 13 hyperparameter configs\nstay inside 0.0119-0.0159",
                xy=(7, 0.0135), xytext=(3.2, 0.045), fontsize=9, color=C_ITEMCF)

    ax.axhline(0.0353, ls="--", lw=1, c="grey")
    ax.annotate("popularity baseline", xy=(20, 0.0353), xytext=(0, 3),
                textcoords="offset points", fontsize=8, color="grey", ha="right")
    ax.axhline(0.0641, ls=":", lw=1, c=C_ITEMCF)
    ax.annotate("Item-CF 0.0641", xy=(20, 0.0641), xytext=(0, 3),
                textcoords="offset points", fontsize=8, color=C_ITEMCF, ha="right")

    ax.annotate("", xy=(14, 0.0773), xytext=(13, 0.0144),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.4))
    ax.annotate("one feature column\n= 6x", xy=(12.6, 0.055), fontsize=10, ha="right")

    ax.set_xlabel("configuration index (val sweep)")
    ax.set_ylabel("val Recall@10")
    ax.set_title("LightFM: the bottleneck is feature design, not tuning")
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    ax.set_ylim(0, 0.105)
    ax.set_xlim(0, 21)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_xticks(range(0, 22, 2))
    fig.tight_layout()
    fig.savefig(OUT / "lightfm_feature_vs_tuning.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_main_results()
    fig_popularity()
    fig_lightfm_sweep()
    for p in sorted(OUT.glob("*.png")):
        print(f"  {p.stat().st_size/1024:7.1f} KB  {p.relative_to(ROOT)}")
