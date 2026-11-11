"""
slides/content.py  —  Single source of truth for the slide deck and the speaker notes.
=====================================================================================

Edit here once; `build_deck.py` (PPTX) and `build_notes.py` (Markdown) both read it.

Constraints from the instructor
-------------------------------
* 5 minutes talk + 5 minutes Q&A
* NO code on slides
* Only: the models applied, the experimental results, and thoughts about them
* Slides in English. Speaker notes: English primary, Chinese only as short hints.

Field meanings
--------------
title     : slide headline (English)
kicker    : small line above the title (English)
bullets   : list of (text, level) — level 0 = main bullet, 1 = sub bullet
table     : optional dict(header=[...], rows=[[...]], widths=[...])
figure    : optional file name under slides/figs/
seconds   : planned speaking time
notes     : full speaker script (English)
hints     : short Chinese cues for the speaker (not shown on slides)
"""

from __future__ import annotations

TOTAL_SECONDS = 300

SLIDES: list[dict] = [
    # ---------------------------------------------------------------- 1
    {
        "title": "Movie Recommendation on MovieLens-1M",
        "kicker": "CS5481 Recommender Systems  ·  Group Project",
        "dark": True,
        "bullets": [
            ("Three recommenders, one frozen evaluation protocol", 0),
            ("Item-based CF  ·  LightFM  ·  LRURec (WSDM 2024)", 0),
            ("", 0),
            ("What we will cover: the models, the results, and what we learned from them", 0),
        ],
        "seconds": 14,
        "notes": (
            "Good morning. We built a movie recommender on MovieLens-1M and compared three "
            "models under one fixed protocol. I will skip the implementation and focus on "
            "the models, the numbers, and what those numbers actually tell us."
        ),
        "hints": "10 秒，只说三句：做了什么、三个模型、接下来讲什么。不要念标题。",
    },
    # ---------------------------------------------------------------- 2
    {
        "title": "Task and evaluation protocol",
        "kicker": "One split, one metric path, one candidate set — for all three models",
        "bullets": [
            ("Task: rank 10 movies the user has not rated, put the held-out item in the top 10", 0),
            ("Leave-one-out: last interaction = test, second-to-last = validation", 0),
            ("6,040 users  ·  3,416 movies  ·  987,531 training interactions", 0),
            ("Implicit protocol: every observed interaction is positive", 0),
            ("41.8% of test targets are rated 3 stars or less — a rating threshold would mark them irrelevant", 1),
            ("Two views of the same score matrix: full ranking over 3,416 items (primary), 1 + 100 candidates (secondary)", 0),
        ],
        "seconds": 37,
        "notes": (
            "The task is top-10 recommendation with leave-one-out: the user's last interaction "
            "is held out and must be ranked inside the top 10. "
            "Two protocol decisions matter for reading every number that follows. "
            "First, we treat all observed interactions as positive, because 41.8 percent of the "
            "held-out targets are rated 3 stars or less — if we only counted 4 stars and above, "
            "those targets would be labelled as irrelevant and the metric would be measuring "
            "the wrong thing. "
            "Second, all three models score the same matrix, so the full ranking and the "
            "101-candidate view come from one computation, not two."
        ),
        "hints": "30 秒。只强调两点：为什么用隐式协议（41.8% 这个数字），以及两种评估来自同一分数矩阵。",
    },
    # ---------------------------------------------------------------- 3
    {
        "title": "The three models",
        "kicker": "Different families, deliberately: neighborhood, hybrid factorization, sequence",
        "table": {
            "header": ["Model", "Family", "Signal used", "Why this model"],
            "widths": [2.0, 3.1, 3.6, 3.9],
            "rows": [
                ["Item-CF", "Neighborhood", "item-item co-occurrence", "the lecture baseline"],
                ["LightFM", "Hybrid matrix factorization",
                 "metadata embeddings + identity, WARP loss", "reference algorithm; metadata is its selling point"],
                ["LRURec", "Sequential (linear RNN)", "ordered history, not a bag of items",
                 "2024 method; uses the order the other two ignore"],
            ],
        },
        "bullets": [
            ("Same interaction matrix for all three; only the inductive bias differs", 0),
        ],
        "seconds": 35,
        "notes": (
            "We deliberately picked three different families. "
            "Item-CF is the lecture baseline: it only knows which movies co-occur. "
            "LightFM is the reference algorithm, and its selling point is metadata: every user "
            "and every movie is represented as a sum of feature embeddings, so it can score "
            "something it has seen very little of. "
            "LRURec is the 2024 method and the only one that uses order — it reads the history "
            "as a sequence, whereas the other two treat it as an unordered bag. "
            "That difference in what each model can see is what explains the results on the next slide."
        ),
        "hints": "50 秒。重点不是介绍三个模型，而是点出「三者能看到的信号不同」——为后面解释差距做铺垫。",
    },
    # ---------------------------------------------------------------- 4
    {
        "title": "Results: full ranking, held-out test set",
        "kicker": "All numbers on the same split, the same candidates and the same metric code",
        "figure": "results_full.png",
        "fig_w": 6.75,
        "table": {
            "header": ["Model", "Recall@10", "NDCG@10", "vs Item-CF"],
            "widths": [2.2, 2.2, 2.4, 2.6],
            "font": 14,
            "rows": [
                ["Item-CF", "0.0641", "0.0335", "—"],
                ["LightFM", "0.0775", "0.0398", "+20.9%"],
                ["LRURec", "0.2272", "0.1233", "+254%"],
            ],
        },
        "seconds": 34,
        "notes": (
            "Here is the main table. Item-CF reaches 0.0641 recall at 10, LightFM 0.0775, "
            "and LRURec 0.2272. The popularity baseline is 0.0353, so every model beats it, "
            "but they are not in the same league: LRURec is about 2.9 times LightFM. "
            "The gain is consistent in NDCG as well, so it is not an artefact of one metric. "
            "Before I explain why, note that these three numbers are not a fair fight — they "
            "were given different information. LRURec sees order; the other two do not. "
            "So the interesting question is not who won, but where each of them wins."
        ),
        "hints": "45 秒。念表 → 点出 2.9 倍 → 立刻说明「信息量不同，不是公平的架构对比」→ 引出下一张图。",
    },
    # ---------------------------------------------------------------- 5
    {
        "title": "Thought 1 — feature design beat tuning by 6×",
        "kicker": "LightFM, 20 configurations, all selected on validation only",
        "figure": "lightfm_feature_vs_tuning.png",
        "caption": (
            "Validation sweep, 20 configurations · the test set was evaluated once, "
            "with the configuration selected on validation"
        ),
        "bullets": [
            ("13 hyperparameter configurations stayed between 0.0119 and 0.0159", 0),
            ("Their entire spread is smaller than one standard error (0.0076)", 1),
            ("Adding user and item identity columns took it to 0.077–0.093", 0),
            ("Root cause: metadata alone produces only 241 distinct user vectors for 6,040 users", 0),
        ],
        "seconds": 44,
        "notes": (
            "This is the finding I find most useful. We swept twenty LightFM configurations on "
            "the validation set. The first thirteen varied components, learning rate, epochs and "
            "regularisation — and every single one landed between 0.0119 and 0.0159. The whole "
            "spread is smaller than one standard error, so tuning was doing nothing. "
            "Then we added one feature group: an identity column for each user and each movie. "
            "Recall jumped to between 0.077 and 0.093. "
            "The reason is measurable: with metadata only, six thousand and forty users collapse "
            "into two hundred and forty-one distinct feature vectors, so on average twenty-five "
            "users are literally indistinguishable to the model. It cannot rank for an individual. "
            "The lesson is that on sparse side information the bottleneck is the feature space, "
            "not the hyperparameters."
        ),
        "hints": "55 秒，本场最重要的一页。三个动作：指左边那坨点 → 说「比一个标准误还小」→ 指箭头说 6 倍 → 给 241 这个根因。",
    },
    # ---------------------------------------------------------------- 6
    {
        "title": "Thought 2 — each model wins a different slice",
        "kicker": "Hit rate grouped by how popular the held-out target is",
        "figure": "popularity_by_model.png",
        "bullets": [
            ("Below 100 ratings, every model is essentially at zero", 0),
            ("Gains concentrate in the mid-tail: 94% of LightFM's improvement comes from 100–1,499 ratings", 0),
            ("At the head (1,500+ ratings) all three converge — order and metadata add little", 0),
        ],
        "seconds": 36,
        "notes": (
            "This chart slices the same test set by how many training ratings the target movie has. "
            "Two things stand out. First, below one hundred ratings every model is essentially at "
            "zero — including the strongest one. Cold movies are not solved by any of our models. "
            "Second, the gains are concentrated in the mid-tail: ninety-four percent of LightFM's "
            "improvement over Item-CF comes from movies with one hundred to fifteen hundred ratings. "
            "At the head, all three converge, because those movies are guessed by popularity alone. "
            "So the honest conclusion is that better modelling power buys you the middle of the "
            "distribution, not the tail."
        ),
        "hints": "50 秒。两个动作：指最左边三根柱子「都是 0」→ 指中间三组「增益全在中段」。结论句务必说：能力买到的是中段，不是长尾。",
    },
    # ---------------------------------------------------------------- 7
    {
        "title": "Thought 3 — validation flatters, so we report test only",
        "kicker": "A prediction made from the data, then confirmed by two independent models",
        "bullets": [
            ("Validation is systematically optimistic: LightFM 0.0919 (val) vs 0.0775 (test), a gap of 0.0144", 0),
            ("We predicted a gap of about 0.015 from the split alone, before running anything", 0),
            ("Two models confirmed it independently: 0.0144 (LightFM) and 0.0145 (LRURec)", 0),
            ("Consequence: validation is used only to choose configurations; the test set is evaluated once", 0),
            ("Caveats we state openly: LRURec trained for 5 epochs under a fixed budget, and its test run is a repeat on a set examined earlier", 0),
        ],
        "seconds": 35,
        "notes": (
            "One methodological point we care about. Validation numbers are systematically "
            "optimistic on this split: LightFM scores 0.0919 on validation but 0.0775 on test. "
            "We had predicted a gap of about 0.015 from the data characteristics before running "
            "anything, and two independent models landed on 0.0144 and 0.0145. "
            "So validation is only used to choose configurations, and the test set is touched once. "
            "We also state our caveats rather than hiding them: LRURec was trained for five epochs "
            "under a fixed budget, so it is not converged, and its test run is a repeat on a test "
            "set we had already examined — so it is not a fresh unseen test."
        ),
        "hints": "40 秒。核心是「预测→验证」这个动作；然后主动交代 LRURec 两个短板（5 epoch、重复实验）——主动说比被问出来好。",
    },
    # ---------------------------------------------------------------- 8
    {
        "title": "What we take away",
        "kicker": "Three lessons and one open problem",
        "bullets": [
            ("Comparability is a feature, not overhead: one split, one metric path, one candidate set", 0),
            ("On sparse side information, feature design matters more than hyperparameter tuning", 0),
            ("Model choice is regime-dependent — head, mid-tail and cold items reward different models", 0),
            ("Still open: zero-interaction items. Genres are too coarse; the title text is the next place to look", 0),
        ],
        "seconds": 20,
        "notes": (
            "To summarise: comparability came first and made the rest possible; feature design beat "
            "tuning by a wide margin; and model choice depends on which regime you care about. "
            "The open problem is cold items — nothing we tried moves them, and genres alone are too "
            "coarse. Using the title text is where we would go next. Thank you — happy to take questions."
        ),
        "hints": "20 秒收尾。最后一句「happy to take questions」要说得清楚，然后停住，等主持人接。",
    },
]

# ---------------------------------------------------------------------------
# Live demo segment — no slide of its own, played right after slide 4.
# The deck totals 255s; this adds 30s, for 4:45 of the 5:00 budget.
# ---------------------------------------------------------------------------
DEMO = {
    "label": "Live demo — Streamlit, one user, three models side by side",
    "seconds": 30,
    "after_slide": 4,
    "script": (
        "Let me show this working. I will pick a user, and you will see all three models "
        "rank movies for the same person, side by side. Same user, same catalogue, three "
        "very different lists — and the third one, the sequential model, is the one that "
        "most often recovers a title the user actually watched: 22.7 percent of users "
        "against 6.4 for the first column."
    ),
    "hint": "演示 30 秒：只点一次「随机换一个」，指着三列说同一句话就切回。不要现场调参、不要滚页面。",
    # 台上必须一眼看懂的，只保留这三块（列的定义放 Q&A）
    "on_screen": [
        ("Top panel", "three algorithms × two evaluation views — full ranking and 101-candidate"),
        ("Tab 1", "one user, three recommendation lists side by side, ranks 1–10"),
        ("The one thing to say", "only the ranks are comparable; each Score lives on its own scale"),
    ],
    # 精确动作，按顺序念给自己听
    "walkthrough": [
        "Click **🎲 Random user** exactly once. Nothing else on this page.",
        "Read the user line aloud: gender · age · occupation · number of ratings.",
        "Point at the three columns; name the gap: only the third list routinely recovers "
        "what the user actually watched — 22.7% of users against 6.4% for Item-CF.",
        "Stop. **Do not open the expanders on stage** — they are Q&A material, not demo material.",
    ],
}

# ---------------------------------------------------------------------------
# Anticipated questions (5-minute Q&A). Not shown on slides.
# ---------------------------------------------------------------------------
QA: list[tuple[str, str]] = [
    ("Why is Recall@10 equal to HR@10?",
     "Because leave-one-out gives exactly one positive per user. Recall over one relevant item is "
     "either 0 or 1, which is the definition of hit rate. We report both names because papers use both."),
    ("Isn't LRURec's advantage unfair?",
     "It is not an architecture-only comparison. All three share the split, the candidates and the "
     "metric code, so the numbers are comparable — but they were given different information: LRURec "
     "sees order. We report the gap as a property of the information, not as 'sequential beats CF'."),
    ("Only 5 epochs for LRURec?",
     "Correct, and we do not claim convergence. Twenty configurations ran under one fixed budget so "
     "that the comparison between configurations is fair. More epochs would likely raise the number; "
     "we did not use that as a result."),
    ("Doesn't adding identity features turn LightFM into plain matrix factorisation?",
     "Effectively it becomes a hybrid model: identity columns are free trainable vectors and metadata "
     "acts as a shared prior. We report the metadata-only number as well (validation 0.0151), so the "
     "difference is visible rather than hidden."),
    ("Why is Item-CF so low?",
     "It ranks against 3,416 candidates with a single positive per user, which is hard in absolute terms. "
     "It is not weak at the head — in the 1,500+ bucket it reaches 0.280, within 0.03 of LightFM."),
    ("What would you do with more time?",
     "A cold-item hold-out experiment: remove all ratings of a random 20% of warm movies, retrain, and "
     "measure whether metadata can recover them. And use the movie-title text, since genres are too coarse."),

    # ---- demo-screen questions (detail deliberately kept out of the demo block) ----
    ("Walk me through the results table at the top of the demo.",
     "**Full R@10 / Full NDCG@10** rank the held-out item against all 3,416 movies — that is the "
     "primary view and the one we quote. **Cand R@10 / Cand NDCG@10** rank it inside 1 positive plus "
     "100 sampled negatives; that is the secondary view, kept only for comparability with papers that "
     "use a 1+99 protocol. The two are not comparable in absolute value, because the candidate view "
     "has 33× fewer distractors."),
    ("Why are the three Score columns on completely different scales?",
     "Because each model's score means something different. **Item-CF** sums item-item similarities "
     "and lands around 1.8–4.2. **LightFM** is a dot product plus user and item biases, and under WARP "
     "those biases drift together, so the score matrix sits around −751 to −290 — negative values are "
     "expected. **LRURec** is the dot product of a sequence state with an item embedding. Only the "
     "ordering is meaningful, so we compare ranks and never raw scores across columns."),
    ("What is the Split column in the history table?",
     "It shows where each rating lives in our leave-one-out protocol. `train` is what every model "
     "learns from; `val` was used only to choose hyper-parameters; `test` is the single held-out rating "
     "that the metric tries to rank inside the top ten. A model never sees its own `test` row — we show "
     "it on screen so the audience can see what is being predicted."),
    ("What do Cohort affinity and Rated by mean in Tab 2?",
     "**Cohort affinity** = Σ max(rating − 3, 0) ÷ (number of raters + 10) across the matched cohort. "
     "Only 4- and 5-star ratings count, and the +10 shrinkage stops a movie with three enthusiastic "
     "raters from topping the list. **Rated by** is how many cohort members rated that movie; anything "
     "below 15 raters is dropped before ranking."),
    ("What happens if a profile matches too few users?",
     "The cohort builder relaxes conditions in a fixed order — first drop the genre filter, then the "
     "age filter — and says so in the UI, for example 'genre filter too strict (21 users) — relaxed to "
     "any genre'. It never silently proceeds on a cohort too small to be meaningful."),
    ("Some numbers in the demo table carry an asterisk. Why?",
     "LightFM's owner did not commit its metrics JSON, so the full-ranking numbers were recomputed by "
     "`src/verify_results.py` from the committed top-10 export — no model code involved. The "
     "candidate-mode numbers cannot be derived from a top-10 export at all, so those two are **cited** "
     "from the owner's documentation and marked with an asterisk. Everything else in the table is the "
     "owner's own evaluation output."),
    ("What do the ✅ / ⬜ cards under the title mean?",
     "They only report whether that model's `top10.parquet` was found in the repository. Three ✅ means "
     "the demo can show all three lists; a ⬜ means that column will show a placeholder instead of "
     "fabricated recommendations."),
]
