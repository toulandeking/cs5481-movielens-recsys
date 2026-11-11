# Presentation Plan — 5 minutes talk + 5 minutes Q&A

> **Slides: English. Speaker notes: English, with short Chinese cues.**

> Covers only the models, the results and our thoughts — no code, as instructed.

> Generated from `slides/content.py`. Edit that file, then re-run `python slides/build_notes.py`.


**Planned speaking time: 4:50 / 5:00**


---

## 1. Timing at a glance

| # | Slide | Time | Cumulative |
|---:|---|---:|---:|
| 1 | Movie Recommendation on MovieLens-1M | 10s | 0:10 |
| 2 | Task and evaluation protocol | 30s | 0:40 |
| 3 | The three models | 45s | 1:25 |
| 4 | Results: full ranking, held-out test set | 45s | 2:10 |
| 5 | Thought 1 — feature design beat tuning by 6× | 50s | 3:00 |
| 6 | Thought 2 — each model wins a different slice | 45s | 3:45 |
| 7 | Thought 3 — validation flatters, so we report test only | 40s | 4:25 |
| 8 | What we take away | 25s | 4:50 |
| — | **Total** | **290s** | **4:50** |


> Leave ~10s of slack. If you run long, cut Thought 3 (slide 7) — it is the least visual and can survive as a spoken sentence.


---

## 2. Slide by slide

### Slide 1 — Movie Recommendation on MovieLens-1M  ·  `10s`

*On-slide kicker:* CS5481 Recommender Systems  ·  Group Project


*Bullets:*

- Three recommenders, one frozen evaluation protocol
- Item-based CF  ·  LightFM  ·  LRURec (WSDM 2024)
- What we will cover: the models, the results, and what we learned from them


**Speaker notes (EN)**

> Good morning. We built a movie recommender on MovieLens-1M and compared three models under one fixed protocol. I will skip the implementation and focus on the models, the numbers, and what those numbers actually tell us.


**中文提示**：10 秒，只说三句：做了什么、三个模型、接下来讲什么。不要念标题。


---

### Slide 2 — Task and evaluation protocol  ·  `30s`

*On-slide kicker:* One split, one metric path, one candidate set — for all three models


*Bullets:*

- Task: rank 10 movies the user has not rated, put the held-out item in the top 10
- Leave-one-out: last interaction = test, second-to-last = validation
- 6,040 users  ·  3,416 movies  ·  987,531 training interactions
- Implicit protocol: every observed interaction is positive
  - 41.8% of test targets are rated 3 stars or less — a rating threshold would mark them irrelevant
- Two views of the same score matrix: full ranking over 3,416 items (primary), 1 + 100 candidates (secondary)


**Speaker notes (EN)**

> The task is top-10 recommendation with leave-one-out: the user's last interaction is held out and must be ranked inside the top 10. Two protocol decisions matter for reading every number that follows. First, we treat all observed interactions as positive, because 41.8 percent of the held-out targets are rated 3 stars or less — if we only counted 4 stars and above, those targets would be labelled as irrelevant and the metric would be measuring the wrong thing. Second, all three models score the same matrix, so the full ranking and the 101-candidate view come from one computation, not two.


**中文提示**：30 秒。只强调两点：为什么用隐式协议（41.8% 这个数字），以及两种评估来自同一分数矩阵。


---

### Slide 3 — The three models  ·  `45s`

*On-slide kicker:* Different families, deliberately: neighborhood, hybrid factorization, sequence


*Table:*

| Model | Family | Signal used | Why this model |
|---|---|---|---|
| Item-CF | Neighborhood | item-item co-occurrence | the lecture baseline |
| LightFM | Hybrid matrix factorization | metadata embeddings + identity, WARP loss | reference algorithm; metadata is its selling point |
| LRURec | Sequential (linear RNN) | ordered history, not a bag of items | 2024 method; uses the order the other two ignore |


*Bullets:*

- Same interaction matrix for all three; only the inductive bias differs


**Speaker notes (EN)**

> We deliberately picked three different families. Item-CF is the lecture baseline: it only knows which movies co-occur. LightFM is the reference algorithm, and its selling point is metadata: every user and every movie is represented as a sum of feature embeddings, so it can score something it has seen very little of. LRURec is the 2024 method and the only one that uses order — it reads the history as a sequence, whereas the other two treat it as an unordered bag. That difference in what each model can see is what explains the results on the next slide.


**中文提示**：50 秒。重点不是介绍三个模型，而是点出「三者能看到的信号不同」——为后面解释差距做铺垫。


---

### Slide 4 — Results: full ranking, held-out test set  ·  `45s`

*On-slide kicker:* All numbers on the same split, the same candidates and the same metric code


*Figure:* `slides/figs/results_full.png`


*Table:*

| Model | Recall@10 | NDCG@10 | vs Item-CF |
|---|---|---|---|
| Item-CF | 0.0641 | 0.0335 | — |
| LightFM | 0.0775 | 0.0398 | +20.9% |
| LRURec | 0.2272 | 0.1233 | +254% |


**Speaker notes (EN)**

> Here is the main table. Item-CF reaches 0.0641 recall at 10, LightFM 0.0775, and LRURec 0.2272. The popularity baseline is 0.0353, so every model beats it, but they are not in the same league: LRURec is about 2.9 times LightFM. The gain is consistent in NDCG as well, so it is not an artefact of one metric. Before I explain why, note that these three numbers are not a fair fight — they were given different information. LRURec sees order; the other two do not. So the interesting question is not who won, but where each of them wins.


**中文提示**：45 秒。念表 → 点出 2.9 倍 → 立刻说明「信息量不同，不是公平的架构对比」→ 引出下一张图。


---

### Slide 5 — Thought 1 — feature design beat tuning by 6×  ·  `50s`

*On-slide kicker:* LightFM, 20 configurations, all selected on validation only


*Figure:* `slides/figs/lightfm_feature_vs_tuning.png`


*Bullets:*

- 13 hyperparameter configurations stayed between 0.0119 and 0.0159
  - Their entire spread is smaller than one standard error (0.0076)
- Adding user and item identity columns took it to 0.077–0.093
- Root cause: metadata alone produces only 241 distinct user vectors for 6,040 users


**Speaker notes (EN)**

> This is the finding I find most useful. We swept twenty LightFM configurations on the validation set. The first thirteen varied components, learning rate, epochs and regularisation — and every single one landed between 0.0119 and 0.0159. The whole spread is smaller than one standard error, so tuning was doing nothing. Then we added one feature group: an identity column for each user and each movie. Recall jumped to between 0.077 and 0.093. The reason is measurable: with metadata only, six thousand and forty users collapse into two hundred and forty-one distinct feature vectors, so on average twenty-five users are literally indistinguishable to the model. It cannot rank for an individual. The lesson is that on sparse side information the bottleneck is the feature space, not the hyperparameters.


**中文提示**：55 秒，本场最重要的一页。三个动作：指左边那坨点 → 说「比一个标准误还小」→ 指箭头说 6 倍 → 给 241 这个根因。


---

### Slide 6 — Thought 2 — each model wins a different slice  ·  `45s`

*On-slide kicker:* Hit rate grouped by how popular the held-out target is


*Figure:* `slides/figs/popularity_by_model.png`


*Bullets:*

- Below 100 ratings, every model is essentially at zero
- Gains concentrate in the mid-tail: 94% of LightFM's improvement comes from 100–1,499 ratings
- At the head (1,500+ ratings) all three converge — order and metadata add little


**Speaker notes (EN)**

> This chart slices the same test set by how many training ratings the target movie has. Two things stand out. First, below one hundred ratings every model is essentially at zero — including the strongest one. Cold movies are not solved by any of our models. Second, the gains are concentrated in the mid-tail: ninety-four percent of LightFM's improvement over Item-CF comes from movies with one hundred to fifteen hundred ratings. At the head, all three converge, because those movies are guessed by popularity alone. So the honest conclusion is that better modelling power buys you the middle of the distribution, not the tail.


**中文提示**：50 秒。两个动作：指最左边三根柱子「都是 0」→ 指中间三组「增益全在中段」。结论句务必说：能力买到的是中段，不是长尾。


---

### Slide 7 — Thought 3 — validation flatters, so we report test only  ·  `40s`

*On-slide kicker:* A prediction made from the data, then confirmed by two independent models


*Bullets:*

- Validation is systematically optimistic: LightFM 0.0919 (val) vs 0.0775 (test), a gap of 0.0144
- We predicted a gap of about 0.015 from the split alone, before running anything
- Two models confirmed it independently: 0.0144 (LightFM) and 0.0145 (LRURec)
- Consequence: validation is used only to choose configurations; the test set is evaluated once
- Caveats we state openly: LRURec trained for 5 epochs under a fixed budget, and its test run is a repeat on a set examined earlier


**Speaker notes (EN)**

> One methodological point we care about. Validation numbers are systematically optimistic on this split: LightFM scores 0.0919 on validation but 0.0775 on test. We had predicted a gap of about 0.015 from the data characteristics before running anything, and two independent models landed on 0.0144 and 0.0145. So validation is only used to choose configurations, and the test set is touched once. We also state our caveats rather than hiding them: LRURec was trained for five epochs under a fixed budget, so it is not converged, and its test run is a repeat on a test set we had already examined — so it is not a fresh unseen test.


**中文提示**：40 秒。核心是「预测→验证」这个动作；然后主动交代 LRURec 两个短板（5 epoch、重复实验）——主动说比被问出来好。


---

### Slide 8 — What we take away  ·  `25s`

*On-slide kicker:* Three lessons and one open problem


*Bullets:*

- Comparability is a feature, not overhead: one split, one metric path, one candidate set
- On sparse side information, feature design matters more than hyperparameter tuning
- Model choice is regime-dependent — head, mid-tail and cold items reward different models
- Still open: zero-interaction items. Genres are too coarse; the title text is the next place to look


**Speaker notes (EN)**

> To summarise: comparability came first and made the rest possible; feature design beat tuning by a wide margin; and model choice depends on which regime you care about. The open problem is cold items — nothing we tried moves them, and genres alone are too coarse. Using the title text is where we would go next. Thank you — happy to take questions.


**中文提示**：20 秒收尾。最后一句「happy to take questions」要说得清楚，然后停住，等主持人接。


---

## 3. Q&A preparation (5 minutes)

Answer in one breath: *concede what is true, then give the number.*

**Q: Why is Recall@10 equal to HR@10?**

A: Because leave-one-out gives exactly one positive per user. Recall over one relevant item is either 0 or 1, which is the definition of hit rate. We report both names because papers use both.

**Q: Isn't LRURec's advantage unfair?**

A: It is not an architecture-only comparison. All three share the split, the candidates and the metric code, so the numbers are comparable — but they were given different information: LRURec sees order. We report the gap as a property of the information, not as 'sequential beats CF'.

**Q: Only 5 epochs for LRURec?**

A: Correct, and we do not claim convergence. Twenty configurations ran under one fixed budget so that the comparison between configurations is fair. More epochs would likely raise the number; we did not use that as a result.

**Q: Doesn't adding identity features turn LightFM into plain matrix factorisation?**

A: Effectively it becomes a hybrid model: identity columns are free trainable vectors and metadata acts as a shared prior. We report the metadata-only number as well (validation 0.0151), so the difference is visible rather than hidden.

**Q: Why is Item-CF so low?**

A: It ranks against 3,416 candidates with a single positive per user, which is hard in absolute terms. It is not weak at the head — in the 1,500+ bucket it reaches 0.280, within 0.03 of LightFM.

**Q: What would you do with more time?**

A: A cold-item hold-out experiment: remove all ratings of a random 20% of warm movies, retrain, and measure whether metadata can recover them. And use the movie-title text, since genres are too coarse.


---

## 4. Rehearsal checklist

- [ ] Read the speaker notes aloud once; mark words you stumble on and simplify them
- [ ] Time it with a stopwatch — target **4:45**, hard stop at 5:00
- [ ] For each figure, say *what to look at* before *what it means* (point, then interpret)
- [ ] Prepare the Streamlit window in advance; do not debug the demo live
- [ ] Decide who answers which Q&A area (results / method / models)
- [ ] Have the number sheet in hand: 0.0641 · 0.0775 · 0.2272 · baseline 0.0353
