"""
slides/build_notes.py  —  Render PRESENTATION.md from slides/content.py.
=======================================================================
No third-party dependency, so it can run before python-pptx is installed.

    python slides/build_notes.py
"""

from __future__ import annotations

from pathlib import Path

from content import DEMO, QA, SLIDES, TOTAL_SECONDS

ROOT = Path(__file__).resolve().parents[1]


def _mmss(sec: int) -> str:
    return f"{sec // 60}:{sec % 60:02d}"


def render() -> str:
    total = sum(s["seconds"] for s in SLIDES)
    grand = total + DEMO["seconds"]
    out: list[str] = []
    a = out.append

    a("# Presentation Plan — 5 minutes talk + 5 minutes Q&A\n")
    a("> **Slides: English. Speaker notes: English, with short Chinese cues.**\n")
    a("> Covers only the models, the results and our thoughts — no code, as instructed.\n")
    a("> Generated from `slides/content.py`. Edit that file, then re-run "
      "`python slides/build_notes.py`.\n")
    a(f"\n**Planned: {_mmss(total)} of slides + {DEMO['seconds']}s live demo "
      f"= {_mmss(grand)} / {_mmss(TOTAL_SECONDS)}**\n")
    a("\n---\n")

    # ---- timing table ----
    a("## 1. Timing at a glance\n")
    a("| # | Segment | Time | Cumulative |")
    a("|---:|---|---:|---:|")
    cum = 0
    for i, s in enumerate(SLIDES, 1):
        cum += s["seconds"]
        a(f"| {i} | {s['title']} | {s['seconds']}s | {_mmss(cum)} |")
        if i == DEMO["after_slide"]:
            cum += DEMO["seconds"]
            a(f"| ▶ | **{DEMO['label']}** | **{DEMO['seconds']}s** | **{_mmss(cum)}** |")
    a(f"| — | **Total** | **{grand}s** | **{_mmss(cum)}** |\n")
    a("\n> Target **4:45**, hard stop at 5:00. If you run long, cut Thought 3 (slide 7) — "
      "it is the least visual and survives as one spoken sentence.\n")
    a("\n---\n")

    # ---- per slide ----
    a("## 2. Slide by slide\n")
    for i, s in enumerate(SLIDES, 1):
        a(f"### Slide {i} — {s['title']}  ·  `{s['seconds']}s`\n")
        if s.get("kicker"):
            a(f"*On-slide kicker:* {s['kicker']}\n")

        if s.get("figure"):
            a(f"\n*Figure:* `slides/figs/{s['figure']}`\n")

        if s.get("table"):
            t = s["table"]
            a("\n*Table:*\n")
            a("| " + " | ".join(t["header"]) + " |")
            a("|" + "|".join(["---"] * len(t["header"])) + "|")
            for row in t["rows"]:
                a("| " + " | ".join(row) + " |")
            a("")

        bullets = s.get("bullets") or []
        if bullets:
            a("\n*Bullets:*\n")
            for text, level in bullets:
                if not text:
                    continue
                a(("  " * level) + "- " + text)
            a("")

        a("\n**Speaker notes (EN)**\n")
        a(f"> {s['notes']}\n")
        a(f"\n**中文提示**：{s['hints']}\n")
        a("\n---\n")

        if i == DEMO["after_slide"]:
            a(f"### ▶ Live demo — no slide (`{DEMO['seconds']}s`)\n")
            a("**Switch to the Streamlit window here.** Press `B` in PowerPoint to black "
              "the screen while you alt-tab, so the audience does not watch you fumble.\n")
            a("\n**Script (EN)**\n")
            a(f"> {DEMO['script']}\n")

            a("\n**On screen — the only three things the audience needs**\n")
            a("| Where | What it is |")
            a("|---|---|")
            for where, what in DEMO["on_screen"]:
                a(f"| {where} | {what} |")
            a("")

            a("\n**Click order — say it to yourself in this order**\n")
            for n, step in enumerate(DEMO["walkthrough"], 1):
                a(f"{n}. {step}")
            a("")
            a("\n> Column-by-column definitions (Score scales, Split, Cohort affinity, the "
              "asterisk) are deliberately **not** in this block — they are in §4 Q&A, so you "
              "only reach for them if asked.\n")
            a(f"\n**中文提示**：{DEMO['hint']}\n")
            a("\n---\n")

    # ---- run of show ----
    a("## 3. Run of show\n")
    a("### 3.1 Pre-flight (T-10 minutes)\n")
    a("| # | Do this | Why |")
    a("|---:|---|---|")
    a("| 1 | `streamlit run app/streamlit_app.py` — leave the browser **open** on "
      "Tab 1, a user already picked | No waiting, no typing on stage |")
    a("| 2 | Press **F11** for full screen; hide the Streamlit toolbar "
      "(⋮ → Settings → uncheck *Show app menu*) | Kills the toolbar clutter |")
    a("| 3 | Open `slides/deck.pptx` in **Presenter View**, plug in the projector | Notes "
      "appear on the laptop only |")
    a("| 4 | Check the three model columns are **all populated** "
      "(Item-CF / LightFM / LRURec) | A missing column reads as a broken demo |")
    a("| 5 | Have `PRESENTATION.md` open on the phone as a backup script | Screen dies, "
      "talk does not |")
    a("")
    a("### 3.2 The switch itself\n")
    a("```")
    a("at 1:53  → slide 4 is on screen, you have just said '2.9 times'")
    a("        → press B          (screen goes black — audience sees nothing)")
    a("        → Alt+Tab          (browser)")
    a("        → say the demo line, click 'Random user' once")
    a("        → 30 seconds: point at the three columns, one sentence")
    a("        → Alt+Tab back     (PowerPoint)")
    a("        → press B again    (slide 4 returns)")
    a("        → continue to slide 5")
    a("```")
    a("\n> **Why here, not at the end:** the numbers are on screen, so the demo makes them "
      "concrete; and you still finish on your conclusions rather than on a browser window.\n")
    a("### 3.3 If the demo fails\n")
    a("- Say *\"the demo is running locally — I will show it during questions\"* and go on. "
      "Do not debug on stage.")
    a("- Take the first Q&A question about results, then show the window. "
      "A live demo inside Q&A reads as generosity, not as failure.")
    a("- Worst case: the numbers on slide 4 already carry the argument; the demo is "
      "supporting evidence, not the argument itself.\n")

    # ---- Q&A ----
    a("## 4. Q&A preparation (5 minutes)\n")
    a("Answer in one breath: *concede what is true, then give the number.*\n")
    for q, ans in QA:
        a(f"**Q: {q}**\n")
        a(f"A: {ans}\n")

    a("\n---\n")
    a("## 5. Rehearsal checklist\n")
    a("- [ ] Read the speaker notes aloud once; mark words you stumble on and simplify them")
    a("- [ ] Time it with a stopwatch — target **4:45**, hard stop at 5:00")
    a("- [ ] Rehearse the **window switch** at least twice; it is the only choreography")
    a("- [ ] For each figure, say *what to look at* before *what it means* (point, then interpret)")
    a("- [ ] Decide who answers which Q&A area (results / method / models)")
    a("- [ ] Have the number sheet in hand: 0.0641 · 0.0775 · 0.2272 · baseline 0.0353\n")

    return "\n".join(out)


if __name__ == "__main__":
    path = ROOT / "PRESENTATION.md"
    path.write_text(render(), encoding="utf-8")
    print(f"wrote {path}  ({path.stat().st_size / 1024:.1f} KB)")
