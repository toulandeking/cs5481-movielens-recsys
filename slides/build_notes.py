"""
slides/build_notes.py  —  Render PRESENTATION.md from slides/content.py.
=======================================================================
No third-party dependency, so it can run before python-pptx is installed.

    python slides/build_notes.py
"""

from __future__ import annotations

from pathlib import Path

from content import QA, SLIDES, TOTAL_SECONDS

ROOT = Path(__file__).resolve().parents[1]


def _mmss(sec: int) -> str:
    return f"{sec // 60}:{sec % 60:02d}"


def render() -> str:
    total = sum(s["seconds"] for s in SLIDES)
    out: list[str] = []
    a = out.append

    a("# Presentation Plan — 5 minutes talk + 5 minutes Q&A\n")
    a("> **Slides: English. Speaker notes: English, with short Chinese cues.**\n")
    a("> Covers only the models, the results and our thoughts — no code, as instructed.\n")
    a("> Generated from `slides/content.py`. Edit that file, then re-run "
      "`python slides/build_notes.py`.\n")
    a(f"\n**Planned speaking time: {_mmss(total)} / {_mmss(TOTAL_SECONDS)}**\n")
    a("\n---\n")

    # ---- timing table ----
    a("## 1. Timing at a glance\n")
    a("| # | Slide | Time | Cumulative |")
    a("|---:|---|---:|---:|")
    cum = 0
    for i, s in enumerate(SLIDES, 1):
        cum += s["seconds"]
        a(f"| {i} | {s['title']} | {s['seconds']}s | {_mmss(cum)} |")
    a(f"| — | **Total** | **{total}s** | **{_mmss(cum)}** |\n")
    a("\n> Leave ~10s of slack. If you run long, cut Thought 3 (slide 7) — it is the "
      "least visual and can survive as a spoken sentence.\n")
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

    # ---- Q&A ----
    a("## 3. Q&A preparation (5 minutes)\n")
    a("Answer in one breath: *concede what is true, then give the number.*\n")
    for q, ans in QA:
        a(f"**Q: {q}**\n")
        a(f"A: {ans}\n")

    a("\n---\n")
    a("## 4. Rehearsal checklist\n")
    a("- [ ] Read the speaker notes aloud once; mark words you stumble on and simplify them")
    a("- [ ] Time it with a stopwatch — target **4:45**, hard stop at 5:00")
    a("- [ ] For each figure, say *what to look at* before *what it means* (point, then interpret)")
    a("- [ ] Prepare the Streamlit window in advance; do not debug the demo live")
    a("- [ ] Decide who answers which Q&A area (results / method / models)")
    a("- [ ] Have the number sheet in hand: 0.0641 · 0.0775 · 0.2272 · baseline 0.0353\n")

    return "\n".join(out)


if __name__ == "__main__":
    path = ROOT / "PRESENTATION.md"
    path.write_text(render(), encoding="utf-8")
    print(f"wrote {path}  ({path.stat().st_size / 1024:.1f} KB)")
