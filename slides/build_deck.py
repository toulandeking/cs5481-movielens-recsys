"""
slides/build_deck.py  —  Build slides/deck.pptx from slides/content.py.
=======================================================================

Needs python-pptx (not installed in this workspace):

    python -m pip install python-pptx
    python slides/build_deck.py

Output: slides/deck.pptx   (16:9, English, speaker notes embedded)
Speaker notes carry the English script plus a short Chinese cue line.

Design rules baked in
---------------------
* No code on any slide (instructor requirement).
* Every slide has exactly one idea stated in the title.
* Figures get a small source line underneath.
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from pptx import Presentation
    from pptx.util import Inches, Pt
    from pptx.dml.color import RGBColor
    from pptx.enum.text import PP_ALIGN
except ImportError:  # pragma: no cover
    sys.exit(
        "python-pptx is required.\n"
        "    python -m pip install python-pptx\n"
        "then re-run: python slides/build_deck.py"
    )

from content import SLIDES

HERE = Path(__file__).resolve().parent
FIGS = HERE / "figs"
ASSETS = HERE / "assets"
OUT = HERE / "deck.pptx"

BG_BODY = ASSETS / "bg_body.png"
BG_TITLE = ASSETS / "bg_title.png"
LOGO = ASSETS / "cityu_logo.png"        # drop the official file here; picked up automatically
CAMPUS = ASSETS / "cityu_campus.jpg"    # optional full-bleed title backdrop (supply 16:9)
FOOTER = "City University of Hong Kong (Dongguan)  ·  CS5481 Recommender Systems"
DEFAULT_CAPTION = ("Held-out test set · leave-one-out · K = 10 · "
                   "full ranking over 3,416 items")
NAVY = RGBColor(0x18, 0x27, 0x45)

# --- geometry (inches, 16:9) -------------------------------------------------
SW, SH = 13.333, 7.5
M = 0.62                      # outer margin
TEXT_W = SW - 2 * M

# --- palette -----------------------------------------------------------------
INK = RGBColor(0x1F, 0x27, 0x33)
MUTED = RGBColor(0x6B, 0x72, 0x80)
ACCENT = RGBColor(0x2F, 0x4F, 0x8F)
RULE = RGBColor(0xD8, 0xDD, 0xE4)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)


def _textbox(slide, left, top, width, height):
    tb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = 0
    tf.margin_top = tf.margin_bottom = 0
    return tf


def _set(p, text, size, *, bold=False, color=INK, space_before=0, space_after=6, font="Calibri"):
    p.text = text
    r = p.runs[0]
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.color.rgb = color
    r.font.name = font
    p.space_before = Pt(space_before)
    p.space_after = Pt(space_after)


def _rule(slide, top, left=M, width=TEXT_W, color=RULE):
    from pptx.enum.shapes import MSO_SHAPE
    bar = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(left), Inches(top), Inches(width), Pt(1.2)
    )
    bar.fill.solid()
    bar.fill.fore_color.rgb = color
    bar.line.fill.background()
    bar.shadow.inherit = False
    return bar


# ---------------------------------------------------------------------------
def add_slide(prs: Presentation, s: dict, index: int, total: int) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])   # blank
    dark = bool(s.get("dark"))

    # ---- background first, so it sits behind every other shape ----
    if dark and CAMPUS.is_file():
        # a real campus photo as the hero backdrop, with a solid panel behind the text
        from pptx.enum.shapes import MSO_SHAPE
        slide.shapes.add_picture(str(CAMPUS), 0, 0,
                                 width=prs.slide_width, height=prs.slide_height)
        panel = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0,
                                       int(prs.slide_width * 0.60), prs.slide_height)
        panel.fill.solid()
        panel.fill.fore_color.rgb = NAVY
        panel.line.fill.background()
        panel.shadow.inherit = False
    else:
        bg = BG_TITLE if dark else BG_BODY
        if bg.is_file():
            slide.shapes.add_picture(str(bg), 0, 0,
                                     width=prs.slide_width, height=prs.slide_height)

    # colour scheme depends on the background
    ink = WHITE if dark else INK
    muted = RGBColor(0xC2, 0xCE, 0xE4) if dark else MUTED
    accent = RGBColor(0x9F, 0xBC, 0xEC) if dark else ACCENT
    rule_color = RGBColor(0x55, 0x6B, 0x99) if dark else RULE

    # ---- official logo slot: appears automatically once the file exists ----
    right_pad = 0
    if LOGO.is_file():
        logo_h = 0.78 if dark else 0.58
        pic = slide.shapes.add_picture(str(LOGO), 0, 0, height=Inches(logo_h))
        pic.left = int(prs.slide_width - pic.width - Inches(M))
        pic.top = Inches(M - 0.14)
        right_pad = pic.width / 914400 + 0.25      # keep the title clear of the logo

    title_w = TEXT_W - right_pad

    # kicker
    if s.get("kicker"):
        tf = _textbox(slide, M, M - 0.06, title_w, 0.32)
        _set(tf.paragraphs[0], s["kicker"].upper(), 11.5, bold=True, color=accent)

    # title
    tf = _textbox(slide, M, M + 0.30, title_w, 0.95)
    _set(tf.paragraphs[0], s["title"], 30, bold=True, color=ink, space_after=0)

    _rule(slide, M + 1.26, width=title_w, color=rule_color)

    body_top = M + 1.46
    body_h = SH - body_top - 0.62
    has_fig = bool(s.get("figure"))

    # ---------------- figure ----------------
    fig_left, fig_w = M, 7.55
    if has_fig:
        path = FIGS / s["figure"]
        if path.is_file():
            pic = slide.shapes.add_picture(str(path), Inches(fig_left), Inches(body_top),
                                           width=Inches(fig_w))
            # keep aspect ratio already applied by width=; clamp height
            if pic.height > Inches(body_h):
                ratio = body_h / (pic.height / 914400)
                pic.width = int(pic.width * ratio)
                pic.height = int(pic.height * ratio)
            cap = _textbox(slide, fig_left, SH - 0.92, fig_w, 0.3)
            _set(cap.paragraphs[0], s.get("caption") or DEFAULT_CAPTION,
                 9.5, color=muted, space_after=0)
        else:
            tf = _textbox(slide, fig_left, body_top, fig_w, 0.6)
            _set(tf.paragraphs[0], f"[missing figure: {s['figure']}]", 12, color=MUTED)

    # text column geometry
    if has_fig:
        col_left, col_w = M + fig_w + 0.32, TEXT_W - fig_w - 0.32
    else:
        col_left, col_w = M, TEXT_W

    # ---------------- table ----------------
    if s.get("table"):
        t = s["table"]
        n_rows = len(t["rows"]) + 1
        n_cols = len(t["header"])
        widths = t.get("widths") or [col_w / n_cols] * n_cols
        total_w = sum(widths)
        widths = [w * col_w / total_w for w in widths]

        row_h = 0.42
        shape = slide.shapes.add_table(
            n_rows, n_cols, Inches(col_left), Inches(body_top + 0.06),
            Inches(col_w), Inches(row_h * n_rows),
        )
        table = shape.table
        for i, w in enumerate(widths):
            table.columns[i].width = Inches(w)

        for c, head in enumerate(t["header"]):
            cell = table.cell(0, c)
            cell.text = head
            para = cell.text_frame.paragraphs[0]
            _set(para, head, 12.5, bold=True, color=WHITE, space_after=0)
            cell.fill.solid()
            cell.fill.fore_color.rgb = ACCENT
            cell.margin_left = cell.margin_right = Inches(0.08)

        for r, row in enumerate(t["rows"], start=1):
            for c, val in enumerate(row):
                cell = table.cell(r, c)
                cell.text = val
                para = cell.text_frame.paragraphs[0]
                _set(para, val, 12.5, bold=(c == 0), space_after=0)
                cell.fill.solid()
                cell.fill.fore_color.rgb = WHITE if r % 2 else RGBColor(0xF5, 0xF7, 0xFA)
                cell.margin_left = cell.margin_right = Inches(0.08)

        if has_fig:
            note = _textbox(slide, col_left, body_top + 0.06 + row_h * n_rows + 0.14,
                            col_w, 0.9)
            _set(note.paragraphs[0],
                 "Gains are consistent in NDCG, so the ranking is not an artefact of one metric.",
                 10.5, color=MUTED, space_after=0)
        body_top_text = body_top + 0.06 + row_h * n_rows + (1.05 if has_fig else 0.2)

    else:
        body_top_text = body_top

    # ---------------- bullets ----------------
    bullets = [b for b in (s.get("bullets") or []) if b[0].strip()]
    if bullets:
        tf = _textbox(slide, col_left, body_top_text, col_w, max(1.0, SH - body_top_text - 1.0))
        first = True
        for text, level in bullets:
            p = tf.paragraphs[0] if first else tf.add_paragraph()
            first = False
            if level == 0:
                _set(p, "•  " + text, 15.5, color=ink, space_before=7, space_after=5)
            else:
                _set(p, "      –  " + text, 13, color=muted, space_before=0, space_after=4)

    # ---------------- footer + page number ----------------
    ff = _textbox(slide, M + 0.22, SH - 0.44, 6.6, 0.26)
    _set(ff.paragraphs[0], FOOTER, 8.5, color=muted, space_after=0)

    tf = _textbox(slide, SW - M - 0.9, SH - 0.44, 0.9, 0.3)
    _set(tf.paragraphs[0], f"{index} / {total}", 10, color=muted, space_after=0)
    tf.paragraphs[0].alignment = PP_ALIGN.RIGHT

    # ---------------- speaker notes ----------------
    notes = slide.notes_slide.notes_text_frame
    notes.text = s["notes"]
    p = notes.add_paragraph()
    _set(p, "【中文提示】" + s["hints"], 11, color=INK)


def build() -> None:
    prs = Presentation()
    prs.slide_width = Inches(SW)
    prs.slide_height = Inches(SH)
    total = len(SLIDES)
    for i, s in enumerate(SLIDES, 1):
        add_slide(prs, s, i, total)
    prs.save(OUT)
    print(f"wrote {OUT}  ({OUT.stat().st_size / 1024:.1f} KB, {total} slides)")


if __name__ == "__main__":
    build()
