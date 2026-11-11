"""
slides/build_assets.py  —  Generate slide backgrounds and the brand asset slots.
================================================================================

Why generated rather than downloaded
------------------------------------
The official CityU (Dongguan) logo cannot be fetched from this environment, and
inventing a crest or guessing an official brand colour would be wrong. So:

  * backgrounds are generated locally (geometry only, no fake logo);
  * a real logo file is picked up automatically if you put it in slides/assets/;
  * the file names and where to obtain them are documented in assets/README.md.

Outputs (1920x1080, 16:9 to match the deck)
    slides/assets/bg_body.png     light background for content slides
    slides/assets/bg_title.png    dark hero background for the title slide
    slides/assets/README.md       what to drop in, and where to download it

Usage
    python slides/build_assets.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
OUT = HERE / "assets"
OUT.mkdir(parents=True, exist_ok=True)

W, H = 1920, 1080

# Deck accent (kept in sync with build_deck.py)
ACCENT = (0x2F, 0x4F, 0x8F)
INK = (0x1F, 0x27, 0x33)
NAVY_TOP = (0x14, 0x22, 0x3D)
NAVY_BOT = (0x25, 0x3C, 0x69)


def _vgradient(size: tuple[int, int], top: tuple[int, int, int],
               bottom: tuple[int, int, int]) -> Image.Image:
    img = Image.new("RGB", size, top)
    d = ImageDraw.Draw(img)
    w, h = size
    for y in range(h):
        t = y / (h - 1)
        c = tuple(int(top[i] + (bottom[i] - top[i]) * t) for i in range(3))
        d.line([(0, y), (w, y)], fill=c)
    return img


def _arcs(size: tuple[int, int], colour: tuple[int, int, int], alpha: int,
          centre: tuple[int, int], radii: list[int], width: int = 2) -> Image.Image:
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    cx, cy = centre
    for r in radii:
        box = [cx - r, cy - r, cx + r, cy + r]
        d.arc(box, start=156, end=340, fill=colour + (alpha,), width=width)
    return layer


def build_body() -> None:
    img = _vgradient((W, H), (0xFF, 0xFF, 0xFF), (0xF4, 0xF6, 0xFA))

    # faint concentric arcs, bottom-right, well away from the text column
    img = Image.alpha_composite(
        img.convert("RGBA"),
        _arcs((W, H), ACCENT, 16, (W - 60, H - 40), [340, 470, 610, 760], width=2),
    ).convert("RGB")

    d = ImageDraw.Draw(img)
    # left accent bar
    d.rectangle([0, 0, 13, H], fill=ACCENT)
    # hairline above the footer
    d.line([(150, H - 74), (W - 60, H - 74)], fill=(0xE3, 0xE7, 0xEE), width=2)

    img.save(OUT / "bg_body.png", optimize=True)


def build_title() -> None:
    img = _vgradient((W, H), NAVY_TOP, NAVY_BOT)

    # light sweeping arcs, right side
    img = Image.alpha_composite(
        img.convert("RGBA"),
        _arcs((W, H), (0xFF, 0xFF, 0xFF), 22, (W - 120, H + 60), [420, 600, 790, 990], width=3),
    ).convert("RGB")

    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    # soft horizontal scrim: darker on the left so white text always reads,
    # fading to fully transparent before the arcs — no hard vertical seam
    x_end = 1380
    for x in range(x_end):
        a = int(86 * (1 - x / x_end) ** 1.6)
        if a:
            d.line([(x, 0), (x, H)], fill=(5, 10, 24, a))
    # a light accent rule bottom-left, readable on the dark ground
    light_accent = (0x7D, 0x9B, 0xD6)
    d.rectangle([150, H - 152, 150 + 168, H - 147], fill=light_accent + (255,))
    img = Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB")

    img.save(OUT / "bg_title.png", optimize=True)


def write_readme() -> None:
    (OUT / "README.md").write_text(
        """# Slide brand assets

## What is generated here (do not hand-edit)

| File | Used by | Purpose |
|---|---|---|
| `bg_body.png` | content slides | light background, left accent bar, faint arcs |
| `bg_title.png` | slide 1 | dark hero background |

Regenerate with `python slides/build_assets.py`. They are plain images, so you can
also replace them with your own 1920x1080 design and keep the same file names.

## What you should add (official material)

Put the **official** logo here and the deck picks it up automatically on the next
`python slides/build_deck.py` — no code change:

```
slides/assets/cityu_logo.png     transparent background, ideally >= 600 px wide
slides/assets/cityu_campus.jpg   optional; full-bleed title backdrop.
                                 Supply a 16:9 image (e.g. 1920x1080) — a different
                                 aspect ratio will be stretched by the deck.
```

### Where to get the official files

Use the university's own channels rather than a third-party mirror:

- CityU (Dongguan) official site: <https://www.cityu-dg.edu.cn/>
  (look for 视觉标识 / 校徽 / brand or press resources)
- CityU Hong Kong corporate identity guidelines:
  <https://www.cityu.edu.hk/>
- Wikipedia has a copy of the emblem for reference:
  <https://zh.wikipedia.org/wiki/香港城市大學（東莞）>

**Do not** recolour, distort, or crop the logos, and keep clear space around them.
If the university publishes a brand guideline, follow its rules on minimum size
and background contrast.

## Notes on what was deliberately *not* done

No crest or wordmark was drawn programmatically. A hand-drawn approximation of an
institutional logo would misrepresent the university, and guessing the official
brand colour would be worse than using a neutral one. Until a real file is present,
the deck carries institutional branding through the text footer only
("City University of Hong Kong (Dongguan) · CS5481 Recommender Systems").
""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    build_body()
    build_title()
    write_readme()
    for p in sorted(OUT.iterdir()):
        print(f"  {p.stat().st_size / 1024:8.1f} KB  {p.relative_to(HERE.parent)}")
