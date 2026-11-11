# Slide brand assets

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
