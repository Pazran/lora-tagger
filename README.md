# lora-tagger

Auto-tag dataset images for LoRA training using a local LM Studio vision model
(Qwen3-VL by default). Writes Danbooru-style `<stem>.txt` captions next to each
image — the standard kohya/ai-toolkit dataset layout.

Designed to be **dataset-agnostic**: works for character, outfit, or style LoRAs,
on any image composition (full body, half body, head-cropped, mixed). No code
changes per dataset — you steer with flags.

## Prerequisites

- LM Studio running with a vision model loaded (`qwen3-vl-8b-instruct` by
  default, auto-detected). API at `http://localhost:1234/v1` by default.
- Python deps installed (already done in `venv/`): `requests`, `Pillow`.

## Config file (the long command, once)

Dataset settings live in a `tagger.toml` **inside the dataset folder** — the folder
carries its own config. Save it once, then plain `tagger .` just works:

```bash
# one time, in the dataset folder:
tagger . --subject outfit --trigger bg3_wavemother_robe \
  --character "1girl, scale_armor, armored_dress" \
  --hint "face_paint" \
  --blacklist "fantasy_*, scale_mail, chain_mail" \
  --save-config

# every run after that:
tagger . --dry-run --limit 5      # preview
# or, when ready:
tagger .
```

Generated `tagger.toml` (edit by hand anytime):

```toml
subject = "outfit"
trigger = "bg3_wavemother_robe"
character = ["1girl", "scale_armor", "armored_dress"]
hint = ["face_paint"]
blacklist = ["fantasy_*", "scale_mail", "chain_mail"]
temperature = 0.3
max_tags = 40
max_size = 1280
workers = 1
base_url = "http://localhost:1234/v1"
```

**Blacklist pattern syntax:** `foo` exact, `foo*` prefix, `*foo` suffix, `*foo*`
any tag containing `foo` — the containing form kills whole variant families
(`*scale*` catches `scale_mail`, `blue_scale_armor`, `silver_scale_mail`, ...).
**Hint precedence:** hint tags always survive the blacklist, so `--hint "face_paint"`
works even with `*paint*` blacklisted.

**Precedence:** CLI flags override config scalars (`--subject`, `--trigger`,
`--character`); `--hint` and `--blacklist` **merge** with config entries. Use
`--config path/to/file.toml` for an explicit config elsewhere.

**Cross-image consistency:** add `--batch-size N` to tag N images in one API call
so the model aligns vocabulary across similar images (same feature = same tag name).
An anti-copy rule keeps per-image differences intact. Missing images fall back to
single calls automatically. Example: `tagger . --force --batch-size 4`.

**Universal recipe (variation datasets):** hint = the *family's canonical
vocabulary*, blacklist = *whole-family wildcards*. Hint entries always survive the
blacklist, so `hint = ["scale_mail", "chain_mail"]` + `"*scale*", "*chain*"` in
the blacklist means variant spellings you blessed survive while invented compounds
(`silver_scale_mail`) die — the model still picks which variant each image shows.
Header holds only what's true on **every** image; everything else goes to hint.

## Quick start

```bash
# from any directory (D:/Scripts must be on PATH)
tagger D:\dataset                       # tag all images, skip non-empty captions
tagger . --dry-run --limit 5            # preview 5 images, write nothing
tagger . --subject outfit --trigger myoutfit
tagger . --character "1girl, elf, silver_hair" --trigger mychar
tagger . --subject style --trigger mystyle
tagger . --hint "face_paint" --blacklist "fantasy_*, ornate"
```

## Full pipeline (compose with renamer)

```bash
renamer bg3 .jpg --txt          # normalize names + create empty caption stubs
tagger . --trigger mychar       # fill the empty stubs with Danbooru tags
```

Empty `.txt` stubs are filled; non-empty captions are never overwritten, so
interrupted runs resume cleanly and hand-written captions survive.

## The three steering mechanisms

1. **`--character` = the invariant set.** Tags that are true for EVERY image in
   the folder (`1girl, white_hair`). Prepended to every caption and excluded
   from model output. Keep it minimal — if a feature varies across images
   (face paint, poses), it does NOT belong here.
2. **`--hint` = canonical vocabulary.** "When this feature is visible, use
   exactly these tags, never synonyms" (e.g. `face_paint` instead of the
   model's random `face_tattoo / blue_makeup / glowing_blue_tattoo`). For
   variable features the model must decide per-image.
3. **`--blacklist` = steering.** Exact tags or `foo_*` wildcard families
   (`fantasy_*` kills the whole spam family). Merged with defaults.

Built-in prompt rules apply to every dataset:
- **Only tag what's visible** — head-cropped images never get hallucinated
  hair/eye tags, no matter the folder composition.
- No invented compound tags (`fantasy_woman`, `intricate_design`).
- No rating, quality, or resolution/meta tags.
- `no_*` junk tags (`no_face`, `no_hair`) stripped by default.

## Subjects

| Subject | Focus |
|---------|-------|
| `character` | full design: hair, eyes, expression, pose, clothing |
| `outfit` | garment detail: types, materials, patterns, layering; identity secondary |
| `style` | medium, rendering, palette, lighting, composition; no subject identity |

## Options

| Flag | Default | Purpose |
|------|---------|---------|
| `--subject` | `character` | `character`, `outfit`, or `style` |
| `--trigger` | — | token prepended to every caption |
| `--character` | — | invariant header tags (see above) |
| `--hint` | — | canonical vocabulary for variable features |
| `--blacklist` | built-in | extra tags / `foo_*` families to strip |
| `--base-url` | `http://localhost:1234/v1` | LM Studio API |
| `--model` | auto | override model id |
| `--temperature` | `0.3` | low = reproducible tags |
| `--max-tags` | `40` | cap on tags per caption |
| `--max-size` | `1280` | downscale longest side before sending |
| `--workers` | `1` | parallel requests (LM Studio queues; keep low) |
| `--limit` | all | process at most N images |
| `--force` | off | re-tag non-empty captions |
| `--dry-run` | off | print captions, write nothing |

## Troubleshooting

- `error: cannot reach LM Studio` — start LM Studio, load a model, check the
  server port in LM Studio's Developer tab.
- Model still hallucinates cropped features — make sure you're on the latest
  version; the visibility rule is enforced by the system prompt.
- Generic "no character" tags on real images — check that the images are the
  actual focus of each crop.
- Slow tagging — Qwen3-VL-8B on CPU is slow; keep `--workers 1`, consider a
  smaller VL model for large datasets.
