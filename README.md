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
  --character "1girl" \
  --hint "face_paint" \
  --blacklist "*armor*, *scale*, *chain*, *dress*, *gown*, *trim*, barefoot" \
  --save-config

# every run after that:
tagger . --dry-run --force --limit 5   # preview (--force needed if all captioned)
# or, when ready:
tagger . --force --batch-size 4
```

Generated `tagger.toml` (edit by hand anytime):

```toml
subject = "outfit"
trigger = "bg3_wavemother_robe"
character = ["1girl"]                  # invariant ONLY — no outfit words (see schools below)
hint = ["face_paint"]                 # identity that varies per image; survives *paint*
blacklist = ["*armor*", "*scale*", "*chain*", "*dress*", "*gown*", "*trim*",
             "*glove*", "*boot*", "*slit*", "*sword*", "barefoot"]
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

## Two schools: tag the subject, or let the trigger own it

**School 1 — tag everything visible** (decomposable): the subject is described
in tags (`blue_armor, scale_mail, gold_trim`). You keep prompt-level control —
recolor, swap pieces, address variants per color. Cost: **leakage** — the model
learns it can render the subject from the tags alone, so the trigger word fires
weakly by itself.

**School 2 — the trigger owns the subject** (fidelity): blacklist the whole
subject vocabulary (`*armor*`, `*scale*`, `*chain*`, ...) and leave the subject
untagged. Every feature binds 100% to the trigger; captions shrink to
`trigger, 1girl, [identity], [context]`. Max trigger fidelity, zero color/
material hedging — but no prompt-level control, and variants blend into a
family average.

> Empirical: on a 25-image outfit set, School 1 produced a **generic outfit**
> when the trigger was prompted alone — the robe only fired when a color tag
> was also prompted. Flipping to School 2 fixed it.

Rule of thumb: small datasets and identity-first goals → **School 2**.
Control-first (recolorable, addressable variants) → **School 1**, and then the
per-image tags must be human-verified (see `--audit`).

**Universal recipe:** pick a school first. School 2: blacklist = the whole
subject family (`*scale*`, `*chain*`), hint = identity that varies per image and
must stay taggable (`face_paint` survives `*paint*` via hint-precedence).
School 1: hint = the family's canonical vocabulary, blacklist = whole-family
wildcards, header holds only what's true on **every** image. Hint entries
always survive the blacklist in either school.

## Quick start

```bash
# from any directory (D:/Scripts must be on PATH)
tagger D:\dataset                       # tag all images, skip non-empty captions
tagger . --init-config                  # commented starter tagger.toml, then edit by hand
tagger . --dry-run --force --limit 5    # preview 5 images, write nothing
tagger . --subject outfit --trigger myoutfit
tagger . --character "1girl, elf, silver_hair" --trigger mychar
tagger . --subject style --trigger mystyle
tagger . --hint "face_paint" --blacklist "fantasy_*, ornate"
tagger . --audit                        # frequency report over existing captions (no API)
tagger . --review                       # human review grid in the browser (no API)
```

## Review grid (`--review`)

The human pass, upgraded from text to pictures. Starts a local web server and opens
the dataset in the browser — no LM Studio needed:

```bash
tagger . --review            # opens http://127.0.0.1:8765
```

- **Grid of thumbnails + captions**, click any caption to edit inline
  (Ctrl+Enter saves, Esc cancels; empty field clears the caption so it gets
  re-tagged on the next run).
- **Singleton tags highlighted in red** — tags appearing on only 1 image
  (hallucination / rare-variant suspects). Click a chip in the header to
  isolate just the images carrying that tag.
- **Badges** for empty / missing captions; orphan `.txt` files listed in the
  header stats.
- **Saves write straight back to `.txt`** — exactly what the trainer sees,
  and the resume logic never touches your edits.
- Local-only (`127.0.0.1`), path-traversal safe, no upload, nothing leaves
  your machine. `--port` to change the port, `--no-browser` to skip auto-open.

`--audit` remains for headless checks; `--review` is the interactive version.

## Setup wizard (`--setup`)

The config black-art, retired. Instead of hand-writing `tagger.toml` (or
memorising the long CLI command), answer eight plain-word questions in the
browser and it writes the config for you:

```bash
tagger . --setup
```

- **Plain words in, wildcards out** — type *"armor, dress, gown, scale, chain"*
  and it generates `*armor*, *dress*, ...` with a live preview showing exactly
  which of your current caption tags each word blocks.
- **School picker** — the two schools explained side by side; flipping flips
  your lists too (subject details move hint ↔ blocklist, with a note saying
  what moved).
- **Re-opening an existing config** — it infers the school from your blocklist,
  recovers the subject words, separates out meta junk, and never loses your
  tuning values (temperature, workers, ...).
- **Trigger validation** — lowercase snake_case enforced before save.

`--init-config` remains for headless scaffold; `--setup` is the interactive way.

## The web UI (tagdeck)

`--review` and `--setup` are two pages of a local web app — **tagdeck** — that
now covers the whole 0 → training-ready pipeline. All pages share one server
and one port; the header nav moves between them. No LM Studio needed for any
of them:

```bash
tagger . --review     # 1. thumbnails + editable captions, singleton flags
```

```bash
tagger . --setup      # 2. plain-word questions -> tagger.toml
```

```bash
tagger . --ui validate   # 3. dataset health checks
```

```bash
tagger . --ui split      # 4. train/val split (copy or manifest)
```

```bash
tagger . --ui export     # 5. OneTrainer config, patched from your last config
```

- **validate** — pairing, corrupt images, pixel duplicates, trigger coverage,
  blacklist leakage (respecting hint-precedence), duplicate tags in one caption.
- **split** — seeded random split, click any thumbnail to move it between
  train/val, then apply: copies into `train/` + `val/` (originals untouched,
  refuses to clobber an existing split) or writes `split.json` only.
- **export** — builds `onetrainer_config.json` in the dataset folder. Uses your
  most recent OneTrainer config (auto-detected from `OneTrainer_Workspace`) as
  the template so your tuned pipeline propagates, and patches only the
  dataset-specific fields (concept path, trigger, output name, epochs/lr/rank).

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
| `--batch-size` | `1` | N images per API call → consistent vocabulary across similar images |
| `--limit` | all | process at most N images |
| `--force` | off | re-tag non-empty captions; **asks for confirmation** before overwriting |
| `--dry-run` | off | print captions, write nothing (no confirmation needed) |
| `--audit` | off | print tag-frequency report over existing captions (no API) |
| `--review` | off | human review grid in the browser (no API) |
| `--setup` | off | config wizard in the browser — plain-word questions that generate `tagger.toml` (no API) |
| `--ui NAME` | — | open a tagdeck page: `review` \| `setup` \| `validate` \| `split` \| `export` |
| `--port` | `8765` | port for `--review` |
| `--no-browser` | off | with `--review`: don't auto-open the browser |
| `--report` | off | print tag-frequency report after tagging |
| `--save-config` | off | write `tagger.toml` into the dataset folder and exit |
| `--init-config` | off | write a commented starter `tagger.toml` (refuses to overwrite) and exit |
| `--config` | auto | explicit config path instead of auto-loading `tagger.toml` |

## Troubleshooting

- `error: cannot reach LM Studio` — start LM Studio, load a model, check the
  server port in LM Studio's Developer tab.
- Model still hallucinates cropped features — make sure you're on the latest
  version; the visibility rule is enforced by the system prompt.
- Generic "no character" tags on real images — check that the images are the
  actual focus of each crop.
- Slow tagging — Qwen3-VL-8B on CPU is slow; keep `--workers 1`, consider a
  smaller VL model for large datasets.
