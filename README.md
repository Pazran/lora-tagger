# lora-tagger

Auto-tag dataset images for LoRA training using a local LM Studio vision model
(Qwen3-VL by default). Writes Danbooru-style `<stem>.txt` captions next to each
image — the standard kohya/ai-toolkit dataset layout.

## Prerequisites

- LM Studio running with a vision model loaded (`qwen3-vl-8b-instruct` by
  default, auto-detected). API at `http://localhost:1234/v1` by default.
- Python deps installed (already done in `venv/`): `requests`, `Pillow`.

## Quick start

```bash
# from any directory (D:/Scripts must be on PATH)
tagger D:\dataset                       # tag all images, skip non-empty captions
tagger . --dry-run --limit 5            # preview 5 images, write nothing
tagger . --subject outfit --trigger myoutfit
tagger . --character "1girl, elf, silver_hair" --trigger mychar
```

## Full pipeline (compose with renamer)

```bash
renamer bg3 .jpg --txt          # normalize names + create empty caption stubs
tagger . --trigger mychar       # fill the empty stubs with Danbooru tags
```

Empty `.txt` stubs are filled; non-empty captions are never overwritten, so
interrupted runs resume cleanly and hand-written captions survive.

## Options

| Flag | Default | Purpose |
|------|---------|---------|
| `--subject` | `character` | `character` or `outfit` — changes prompt focus |
| `--trigger` | — | token prepended to every caption |
| `--character` | — | fixed header tags, also excluded from model output |
| `--blacklist` | built-in | extra tags to strip (merged with defaults) |
| `--base-url` | `http://localhost:1234/v1` | LM Studio API |
| `--model` | auto | override model id |
| `--temperature` | `0.3` | low = reproducible tags |
| `--max-tags` | `40` | cap on tags per caption |
| `--max-size` | `1280` | downscale longest side before sending |
| `--workers` | `1` | parallel requests (LM Studio queues; keep low) |
| `--limit` | all | process at most N images |
| `--force` | off | re-tag non-empty captions |
| `--dry-run` | off | print captions, write nothing |

Default blacklist strips rating (`safe`, `explicit`, ...), quality
(`masterpiece`, ...), and resolution/meta noise (`highres`, `absurdres`, ...).

## Troubleshooting

- `error: cannot reach LM Studio` — start LM Studio, load a model, check the
  server port in LM Studio's Developer tab.
- Model returns generic "no character" tags — your test images aren't anime /
  aren't focused; real cropped dataset images will behave differently. Use
  `--dry-run --limit 5` on a few real images first.
- Slow tagging — Qwen3-VL-8B on CPU is slow; keep `--workers 1`, consider
  loading a smaller VL model for large datasets.
