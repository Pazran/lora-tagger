#!/usr/bin/env python3
"""lora-tagger — auto-tag dataset images for LoRA training via LM Studio.

Sends each image in a folder to a local vision model (Qwen3-VL by default),
normalizes the response into Danbooru tags, and writes <stem>.txt captions
next to each image — the standard kohya/ai-toolkit dataset layout.
"""

import argparse
import base64
import io
import json
import os
import re
import sys
import time
import urllib.parse
import webbrowser
from concurrent.futures import ThreadPoolExecutor, as_completed
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import requests
import tomllib
from PIL import Image

from logger import setup_logger

logger = setup_logger("lora-tagger", level="INFO", log_file="tagger.log")

DEFAULT_BASE_URL = "http://localhost:1234/v1"

CYAN = "\033[36m"
RED = "\033[31m"
YELLOW = "\033[33m"
DIM = "\033[2m"
RESET = "\033[0m"


def color(text: str, code: str) -> str:
    """ANSI colorize when stdout is a console; plain text when piped to a file."""
    if sys.stdout.isatty():
        return f"{code}{text}{RESET}"
    return text
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# User policy: no rating tags, no quality tags, no resolution/meta noise.
DEFAULT_BLACKLIST = {
    # rating
    "safe", "sensitive", "questionable", "explicit",
    "rating:safe", "rating:sensitive", "rating:questionable", "rating:explicit",
    # quality
    "masterpiece", "best_quality", "worst_quality", "low_quality", "normal_quality",
    "high_quality", "good_quality", "bad_quality", "average_quality", "mediocre_quality",
    "amateur", "huge_filesize",
    # resolution / meta / noise
    "absurdres", "highres", "lowres", "newest", "comment", "commentary", "text",
    "watermark", "signature", "scan", "censored", "no_humans", "no humans",
    "none",  # Qwen emits a literal 'none' tag when it has nothing else to say
}

# VL models emit junk negative tags (no_face, no_hair, ...) on crops/absent features.
DEFAULT_BLACKLIST_PREFIXES = ["no_"]

SUBJECT_INSTRUCTIONS = {
    "character": (
        "The character is the main subject. Describe their design thoroughly: "
        "hair color and style, eye color, expression, body, pose, and their clothing."
    ),
    "outfit": (
        "The outfit is the main subject. Describe it in precise danbooru clothing tags: "
        "garment types, materials, patterns, colors, layering, accessories, footwear. "
        "Keep character identity tags (face, hairstyle, eye color) brief and secondary."
    ),
    "style": (
        "The artistic style is the main subject. Describe the medium, rendering "
        "technique, brushwork/lineart, color palette, lighting, and composition. "
        "Avoid identifying specific characters, outfits, or scenes."
    ),
}


def natural_key(s: str):
    """Sort key so bg3_2 comes before bg3_10 (digit-aware ordering)."""
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", s)]


def parse_tag_list(s: str) -> list[str]:
    if not s:
        return []
    return [t.strip().lower().replace(" ", "_") for t in s.split(",") if t.strip()]


def split_blacklist(extra: str) -> tuple[set[str], list[str], list[str], list[str]]:
    """Split a tag list into (exact, prefixes, suffixes, contains) patterns.

    'foo'   -> exact match
    'foo*'  -> any tag starting with 'foo'
    '*foo'  -> any tag ending with 'foo'
    '*foo*' -> any tag containing 'foo'  (kills whole variant families)
    Merged with the built-in defaults.
    """
    exact = set(DEFAULT_BLACKLIST)
    prefixes = list(DEFAULT_BLACKLIST_PREFIXES)
    suffixes, contains = [], []
    for t in parse_tag_list(extra):
        if t.startswith("*") and t.endswith("*"):
            core = t[1:-1]
            if core:
                contains.append(core)
        elif t.startswith("*"):
            suffixes.append(t[1:])
        elif t.endswith("*"):
            prefixes.append(t[:-1])
        else:
            exact.add(t)
    return exact, prefixes, suffixes, contains


def is_blacklisted(tag: str, exact: set[str], prefixes: list[str],
                   suffixes: list[str], contains: list[str]) -> bool:
    if tag in exact:
        return True
    if any(tag.startswith(p) for p in prefixes):
        return True
    if any(tag.endswith(s) for s in suffixes):
        return True
    return any(c in tag for c in contains)


def render_banned(exact: set[str], prefixes: list[str], suffixes: list[str],
                  contains: list[str]) -> str:
    parts = (sorted(exact) + [p + "*" for p in prefixes]
             + ["*" + s for s in suffixes] + ["*" + c + "*" for c in contains])
    return ", ".join(parts) or "(none)"


def parse_args(argv):
    p = argparse.ArgumentParser(
        prog="tagger",
        description="Tag every image in a folder with Danbooru tags via a local "
                    "LM Studio vision model, writing <stem>.txt captions.",
        epilog="Examples:\n"
               "  tagger D:\\dataset                     # tag everything (skip non-empty captions)\n"
               "  tagger . --dry-run --limit 5          # preview 5 images, write nothing\n"
               "  tagger . --subject outfit --trigger myoutfit\n"
               "  tagger . --character \"1girl, elf, silver_hair\" --trigger mychar\n"
               "  tagger . --subject style --trigger mystyle\n"
               "  tagger . --hint \"face_paint\" --blacklist \"fantasy_*, ornate\"\n"
               "  tagger . --save-config                   # persist settings to tagger.toml\n"
               "  tagger . --init-config                   # write a commented starter tagger.toml\n"
               "  tagger . --review                        # human review grid in the browser\n"
               "  tagger . --setup                         # config wizard in the browser",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("folder", nargs="?", default=".",
                   help="folder to scan (default: current folder)")
    p.add_argument("--config", default=None,
                   help="path to a tagger.toml config file; if omitted, tagger.toml "
                        "in the target folder is auto-loaded")
    p.add_argument("--save-config", action="store_true",
                   help="write effective settings to tagger.toml in the folder and exit")
    p.add_argument("--init-config", action="store_true",
                   help="write a commented starter tagger.toml into the folder (refuses to "
                        "overwrite an existing one) and exit")
    p.add_argument("--subject", choices=("character", "outfit", "style"), default=None,
                   help="what the model should focus on (default: character)")
    p.add_argument("--trigger", default=None,
                   help="token prepended to every caption, e.g. mychar")
    p.add_argument("--character", default=None,
                   help="invariant header tags true for EVERY image (e.g. \"1girl, white_hair\"); "
                        "prepended to all captions and excluded from model output; "
                        "overrides config")
    p.add_argument("--hint", default=None,
                   help="canonical vocabulary: tags to use exactly when the feature is visible, "
                        "e.g. \"face_paint\" (never synonyms); merges with config; "
                        "hint tags always survive the blacklist")
    p.add_argument("--blacklist", default=None,
                   help="extra tags to strip: 'foo' exact, 'foo*' prefix, '*foo*' any "
                        "containing foo (merges with config and defaults)")
    p.add_argument("--base-url", default=None,
                   help=f"LM Studio API base URL (default: {DEFAULT_BASE_URL})")
    p.add_argument("--model", default=None,
                   help="model id to use; auto-detects a qwen*-vl model if empty")
    p.add_argument("--temperature", type=float, default=None,
                   help="sampling temperature, low = reproducible (default: 0.3)")
    p.add_argument("--max-tags", type=int, default=None,
                   help="cap on tags per caption (default: 40)")
    p.add_argument("--max-size", type=int, default=None,
                   help="downscale longest side before sending (default: 1280px)")
    p.add_argument("--workers", type=int, default=None,
                   help="parallel requests; LM Studio queues, keep low (default: 1)")
    p.add_argument("--batch-size", type=int, default=1,
                   help="tag N images in one call so the model keeps vocabulary consistent "
                        "across similar images (default: 1 = per-image)")
    p.add_argument("--limit", type=int, default=0,
                   help="process at most N images (default: all)")
    p.add_argument("--force", action="store_true",
                   help="re-tag even if the .txt caption is already non-empty")
    p.add_argument("--dry-run", action="store_true",
                   help="tag and print captions, write nothing")
    p.add_argument("--report", action="store_true",
                   help="print tag frequency report after the run "
                        "(auto-enabled in --dry-run; use for real runs to audit color/tag spread)")
    p.add_argument("--audit", action="store_true",
                   help="no tagging: print the tag frequency report over EXISTING .txt captions "
                        "(the human review pass; works without LM Studio)")
    p.add_argument("--review", action="store_true",
                   help="open the human review grid in the browser: thumbnails + click-to-edit "
                        "captions, singleton/empty/missing flags; saves straight back to .txt "
                        "(no LM Studio needed)")
    p.add_argument("--setup", action="store_true",
                   help="open the config wizard in the browser: plain-word questions that "
                        "generate tagger.toml (no LM Studio needed)")
    p.add_argument("--port", type=int, default=8765,
                   help="port for the --review web UI (default: 8765)")
    p.add_argument("--no-browser", action="store_true",
                   help="with --review: start the server without opening a browser")
    return p.parse_args(argv)


def pick_model(models: list[dict], preferred: str) -> str:
    if preferred:
        return preferred
    ids = [m["id"] for m in models]
    for key in ("qwen3-vl", "qwen2.5-vl", "qwen2-vl", "qwen-vl", "vl"):
        for i in ids:
            if key in i.lower():
                return i
    return ids[0]


def system_prompt(subject: str, banned_txt: str, hint_txt: str, max_tags: int) -> str:
    return (
        "You are a precise Danbooru tagger for anime-style illustrations.\n"
        "You receive one image and must output ONLY a flat comma-separated list of Danbooru tags.\n"
        "Rules:\n"
        "- Output tags only: no sentences, no explanations, no numbering, no markdown, no code fences.\n"
        "- Danbooru conventions: lowercase, underscores instead of spaces (long_hair), singular nouns.\n"
        "- Describe ONLY what is visible in the image. Never invent tags for features that are "
        "cropped out, occluded, or absent (e.g. if the head is out of frame, do not output hair "
        "or eye color tags).\n"
        "- Use only existing Danbooru tags; never invent compound tags (e.g. fantasy_woman, "
        "intricate_design).\n"
        "- Pick exactly ONE color per object. Never output two color tags for the same item "
        "(e.g. do not output both blue_armor and teal_armor for the same garment). If colors "
        "are ambiguous, choose the single color that covers the largest visible area.\n"
        "- Be consistent across images: if the same object, material, or feature appears in "
        "multiple images, use the exact same tag name every time. Never vary the tag for the "
        "same thing (e.g. do not use sword_back on one image and sword_backpack on another).\n"
        f"- {SUBJECT_INSTRUCTIONS[subject]}\n"
        "- Never include quality tags (masterpiece, best_quality, ...), rating tags "
        "(safe, explicit, ...), or resolution/meta tags (highres, absurdres, newest, "
        "commentary, text, watermark).\n"
        f"- Never include these tags even if present in the image: {banned_txt}\n"
        f"- If any of these features is visible, use exactly these tags, never synonyms: {hint_txt}\n"
        f"- Output between 10 and {max_tags} tags."
    )


def image_to_b64(path: str, max_size: int) -> str:
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((max_size, max_size), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode()


def tag_image(args, model: str, b64: str, exact: set[str], prefixes: list[str],
              suffixes: list[str], contains: list[str], hint_set: set[str]) -> str:
    payload = {
        "model": model,
        "temperature": args.temperature,
        "max_tokens": 1024,
        "messages": [
            {"role": "system",
             "content": system_prompt(args.subject, render_banned(exact, prefixes,
                                                                  suffixes, contains),
                                      ", ".join(sorted(hint_set)) or "(none)",
                                      args.max_tags)},
            {"role": "user", "content": [
                {"type": "text",
                 "text": "Describe this image. Output only the comma-separated Danbooru tags."},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ]},
        ],
    }
    r = requests.post(f"{args.base_url.rstrip('/')}/chat/completions",
                      json=payload, timeout=300)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def call_with_retry(args, model: str, b64: str, exact: set[str], prefixes: list[str],
                    suffixes: list[str], contains: list[str], hint_set: set[str],
                    attempts: int = 3) -> str:
    last = None
    for i in range(attempts):
        try:
            return tag_image(args, model, b64, exact, prefixes, suffixes, contains, hint_set)
        except Exception as e:
            last = e
            if i < attempts - 1:
                logger.warning("attempt %d failed: %s; retrying", i + 1, e)
                time.sleep(3 * (i + 1))
    raise last


def normalize_tags(raw: str, exact: set[str], prefixes: list[str],
                   suffixes: list[str], contains: list[str], excluded: set[str],
                   hint_set: set[str], max_tags: int) -> list[str]:
    out, seen = [], set()
    for chunk in re.split(r"[,;\n]", raw):
        t = chunk.strip().lower().replace(" ", "_")
        t = re.sub(r"[^a-z0-9_]", "", t)
        if not t or t in seen or t in excluded:
            continue
        if t in hint_set:
            # canonical vocabulary always wins over the blacklist
            seen.add(t)
            out.append(t)
            continue
        if is_blacklisted(t, exact, prefixes, suffixes, contains):
            continue
        seen.add(t)
        out.append(t)
    return out[:max_tags]


def build_caption(trigger: str, header: list[str], tags: list[str]) -> str:
    parts = ([trigger] if trigger else []) + header + tags
    return ", ".join(p for p in parts if p)


def process_one(args, img: str, exact: set[str], prefixes: list[str],
                suffixes: list[str], contains: list[str], header: list[str],
                hint_set: set[str], model: str) -> str:
    path = os.path.join(args.folder, img)
    b64 = image_to_b64(path, args.max_size)
    raw = call_with_retry(args, model, b64, exact, prefixes, suffixes, contains, hint_set)
    # The trigger and header are prepended by the script; never let the model
    # duplicate them (VL models sometimes echo the subject name back).
    excluded = set(header)
    if args.trigger:
        excluded.add(args.trigger)
    tags = normalize_tags(raw, exact, prefixes, suffixes, contains, excluded,
                          hint_set, args.max_tags)
    return build_caption(args.trigger, header, tags)


def load_config(folder: str, explicit: str | None) -> tuple[dict, str | None]:
    """Load config from explicit path or auto-detect tagger.toml in the folder."""
    path = explicit or os.path.join(folder, "tagger.toml")
    if not os.path.exists(path):
        if explicit:
            sys.exit(f"error: config file not found: {path}")
        return {}, None
    with open(path, "rb") as f:
        return tomllib.load(f), path


def resolve_args(args, cfg: dict):
    """Merge CLI flags (winner) with config file values (defaults).

    Rule: CLI overrides config for scalar settings; --hint and --blacklist are
    additive (config entries + CLI entries).
    """
    if args.subject is None:
        args.subject = cfg.get("subject", "character")
    if args.trigger is None:
        args.trigger = cfg.get("trigger", "")
    if args.character is None:
        args.character = ", ".join(c for c in cfg.get("character", []) if isinstance(c, str))
    cfg_list = lambda key: [str(x).strip() for x in cfg.get(key, []) if str(x).strip()]
    cli_list = lambda s: [x.strip() for x in (s or "").split(",") if x.strip()]
    args.hint = ", ".join(cfg_list("hint") + cli_list(args.hint))
    args.blacklist = ", ".join(cfg_list("blacklist") + cli_list(args.blacklist))
    for attr, key, default in (("temperature", "temperature", 0.3),
                               ("max_tags", "max_tags", 40),
                               ("max_size", "max_size", 1280),
                               ("workers", "workers", 1),
                               ("base_url", "base_url", DEFAULT_BASE_URL),
                               ("model", "model", "")):
        if getattr(args, attr) is None:
            setattr(args, attr, cfg.get(key, default))


def write_toml(path: str, cfg: dict):
    lines = []
    for k, v in cfg.items():
        if isinstance(v, list):
            lines.append(f'{k} = [{ ", ".join(f'"{x}"' for x in v) }]')
        elif isinstance(v, str):
            lines.append(f'{k} = "{v}"')
        else:
            lines.append(f"{k} = {v}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def save_config(args):
    cfg = {
        "subject": args.subject,
        "trigger": args.trigger,
        "character": parse_tag_list(args.character),
        "hint": parse_tag_list(args.hint),
        "blacklist": parse_tag_list(args.blacklist),
        "temperature": args.temperature,
        "max_tags": args.max_tags,
        "max_size": args.max_size,
        "workers": args.workers,
        "base_url": args.base_url,
    }
    if args.model:
        cfg["model"] = args.model
    path = os.path.join(args.folder, "tagger.toml")
    write_toml(path, cfg)
    print(f"config saved to {path}")


SCAFFOLD_CONFIG = """\
# tagger.toml - dataset config (auto-loaded when you run `tagger .` here)
#
# Pick a school first (see README "Two schools"):
#   School 2 (recommended for identity/subject LoRAs): the TRIGGER owns the
#     subject. character = invariants only (e.g. 1girl); blacklist = the whole
#     subject vocabulary (*armor*, *scale*, *chain*, ...) so nothing but the
#     trigger describes it. Captions become short; max fidelity.
#   School 1 (decomposable/controllable): TAG the subject's features so you can
#     prompt them later. hint = canonical words, blacklist = synonyms only.
#     Human-verify the per-image tags (run `tagger . --audit`).
subject = "character"        # character | outfit | style
character = ["1girl"]        # invariant header: true on EVERY image
hint = []                    # canonical tags: use exactly these when visible
blacklist = []               # strip: 'foo' exact, 'foo*' prefix, '*foo' suffix, '*foo*' any

# optional: trigger = "mychar"     # token prepended to every caption
# optional: model = "qwen3-vl-8b-instruct"

temperature = 0.3
max_tags = 40
max_size = 1280
workers = 1
base_url = "http://localhost:1234/v1"
"""


def init_config(folder: str):
    """Write a commented starter tagger.toml; refuse to clobber an existing one."""
    path = os.path.join(folder, "tagger.toml")
    if os.path.exists(path):
        sys.exit(f"error: {path} already exists - edit it instead of re-initializing")
    with open(path, "w", encoding="utf-8") as f:
        f.write(SCAFFOLD_CONFIG)
    print(f"starter config written to {path}")
    print("edit it by hand, or override with CLI flags and run `tagger . --save-config`")
    print("to persist the effective settings.")


def print_tag_report(caption_counts: dict[str, int], total_images: int):
    """Print tag frequency over the final captions (what the trainer sees).

    Imbalance or singletons reveal imprinting risk and hallucinated tags:
    a color on 1 image is suspicious; a color on 12 of 25 is a dominant variant.
    """
    if not caption_counts:
        return
    total = sum(caption_counts.values())
    rows = sorted(caption_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    print(f"\ntag frequency report ({total_images} captions, {total} tags, {len(rows)} unique):")
    for tag, n in rows[:60]:
        print(f"  {n:3d}x  {tag}")
    if len(rows) > 60:
        print(f"  ... and {len(rows) - 60} more unique tags")
    singles = [t for t, n in rows if n == 1]
    if singles:
        print("singletons (1 image only - rare feature or hallucination):")
        print("  " + ", ".join(singles[:40]))


def audit_folder(folder: str):
    """Human review pass: report tag frequency over existing captions, no API calls."""
    image_stems = {os.path.splitext(f)[0] for f in os.listdir(folder)
                   if os.path.splitext(f)[1].lower() in IMAGE_EXTS}
    counts: dict[str, int] = {}
    captioned = 0
    for f in sorted(os.listdir(folder), key=natural_key):
        if not f.lower().endswith(".txt"):
            continue
        stem = f[:-4]
        if stem not in image_stems:
            continue
        content = open(os.path.join(folder, f), encoding="utf-8").read().strip()
        if not content:
            continue
        captioned += 1
        for t in parse_tag_list(content):
            counts[t] = counts.get(t, 0) + 1
    print(f"auditing existing captions: {captioned} non-empty .txt found")
    print_tag_report(counts, captioned)


REVIEW_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>tagger review — {{FOLDER}}</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin:0; font-family: system-ui, sans-serif; background:#141414; color:#ddd; }
  header { position:sticky; top:0; z-index:10; background:#1b1b1b; border-bottom:1px solid #2c2c2c; padding:10px 16px; }
  h1 { font-size:15px; margin:0 0 8px; color:#fff; font-weight:600; }
  h1 small { color:#888; font-weight:400; }
  .row { display:flex; align-items:center; gap:10px; flex-wrap:wrap; }
  .stats { font-size:13px; color:#aaa; }
  .chips { display:flex; gap:6px; flex-wrap:wrap; }
  .chip { background:#2a2a2a; border:1px solid #3a3a3a; border-radius:12px; padding:2px 10px; font-size:12px; cursor:pointer; }
  .chip:hover { background:#333; }
  .chip.on { background:#5a1f1f; border-color:#ff8787; }
  .chip.single { color:#ff8787; }
  select, .btn { background:#2a2a2a; color:#ddd; border:1px solid #3a3a3a; border-radius:6px; padding:4px 10px; font-size:13px; cursor:pointer; }
  a.btn { text-decoration:none; display:inline-block; }
  .btn:hover { background:#333; }
  .grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(340px,1fr)); gap:14px; padding:16px; }
  .card { background:#1b1b1b; border:1px solid #2c2c2c; border-radius:10px; overflow:hidden; display:flex; flex-direction:column; }
  .card img { width:100%; aspect-ratio:4/3; object-fit:contain; background:#000; display:block; }
  .bar { display:flex; justify-content:space-between; align-items:center; gap:8px; padding:6px 10px; font-size:12px; color:#999; }
  .badge { font-size:10px; font-weight:700; letter-spacing:.5px; padding:2px 7px; border-radius:8px; }
  .b-ok { display:none; }
  .b-empty { background:#5a4a1f; color:#ffd43b; }
  .b-missing { background:#5a1f1f; color:#ff8787; }
  .cap { padding:0 10px 10px; display:flex; flex-direction:column; gap:8px; }
  .view { font:12px/1.5 ui-monospace, monospace; color:#ddd; padding:8px; background:#131313; border:1px dashed #333; border-radius:6px; cursor:text; }
  .view:hover { border-color:#555; }
  .hint { font-size:12px; color:#777; font-style:italic; padding:8px; border:1px dashed #333; border-radius:6px; cursor:text; }
  mark.need { background:#5a1f1f; color:#ff8787; border-radius:3px; padding:0 2px; }
  textarea { width:100%; min-height:90px; resize:vertical; background:#131313; color:#ddd; border:1px solid #444; border-radius:6px; padding:8px; font:12px/1.5 ui-monospace, monospace; }
  .btns { display:flex; gap:8px; }
  .saved { color:#69db7c; font-size:12px; }
  .foot { padding:0 16px 20px; font-size:12px; color:#777; }
</style>
</head>
<body>
<header>
  <h1>tagger review <small>— {{FOLDER}}</small></h1>
  <div class="row">
    <span class="stats" id="stats"></span>
    <select id="filter">
      <option value="all">all images</option>
      <option value="attention">⚠ needs attention</option>
      <option value="ok">captioned only</option>
      <option value="empty">empty / missing</option>
    </select>
    <a class="btn" href="/setup">⚙ setup wizard</a>
    <button class="btn" onclick="load()">↻ refresh flags</button>
    <span style="font-size:12px;color:#666" id="shown"></span>
  </div>
  <div class="chips" id="chips"></div>
</header>
<main>
  <div class="grid" id="grid"></div>
  <div class="foot">click a caption to edit · Ctrl+Enter saves · Esc cancels · red tags appear on only 1 image (hallucination / rare-variant suspects)</div>
</main>
<script>
"use strict";
const state = { data: null, filter: 'all', tagFilter: null };

async function load() {
  const r = await fetch('/api/data');
  state.data = await r.json();
  render();
}

function render() {
  const d = state.data;
  if (!d) return;
  document.getElementById('stats').textContent =
    d.stats.captioned + '/' + d.stats.total + ' captioned · ' + d.stats.empty + ' empty · ' + d.stats.missing + ' missing' +
    (d.stats.orphan_txt.length ? ' · ⚠ ' + d.stats.orphan_txt.length + ' orphan .txt: ' + d.stats.orphan_txt.join(', ') : '');
  const chips = document.getElementById('chips');
  chips.innerHTML = '';
  if (!d.stats.singleton_tags.length) {
    const c = document.createElement('span');
    c.className = 'chip';
    c.textContent = 'no singleton tags 🎉';
    chips.appendChild(c);
  }
  for (const t of d.stats.singleton_tags) {
    const c = document.createElement('span');
    c.className = 'chip single' + (state.tagFilter === t ? ' on' : '');
    c.textContent = t + ' ×1';
    c.title = 'click to show only images with this tag';
    c.onclick = () => { state.tagFilter = state.tagFilter === t ? null : t; render(); };
    chips.appendChild(c);
  }
  const grid = document.getElementById('grid');
  grid.innerHTML = '';
  for (const img of d.images) if (passes(img)) grid.appendChild(card(img));
  document.getElementById('shown').textContent = 'showing ' + grid.children.length + '/' + d.images.length;
}

function passes(img) {
  if (state.tagFilter && !img.singletons.includes(state.tagFilter)) return false;
  switch (state.filter) {
    case 'attention': return img.singletons.length > 0 || img.state !== 'ok';
    case 'ok': return img.state === 'ok';
    case 'empty': return img.state !== 'ok';
    default: return true;
  }
}

function esc(s) { return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

function card(img) {
  const el = document.createElement('div');
  el.className = 'card';
  const im = document.createElement('img');
  im.src = '/img/' + encodeURIComponent(img.name) + '?size=340';
  im.loading = 'lazy';
  im.alt = img.name;
  el.appendChild(im);
  const bar = document.createElement('div'); bar.className = 'bar';
  const nm = document.createElement('span'); nm.textContent = img.name;
  const badge = document.createElement('span');
  badge.className = 'badge ' + (img.state === 'empty' ? 'b-empty' : img.state === 'missing' ? 'b-missing' : 'b-ok');
  badge.textContent = img.state === 'empty' ? 'EMPTY' : img.state === 'missing' ? 'NO CAPTION' : '';
  bar.appendChild(nm); bar.appendChild(badge);
  el.appendChild(bar);
  const cap = document.createElement('div'); cap.className = 'cap';
  if (img.state === 'ok') {
    const view = document.createElement('div'); view.className = 'view';
    const sg = new Set(img.singletons);
    view.innerHTML = img.caption.split(',').map(t => t.trim()).filter(Boolean)
      .map(t => sg.has(t) ? '<mark class="need">' + esc(t) + '</mark>' : esc(t)).join(', ');
    view.title = 'click to edit';
    view.onclick = () => editMode(el, img, cap);
    cap.appendChild(view);
  } else {
    const hint = document.createElement('div'); hint.className = 'hint';
    hint.textContent = 'no caption yet — click to write one';
    hint.onclick = () => editMode(el, img, cap);
    cap.appendChild(hint);
  }
  el.appendChild(cap);
  return el;
}

function editMode(el, img, cap) {
  const ta = document.createElement('textarea');
  ta.value = img.caption || '';
  const btns = document.createElement('div'); btns.className = 'btns';
  const save = document.createElement('button'); save.className = 'btn'; save.textContent = 'save';
  const cancel = document.createElement('button'); cancel.className = 'btn'; cancel.textContent = 'cancel';
  save.onclick = () => saveImg(el, img, ta.value);
  cancel.onclick = () => render();
  btns.appendChild(save); btns.appendChild(cancel);
  cap.innerHTML = '';
  cap.appendChild(ta); cap.appendChild(btns);
  ta.focus();
  ta.addEventListener('keydown', e => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') { e.preventDefault(); saveImg(el, img, ta.value); }
    if (e.key === 'Escape') render();
  });
}

async function saveImg(el, img, text) {
  const r = await fetch('/api/save', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name: img.name, caption: text })
  });
  const j = await r.json();
  if (!j.ok) { alert('save failed: ' + (j.error || 'unknown')); return; }
  img.caption = text; img.state = text.trim() ? 'ok' : 'empty';
  const badge = el.querySelector('.badge');
  if (text.trim()) { badge.textContent = ''; badge.className = 'badge b-ok'; }
  else { badge.textContent = 'EMPTY'; badge.className = 'badge b-empty'; }
  const msg = document.createElement('div'); msg.className = 'saved'; msg.textContent = 'saved ✓';
  el.querySelector('.cap').appendChild(msg);
  setTimeout(() => msg.remove(), 1500);
}

document.getElementById('filter').addEventListener('change', e => { state.filter = e.target.value; render(); });
load();
</script>
</body>
</html>"""


SETUP_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>tagger setup — {{FOLDER}}</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin:0; font-family: system-ui, sans-serif; background:#141414; color:#ddd; padding-bottom:40px; }
  header { position:sticky; top:0; z-index:10; background:#1b1b1b; border-bottom:1px solid #2c2c2c; padding:10px 16px; }
  h1 { font-size:15px; margin:0; color:#fff; font-weight:600; }
  h1 small { color:#888; font-weight:400; }
  .nav { margin-top:6px; display:flex; gap:10px; font-size:13px; }
  .nav a { color:#69db7c; text-decoration:none; }
  .wrap { max-width:880px; margin:0 auto; padding:20px 16px; }
  section { background:#1b1b1b; border:1px solid #2c2c2c; border-radius:10px; padding:14px 16px; margin-bottom:14px; }
  section h2 { font-size:13px; margin:0 0 4px; color:#fff; text-transform:uppercase; letter-spacing:.5px; }
  .desc { font-size:12.5px; color:#999; margin:0 0 10px; line-height:1.5; }
  code { background:#222; padding:1px 4px; border-radius:4px; font-size:11px; color:#9ecbff; }
  .radio-row { display:flex; gap:10px; flex-wrap:wrap; }
  .card { flex:1; min-width:230px; border:1px solid #3a3a3a; border-radius:8px; padding:10px 12px; cursor:pointer; background:#222; }
  .card:hover { border-color:#555; }
  .card.on { border-color:#69db7c; background:#1d2b1d; }
  .card h3 { margin:0 0 4px; font-size:13px; color:#fff; }
  .card p { margin:0; font-size:12px; color:#aaa; line-height:1.45; }
  .tag { font-size:10px; color:#ffd43b; background:#5a4a1f; border-radius:8px; padding:1px 7px; margin-left:6px; }
  .trigger-wrap { display:flex; gap:8px; align-items:center; max-width:520px; }
  .trigger-wrap input { font-family:ui-monospace,monospace; flex:1; }
  input[type=text] { width:100%; background:#131313; color:#ddd; border:1px solid #444; border-radius:6px; padding:8px 10px; font-size:13px; }
  .warn { font-size:12px; color:#ff8787; margin-top:6px; }
  .ok { font-size:12px; color:#69db7c; margin-top:6px; }
  .chips { display:flex; flex-wrap:wrap; gap:6px; padding:8px 0 0; }
  .chip { background:#2a2a2a; border:1px solid #3a3a3a; border-radius:12px; padding:2px 6px 2px 10px; font-size:12px; display:inline-flex; align-items:center; gap:6px; }
  .chip button { background:none; border:none; color:#999; cursor:pointer; font-size:13px; padding:0; line-height:1; }
  .chip button:hover { color:#ff8787; }
  .chip.gen { border-color:#4a5a4a; color:#9ef0a9; }
  .chip.junk { border-style:dashed; color:#ccc; }
  .add-row { display:flex; gap:8px; margin-top:10px; }
  .add-row input { flex:1; }
  .btn { background:#2a2a2a; color:#ddd; border:1px solid #3a3a3a; border-radius:6px; padding:6px 14px; font-size:13px; cursor:pointer; }
  .btn:hover { background:#333; }
  .btn.primary { background:#2b4a2b; border-color:#69db7c; color:#c8f7c8; }
  .btn.primary:hover { background:#345c34; }
  .preview { background:#131313; border:1px solid #333; border-radius:8px; padding:10px 12px; margin-top:12px; }
  .pv-line { font-size:12px; font-family:ui-monospace,monospace; color:#ccc; margin:3px 0; }
  .pv-line .n { color:#69db7c; }
  .pv-line .none { color:#ff8787; }
  .sec { font-size:11px; color:#777; text-transform:uppercase; letter-spacing:.5px; margin:12px 0 0; }
  pre.toml { background:#0d0d0d; border:1px solid #333; border-radius:8px; padding:12px; font:12px/1.6 ui-monospace,monospace; color:#9ecbff; overflow-x:auto; margin:0; }
  .saved { color:#69db7c; font-size:13px; }
  .saved a { color:#69db7c; }
  .note { background:#2a2410; border:1px solid #5a4a1f; color:#ffd43b; font-size:12.5px; border-radius:8px; padding:8px 12px; margin-top:8px; display:none; }
  .stepnum { display:inline-block; width:18px; height:18px; line-height:18px; text-align:center; background:#2a2a2a; color:#fff; border-radius:50%; font-size:11px; margin-right:8px; }
</style>
</head>
<body>
<header>
  <h1>tagger setup <small>— {{FOLDER}}</small></h1>
  <div class="nav"><a href="/">← review grid</a></div>
</header>
<div class="wrap">

<section>
  <h2><span class="stepnum">1</span>What are we training?</h2>
  <div class="radio-row" id="subjectRow"></div>
</section>

<section>
  <h2><span class="stepnum">2</span>Trigger word</h2>
  <div class="desc">The word that summons the subject when prompting. Lowercase, no spaces.</div>
  <div class="trigger-wrap"><input type="text" id="trigger" placeholder="e.g. bg3_wavemother_robe"></div>
  <div id="triggerMsg"></div>
</section>

<section>
  <h2><span class="stepnum">3</span>School</h2>
  <div class="desc">How much the tags may say about the subject's look.</div>
  <div class="radio-row" id="schoolRow"></div>
  <div class="note" id="flipNote"></div>
</section>

<section>
  <h2><span class="stepnum">4</span>True on every image <small style="color:#888">(header)</small></h2>
  <div class="desc">Invariant tags — put in the caption of every image. For identity LoRAs: only non-subject invariants (e.g. <code>1girl</code>).</div>
  <div class="chips" id="characterChips"></div>
  <div class="add-row"><input type="text" id="characterInput" placeholder="type a tag, Enter to add"><button class="btn" id="characterAdd">+</button></div>
</section>

<section>
  <h2><span class="stepnum">5</span>Varies per image, keep taggable <small style="color:#888">(hint)</small></h2>
  <div class="desc" id="hintDesc"></div>
  <div class="chips" id="hintChips"></div>
  <div class="add-row"><input type="text" id="hintInput" placeholder="type a tag, Enter to add"><button class="btn" id="hintAdd">+</button></div>
</section>

<section>
  <h2><span class="stepnum">6</span>Describe the subject's look <small style="color:#888">(generates the blocklist)</small></h2>
  <div class="desc" id="wordsDesc"></div>
  <div class="add-row"><input type="text" id="wordsInput" placeholder="e.g. armor, dress, gown, cape, scale, chain, sword, jewelry"><button class="btn" id="wordsAdd">generate</button></div>
  <div class="chips" id="wordsChips"></div>
  <div class="preview" id="wordsPreview"></div>
  <div class="sec">→ blocklist entries (generated + editable)</div>
  <div class="chips" id="blacklistChips"></div>
</section>

<section>
  <h2><span class="stepnum">7</span>Danbooru meta junk</h2>
  <div class="desc">Never useful for training (watermark, censored, translated…).</div>
  <label style="font-size:13px"><input type="checkbox" id="junkOn" checked> block meta junk</label>
  <div class="chips" id="junkChips"></div>
</section>

<section>
  <h2><span class="stepnum">8</span>Preview &amp; save</h2>
  <div class="desc">Written to <code>tagger.toml</code> in the dataset folder. Tuning values (temperature, workers, …) are kept if the config already exists.</div>
  <pre class="toml" id="tomlPreview"></pre>
  <div style="display:flex; gap:10px; align-items:center; margin-top:10px">
    <button class="btn primary" id="saveBtn">save tagger.toml</button>
    <span class="saved" id="savedMsg"></span>
  </div>
</section>

</div>
<script>
"use strict";
const byId = id => document.getElementById(id);

const SUBJECTS = [
  { id: 'character', name: 'character', desc: 'a person/character whose look is the subject' },
  { id: 'outfit', name: 'outfit', desc: 'a specific outfit/costume on a person' },
  { id: 'style', name: 'style', desc: 'an art style or texture — usually School 1' },
];
const SCHOOLS = [
  { id: 1, name: 'School 1 — tag the subject', rec: false, desc: 'The model names every detail (blue_armor, scale_mail…). You prompt trigger + details. Risk: details can summon the subject without the trigger (leakage).' },
  { id: 2, name: 'School 2 — the trigger owns it', rec: true, desc: 'Every subject detail is blocked; only the trigger names it. Prompting the trigger alone reproduces the subject. Recommended.' },
];
const COLOR_WORDS = ['blue','red','green','purple','pink','orange','yellow','black','white','brown','gray','grey','gold','silver','teal','turquoise','cyan','magenta','maroon','violet','indigo','beige','cream','crimson','scarlet','azure','bronze','copper','gilded'];
const MATERIAL_WORDS = ['armor','armour','leather','cloth','fabric','cotton','silk','wool','fur','metal','steel','iron','chain','mail','scale','plate','dress','gown','robe','cape','cloak','skirt','boot','glove','jewel','gem','sword','blade','helmet','crown'];

const state = {
  subject: 'character', trigger: '', school: 2,
  character: [], hint: [], blacklist: [], words: [],
  junkOn: true, junk: [], captionTags: {}, cfg: {}, datasetName: '',
};

function isLookish(t) {
  const low = t.toLowerCase();
  return COLOR_WORDS.some(w => low.includes(w)) || MATERIAL_WORDS.some(w => low.includes(w));
}
function isSubjectish(t) {
  const low = t.toLowerCase();
  return isLookish(low) || state.words.some(w => low.includes(w));
}
function wildcardize(word) {
  const w = word.trim().toLowerCase().replace(/[^a-z0-9_ ]/g, '').replace(/\s+/g, '_');
  if (!w) return [];
  const stems = new Set([w]);
  if (w.endsWith('s') && !w.endsWith('ss') && !w.endsWith('es') && w.length > 4) stems.add(w.slice(0, -1));
  return [...stems].sort().map(s => s.length <= 3 ? s : '*' + s + '*');
}

function bindChips(chipsEl, arr, cls, onRemove, onchange) {
  function render() {
    chipsEl.innerHTML = '';
    for (const c of arr) {
      const span = document.createElement('span');
      span.className = 'chip' + (cls ? ' ' + cls : '');
      span.textContent = c;
      const x = document.createElement('button');
      x.textContent = '✕';
      x.onclick = () => {
        const i = arr.indexOf(c);
        if (i >= 0) arr.splice(i, 1);
        if (onRemove) onRemove(c);
        render(); onchange();
      };
      span.appendChild(x);
      chipsEl.appendChild(span);
    }
  }
  render();
  return render;
}
function addChip(arr, input) {
  const v = input.value.trim().toLowerCase().replace(/\s+/g, '_');
  if (v && !arr.includes(v)) arr.push(v);
  input.value = '';
}
function bindInput(inputEl, btnEl, arr, render) {
  const add = () => { addChip(arr, inputEl); render(); onAnyChange(); };
  btnEl.onclick = add;
  inputEl.addEventListener('keydown', e => {
    if (e.key === 'Enter') { e.preventDefault(); add(); }
    else if (e.key === ',' && inputEl.value.trim()) { e.preventDefault(); add(); }
  });
}

function renderSubject() {
  const row = byId('subjectRow');
  row.innerHTML = '';
  for (const s of SUBJECTS) {
    const c = document.createElement('div');
    c.className = 'card' + (state.subject === s.id ? ' on' : '');
    c.onclick = () => {
      state.subject = s.id;
      if (s.id === 'style' && state.school === 2) { state.school = 1; flipNote('style datasets are usually School 1 — flipped'); }
      renderSubject(); renderSchool(); onAnyChange();
    };
    c.innerHTML = '<h3>' + s.name + '</h3><p>' + s.desc + '</p>';
    row.appendChild(c);
  }
}

function flipNote(text) {
  const n = byId('flipNote');
  n.textContent = text;
  n.style.display = text ? 'block' : 'none';
}
function renderSchool() {
  const row = byId('schoolRow');
  row.innerHTML = '';
  for (const s of SCHOOLS) {
    const c = document.createElement('div');
    c.className = 'card' + (state.school === s.id ? ' on' : '');
    c.onclick = () => flipSchool(s.id);
    c.innerHTML = '<h3>' + s.name + (s.rec ? '<span class="tag">recommended</span>' : '') + '</h3><p>' + s.desc + '</p>';
    row.appendChild(c);
  }
}
function flipSchool(newSchool) {
  if (newSchool === state.school) return;
  const moved = [];
  if (newSchool === 2) {            // 1 → 2: subject details move hint → blacklist
    state.hint = state.hint.filter(t => {
      if (isSubjectish(t)) { moved.push(t); return false; }
      return true;
    });
    for (const t of moved) for (const p of wildcardize(t)) if (!state.blacklist.includes(p)) state.blacklist.push(p);
  } else {                          // 2 → 1: subject rules move blacklist → hint
    const keep = [];
    const movedSet = new Set();
    for (const p of state.blacklist) {
      if (p.startsWith('*') && p.endsWith('*')) {
        const stem = p.slice(1, -1);
        if (isSubjectish(stem)) { movedSet.add(stem); continue; }
      }
      keep.push(p);
    }
    state.blacklist = keep;
    for (const t of movedSet) if (!state.hint.includes(t)) state.hint.push(t);
    moved.push(...movedSet);
  }
  state.school = newSchool;
  renderSchool();
  renderChipsW(); renderChipsBL(); renderChipsH();
  renderWordsPreview();
  onAnyChange();
  flipNote('flipped — moved ' + moved.length + ' tag' + (moved.length === 1 ? '' : 's') + ': ' + moved.slice(0, 6).join(', ') + (moved.length > 6 ? ' …' : ''));
}

function renderWordsPreview() {
  const el = byId('wordsPreview');
  el.innerHTML = '';
  for (const w of state.words) {
    const pats = wildcardize(w);
    const matches = Object.keys(state.captionTags)
      .filter(t => pats.some(p => p.startsWith('*') ? t.includes(p.slice(1, -1)) : t === p))
      .sort();
    const head = document.createElement('div');
    head.className = 'pv-line';
    head.textContent = w + ' → ' + pats.join(', ');
    el.appendChild(head);
    const sub = document.createElement('div');
    sub.className = 'pv-line';
    if (matches.length) {
      sub.innerHTML = '<span class="n">' + matches.length + (matches.length === 1 ? ' tag matched' : ' tags matched') + ':</span> ' + matches.slice(0, 10).join(', ') + (matches.length > 10 ? ' …' : '');
    } else {
      sub.innerHTML = '<span class="none">0 matches in current captions</span> — no captions yet, or try a synonym';
    }
    el.appendChild(sub);
  }
}
function generateWords() {
  const raw = byId('wordsInput').value.split(/[, ]+/).map(s => s.trim().toLowerCase()).filter(Boolean);
  if (!raw.length) return;
  byId('wordsInput').value = '';
  for (const w of raw) {
    if (!state.words.includes(w)) state.words.push(w);
    for (const p of wildcardize(w)) if (!state.blacklist.includes(p)) state.blacklist.push(p);
  }
  renderWords(); renderBlacklist(); renderWordsPreview(); onAnyChange();
}
function removeWord(w) {
  const pats = wildcardize(w);
  state.blacklist = state.blacklist.filter(p => !pats.includes(p));
}

function renderJunk() {
  const el = byId('junkChips');
  el.innerHTML = '';
  if (!state.junkOn) return;
  for (const j of state.junk) {
    const span = document.createElement('span');
    span.className = 'chip junk';
    span.textContent = j;
    const x = document.createElement('button');
    x.textContent = '✕';
    x.onclick = () => { state.junk.splice(state.junk.indexOf(j), 1); renderJunk(); onAnyChange(); };
    span.appendChild(x);
    el.appendChild(span);
  }
}

function fmtList(arr) { return '[ ' + arr.map(x => '"' + x + '"').join(', ') + ' ]'; }
function renderToml() {
  const lines = [
    '# tagger.toml — written by the setup wizard (tagger --setup)',
    '# dataset: ' + state.datasetName,
    '',
    'subject = "' + state.subject + '"',
    'trigger = "' + state.trigger + '"',
    '',
    '# header: invariant tags, true for EVERY image',
    'character = ' + fmtList(state.character),
    '',
    '# hint: identity that varies per image and must stay taggable',
    'hint = ' + fmtList(state.hint),
    '',
    '# blacklist: never emitted. *word* = substring match',
    'blacklist = ' + fmtList([...state.blacklist, ...(state.junkOn ? state.junk : [])]),
  ];
  const extras = [];
  for (const k of ['temperature', 'max_tags', 'max_size', 'workers', 'base_url', 'model']) {
    if (state.cfg[k] !== undefined) extras.push(k + ' = ' + (typeof state.cfg[k] === 'string' ? '"' + state.cfg[k] + '"' : state.cfg[k]));
  }
  if (extras.length) lines.push('', '# tuning (kept from your existing config)', ...extras);
  byId('tomlPreview').textContent = lines.join('\n');
}
function renderTriggerMsg() {
  const msg = byId('triggerMsg');
  const t = state.trigger;
  if (!t) { msg.className = 'warn'; msg.textContent = 'trigger is required'; }
  else if (!/^[a-z0-9_]+$/.test(t)) { msg.className = 'warn'; msg.textContent = 'lowercase letters, digits and underscores only (no spaces)'; }
  else { msg.className = 'ok'; msg.textContent = '✓ valid'; }
}
function renderDescs() {
  byId('wordsDesc').textContent = state.school === 2
    ? 'Plain words — materials, garments, props. Each word becomes a block rule (*word* = matches any tag containing it) so the model NEVER names the subject’s look; only the trigger does.'
    : 'Words whose tags you want blocked (over-broad compounds, unwanted detail). Under School 1 the subject’s look stays taggable.';
  byId('hintDesc').innerHTML = state.school === 2
    ? 'Identity that changes between images and must STAY taggable (e.g. <code>face_paint</code>). Under School 2 this is only non-subject identity — the subject’s look belongs to the trigger.'
    : 'The subject’s look, named as specific tags (e.g. <code>blue_armor</code>, <code>scale_mail</code>). Under School 1 the model tags the subject freely; hint pins the names of the variants.';
}
function onAnyChange() {
  renderTriggerMsg();
  renderDescs();
  renderToml();
}

async function save() {
  if (!state.trigger || !/^[a-z0-9_]+$/.test(state.trigger)) { alert('fix the trigger first: lowercase, digits, underscores only'); return; }
  const blacklist = [...state.blacklist];
  if (state.junkOn) for (const j of state.junk) if (!blacklist.includes(j)) blacklist.push(j);
  const r = await fetch('/api/setup-save', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ subject: state.subject, trigger: state.trigger, character: state.character, hint: state.hint, blacklist }),
  });
  const j = await r.json();
  if (!j.ok) { alert('save failed: ' + (j.error || 'unknown')); return; }
  const msg = byId('savedMsg');
  msg.textContent = 'saved ✓ — ';
  const a = document.createElement('a'); a.href = '/'; a.textContent = 'open the review grid →';
  msg.appendChild(a);
}

async function init() {
  const d = await (await fetch('/api/setup-config')).json();
  state.captionTags = d.caption_tags || {};
  state.datasetName = d.dataset || d.folder;
  state.cfg = d.cfg || {};
  if (d.cfg && Object.keys(d.cfg).length) {
    state.subject = d.cfg.subject || 'character';
    state.trigger = d.cfg.trigger || '';
    state.character = [...(d.cfg.character || [])];
    state.hint = [...(d.cfg.hint || [])];
    const junk = d.junk || [];
    const bl = d.cfg.blacklist || [];
    state.blacklist = bl.filter(x => !junk.includes(x));
    state.junk = bl.filter(x => junk.includes(x));
    state.junkOn = state.junk.length > 0;
    const stems = [...new Set(bl.filter(x => x.startsWith('*') && x.endsWith('*'))
      .map(x => x.slice(1, -1)).filter(s => /^[a-z0-9_]{4,}$/.test(s)))];
    state.words = stems;
    state.school = stems.filter(isLookish).length >= 3 ? 2 : 1;
  } else {
    state.junk = [...d.junk];
    state.junkOn = true;
  }
  FULL_JUNK = [...d.junk];
  renderSubject(); renderSchool();
  renderChipsC(); renderChipsH(); renderChipsBL(); renderChipsW();
  renderWordsPreview();
  renderJunk();
  onAnyChange();
  if (d.cfg && Object.keys(d.cfg).length && state.words.length) flipNote('loaded existing config — School ' + state.school + ' inferred from ' + state.words.length + ' subject words');
}

const renderChipsH = bindChips(byId('hintChips'), state.hint, null, null, onAnyChange);
const renderChipsBL = bindChips(byId('blacklistChips'), state.blacklist, null, null, onAnyChange);
const renderChipsW = bindChips(byId('wordsChips'), state.words, 'gen', removeWord, () => { renderWordsPreview(); onAnyChange(); });
const renderChipsC = bindChips(byId('characterChips'), state.character, null, null, onAnyChange);
let FULL_JUNK = [];

bindInput(byId('characterInput'), byId('characterAdd'), state.character, renderChipsC);
byId('hintAdd').onclick = () => { addChip(state.hint, byId('hintInput')); renderChipsH(); onAnyChange(); };
byId('hintInput').addEventListener('keydown', e => {
  if (e.key === 'Enter') { e.preventDefault(); byId('hintAdd').onclick(); }
  else if (e.key === ',' && byId('hintInput').value.trim()) { e.preventDefault(); byId('hintAdd').onclick(); }
});

byId('wordsAdd').onclick = generateWords;
byId('wordsInput').addEventListener('keydown', e => { if (e.key === 'Enter') { e.preventDefault(); generateWords(); } });
byId('trigger').addEventListener('input', e => { state.trigger = e.target.value.trim(); onAnyChange(); });
byId('junkOn').addEventListener('change', e => {
  state.junkOn = e.target.checked;
  if (state.junkOn && !state.junk.length) state.junk = [...FULL_JUNK];
  renderJunk(); onAnyChange();
});
byId('saveBtn').onclick = save;

init();
</script>
</body>
</html>"""


META_JUNK = ["watermark", "signature", "censored", "commentary", "translated",
             "bad_id", "bad_pixiv_id", "score_*", "rating_*", "text_focus",
             "logo", "monochrome", "greyscale", "multiple_views", "reference_sheet"]


def sanitize_list(items) -> list[str]:
    """Strip, drop empties, dedupe preserving order."""
    out = []
    for x in items or []:
        s = str(x).strip()
        if s and s not in out:
            out.append(s)
    return out


def fmt_toml_list(items) -> str:
    return f'[ { ", ".join(f"\"{x}\"" for x in items) } ]'


def setup_payload(folder: str) -> dict:
    """Prefill + preview data for the setup wizard."""
    cfg = {}
    path = os.path.join(folder, "tagger.toml")
    if os.path.exists(path):
        with open(path, "rb") as f:
            cfg = tomllib.load(f)
    tags: dict[str, int] = {}
    for f in os.listdir(folder):
        if f.lower().endswith(".txt"):
            content = open(os.path.join(folder, f), encoding="utf-8").read()
            for t in parse_tag_list(content):
                tags[t] = tags.get(t, 0) + 1
    return {"folder": folder, "dataset": os.path.basename(folder),
            "exists": bool(cfg), "cfg": cfg,
            "caption_tags": tags, "junk": META_JUNK}


def setup_save(folder: str, body: dict) -> dict:
    """Validate the wizard form and write tagger.toml (keeping existing tunables)."""
    subject = str(body.get("subject", ""))
    trigger = str(body.get("trigger", "")).strip()
    if subject not in ("character", "outfit", "style"):
        return {"ok": False, "error": f"subject must be character|outfit|style, got {subject!r}"}
    if not re.match(r"^[a-z0-9_]+$", trigger):
        return {"ok": False,
                "error": "trigger must be lowercase snake_case (a-z, 0-9, _)"}
    character = sanitize_list(body.get("character"))
    hint = sanitize_list(body.get("hint"))
    blacklist = sanitize_list(body.get("blacklist"))
    path = os.path.join(folder, "tagger.toml")
    tunables: dict = {}
    if os.path.exists(path):
        with open(path, "rb") as f:
            tunables = tomllib.load(f)
    lines = [
        "# tagger.toml - written by the setup wizard (tagger --setup)",
        f"# dataset: {os.path.basename(folder)}",
        "",
        f'subject = "{subject}"   # what are we training: character | outfit | style',
        f'trigger = "{trigger}"   # the word that summons the subject',
        "",
        "# header: invariant tags, true for EVERY image",
        f"character = {fmt_toml_list(character)}",
        "",
        "# hint: identity that varies per image and must stay taggable",
        f"hint = {fmt_toml_list(hint)}",
        "",
        "# blacklist: never emitted. *word* = substring match",
        f"blacklist = {fmt_toml_list(blacklist)}",
    ]
    keep = {k: v for k, v in tunables.items()
            if k not in ("subject", "trigger", "character", "hint", "blacklist")}
    if keep:
        lines.append("")
        lines.append("# tuning (kept from your previous config)")
        for k, v in keep.items():
            if isinstance(v, list):
                lines.append(f"{k} = {fmt_toml_list(v)}")
            elif isinstance(v, str):
                lines.append(f'{k} = "{v}"')
            else:
                lines.append(f"{k} = {v}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("setup wizard wrote config: %s", path)
    return {"ok": True, "path": path}


def build_review_data(folder: str) -> dict:
    """Snapshot of the folder for the review grid: images + captions + flags."""
    images = sorted((f for f in os.listdir(folder)
                     if os.path.splitext(f)[1].lower() in IMAGE_EXTS
                     and os.path.isfile(os.path.join(folder, f))),
                    key=natural_key)
    counts: dict[str, int] = {}
    entries = []
    empty = missing = 0
    for name in images:
        stem = os.path.splitext(name)[0]
        txt = os.path.join(folder, stem + ".txt")
        exists = os.path.exists(txt)
        content = open(txt, encoding="utf-8").read().strip() if exists else ""
        if exists and not content:
            empty += 1
        if not exists:
            missing += 1
        for t in parse_tag_list(content):
            counts[t] = counts.get(t, 0) + 1
        entries.append({"name": name, "caption": content,
                        "state": "ok" if content else ("empty" if exists else "missing"),
                        "singletons": []})
    singleton_tags = sorted(t for t, n in counts.items() if n == 1)
    singleton_set = set(singleton_tags)
    for e in entries:
        e["singletons"] = sorted(t for t in parse_tag_list(e["caption"])
                                  if t in singleton_set)
    image_stems = {os.path.splitext(n)[0] for n in images}
    orphan_txt = sorted(f for f in os.listdir(folder)
                        if f.lower().endswith(".txt")
                        and os.path.splitext(f)[0] not in image_stems)
    return {"folder": folder, "images": entries,
            "stats": {"total": len(images),
                      "captioned": sum(1 for e in entries if e["state"] == "ok"),
                      "empty": empty, "missing": missing,
                      "singleton_tags": singleton_tags, "orphan_txt": orphan_txt}}


def make_app_handler(folder: str):
    """HTTP handler factory for the local app (review grid + setup wizard)."""
    _thumb_cache: dict = {}

    def _thumb(name: str, size: int) -> bytes:
        key = (name, size)
        if key in _thumb_cache:
            return _thumb_cache[key]
        with Image.open(os.path.join(folder, name)) as im:
            im = im.convert("RGB")
            im.thumbnail((size, size), Image.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=85)
            blob = buf.getvalue()
        if len(_thumb_cache) > 300:
            _thumb_cache.clear()
        _thumb_cache[key] = blob
        return blob

    class ReviewHandler(BaseHTTPRequestHandler):
        def _respond(self, blob: bytes, ctype: str, status: int = 200,
                     cache: bool = False):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(blob)))
            if cache:
                self.send_header("Cache-Control", "max-age=3600")
            self.end_headers()
            self.wfile.write(blob)

        def _json(self, obj, status: int = 200):
            self._respond(json.dumps(obj).encode("utf-8"),
                          "application/json", status)

        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path == "/":
                return self._respond(
                    REVIEW_HTML.replace("{{FOLDER}}", folder).encode("utf-8"),
                    "text/html; charset=utf-8")
            if parsed.path == "/api/data":
                return self._json(build_review_data(folder))
            if parsed.path == "/setup":
                return self._respond(
                    SETUP_HTML.replace("{{FOLDER}}", folder).encode("utf-8"),
                    "text/html; charset=utf-8")
            if parsed.path == "/api/setup-config":
                return self._json(setup_payload(folder))
            if parsed.path.startswith("/img/"):
                name = os.path.basename(
                    urllib.parse.unquote(parsed.path[len("/img/"):]))
                full = os.path.join(folder, name)
                ext = os.path.splitext(name)[1].lower()
                if not (os.path.isfile(full) and ext in IMAGE_EXTS):
                    return self.send_error(404)
                qs = urllib.parse.parse_qs(parsed.query)
                try:
                    size = int(qs.get("size", ["0"])[0])
                except ValueError:
                    size = 0
                if size > 0:
                    blob = _thumb(name, size)
                    return self._respond(blob, "image/jpeg", cache=True)
                with open(full, "rb") as f:
                    blob = f.read()
                ctype = {".png": "image/png", ".webp": "image/webp"}.get(
                    ext, "image/jpeg")
                return self._respond(blob, ctype, cache=True)
            self.send_error(404)

        def do_POST(self):
            path = urllib.parse.urlparse(self.path).path
            length = int(self.headers.get("Content-Length", 0) or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except Exception:
                return self._json({"ok": False, "error": "bad json"}, 400)
            if path == "/api/save":
                name = os.path.basename(str(body.get("name", "")))
                caption = str(body.get("caption", "")).strip()
                full = os.path.join(folder, name)
                if not (os.path.isfile(full)
                        and os.path.splitext(name)[1].lower() in IMAGE_EXTS):
                    return self._json({"ok": False, "error": "unknown image"}, 400)
                txt = os.path.join(folder, os.path.splitext(name)[0] + ".txt")
                with open(txt, "w", encoding="utf-8") as f:
                    f.write(caption)
                logger.info("review save: %s (%d chars)", name, len(caption))
                return self._json({"ok": True, "name": name})
            if path == "/api/setup-save":
                return self._json(setup_save(folder, body))
            self.send_error(404)

        def log_message(self, fmt, *args):
            pass

    return ReviewHandler


def serve_app(folder: str, port: int, page: str, open_browser: bool):
    """Serve the local web app (review grid or setup wizard) until Ctrl+C."""
    if not any(os.path.splitext(f)[1].lower() in IMAGE_EXTS
               for f in os.listdir(folder)):
        sys.exit(f"no images ({', '.join(sorted(IMAGE_EXTS))}) found in {folder}")
    url = f"http://127.0.0.1:{port}{'/setup' if page == 'setup' else ''}"
    print(f"tagger {'setup wizard' if page == 'setup' else 'review grid'}: {url}  (Ctrl+C to stop)")
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        ThreadingHTTPServer(("127.0.0.1", port),
                            make_app_handler(folder)).serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    except OSError as e:
        sys.exit(f"error: cannot bind port {port}: {e}\n"
                 f"(is another instance running? use --port to change it)")


def tag_batch(args, model: str, batch: list[tuple[str, str]], exact: set[str],
              prefixes: list[str], suffixes: list[str], contains: list[str],
              hint_set: set[str]) -> str:
    """Tag multiple images in ONE call so the model aligns vocabulary across the set."""
    names = [name for name, _ in batch]
    payload = {
        "model": model,
        "temperature": args.temperature,
        "max_tokens": 4096,
        "messages": [
            {"role": "system",
             "content": system_prompt(args.subject, render_banned(exact, prefixes,
                                                                  suffixes, contains),
                                      ", ".join(sorted(hint_set)) or "(none)",
                                      args.max_tags)},
            {"role": "user", "content": [
                {"type": "text",
                 "text": f"Tag these {len(batch)} images. Output one line per image, "
                         f"starting each line with the image name and a colon, e.g.:\n"
                         + "\n".join(f"{n}: <tags>" for n in names)
                         + "\n\nImportant: these images are DIFFERENT (different crops, poses, "
                           "visible details). Tag each image independently from what is visible "
                           "in THAT image only. Never copy or reuse tags from another image "
                           "unless that feature is actually visible in it. Use the same tag "
                           "NAMES for the same features across images, but the tag LISTS may "
                           "differ."},
            ] + [{"type": "image_url",
                  "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                 for _, b64 in batch]},
        ],
    }
    r = requests.post(f"{args.base_url.rstrip('/')}/chat/completions",
                      json=payload, timeout=600)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def parse_batch_response(raw: str, names: list[str]) -> dict[str, str]:
    """Map model lines ('name: tags') back to image names. Best-effort; missing
    names are omitted so the caller can fall back to a single-image call."""
    results = {}
    for line in raw.splitlines():
        if ":" not in line:
            continue
        head, tags = line.split(":", 1)
        head = head.strip().strip("`*#").strip().strip('"')
        if not head or not tags.strip():
            continue
        for n in names:
            if head == n or head in n or n in head or head.replace(" ", "_") == n:
                results[n] = tags
                break
    return results


def process_batch(args, imgs: list[str], exact: set[str], prefixes: list[str],
                  suffixes: list[str], contains: list[str], header: list[str],
                  hint_set: set[str], model: str) -> dict[str, str]:
    """Tag a batch of images; returns {img: final_caption} for images the model named."""
    batch = [(img, image_to_b64(os.path.join(args.folder, img), args.max_size))
             for img in imgs]
    raw, last = None, None
    for i in range(3):
        try:
            raw = tag_batch(args, model, batch, exact, prefixes, suffixes, contains, hint_set)
            break
        except Exception as e:
            last = e
            if i < 2:
                logger.warning("batch attempt %d failed: %s; retrying", i + 1, e)
                time.sleep(3 * (i + 1))
    if raw is None:
        raise last
    excluded = set(header)
    if args.trigger:
        excluded.add(args.trigger)
    return {img: build_caption(args.trigger, header,
                               normalize_tags(tags, exact, prefixes, suffixes, contains,
                                              excluded, hint_set, args.max_tags))
            for img, tags in parse_batch_response(raw, imgs).items()}


def handle_caption(args, img: str, caption: str, done: int, total: int,
                   tagged: list, caption_counts: dict):
    tagged.append((img, caption))
    for t in parse_tag_list(caption):
        caption_counts[t] = caption_counts.get(t, 0) + 1
    print(f"{color(f'[{done}/{total}]', CYAN)} {img}  ({len(caption.split(','))} tags)")
    if args.dry_run:
        print(f"    {color(caption, DIM)}")
    else:
        txt = os.path.join(args.folder, os.path.splitext(img)[0] + ".txt")
        with open(txt, "w", encoding="utf-8") as f:
            f.write(caption)
    print()  # blank line between image blocks


def main(argv=None):
    args = parse_args(argv)

    if not os.path.isdir(args.folder):
        sys.exit(f"error: not a directory: {args.folder}")

    cfg, cfg_path = load_config(args.folder, args.config)
    if cfg_path:
        logger.info("loaded config: %s", cfg_path)
    resolve_args(args, cfg)

    if args.init_config:
        init_config(args.folder)
        return

    if args.save_config:
        save_config(args)
        return

    if args.audit:
        audit_folder(args.folder)
        return

    if args.review:
        serve_app(args.folder, args.port, "review", not args.no_browser)
        return

    if args.setup:
        serve_app(args.folder, args.port, "setup", not args.no_browser)
        return

    try:
        models = requests.get(f"{args.base_url.rstrip('/')}/models", timeout=10).json()["data"]
    except Exception as e:
        sys.exit(f"error: cannot reach LM Studio at {args.base_url}\n  {e}")

    model = pick_model(models, args.model)
    logger.info("using model: %s (subject=%s)", model, args.subject)

    images = sorted(
        (f for f in os.listdir(args.folder)
         if os.path.splitext(f)[1].lower() in IMAGE_EXTS
         and os.path.isfile(os.path.join(args.folder, f))),
        key=natural_key,
    )
    if not images:
        sys.exit(f"no images ({', '.join(sorted(IMAGE_EXTS))}) found in {args.folder}")

    blacklist_exact, blacklist_prefixes, blacklist_suffixes, blacklist_contains = \
        split_blacklist(args.blacklist)
    header = parse_tag_list(args.character)
    hint = set(parse_tag_list(args.hint))

    jobs, skipped = [], 0
    existing_caption = set()
    for img in images:
        stem = os.path.splitext(img)[0]
        txt = os.path.join(args.folder, stem + ".txt")
        has_caption = os.path.exists(txt) and open(txt, encoding="utf-8").read().strip()
        if has_caption:
            existing_caption.add(img)
            if not args.force:
                skipped += 1
                continue
        jobs.append(img)

    if args.limit:
        jobs = jobs[:args.limit]

    total = len(jobs)
    if total == 0:
        print(f"nothing to tag ({skipped} already captioned)")
        return

    overwritten = sum(1 for img in jobs if img in existing_caption)
    if args.force and overwritten and not args.dry_run:
        print(f"{color('WARNING:', YELLOW)} --force will re-tag {total} image(s), "
              f"overwriting {overwritten} existing caption(s) "
              f"(hand-edited captions included).")
        try:
            answer = input("continue? [y/N] ").strip().lower()
        except EOFError:
            answer = ""
        if answer not in ("y", "yes"):
            print("aborted - no files touched")
            return


    print(f"tagging {total} image(s) with {model}"
          + ("  [dry-run: nothing will be written]" if args.dry_run else ""))
    tagged, failed = [], []
    done = 0
    caption_counts: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        if args.batch_size > 1:
            batches = [jobs[i:i + args.batch_size]
                       for i in range(0, len(jobs), args.batch_size)]
            futures = {ex.submit(process_batch, args, b, blacklist_exact, blacklist_prefixes,
                                 blacklist_suffixes, blacklist_contains, header, hint,
                                 model): b for b in batches}
            for fut in as_completed(futures):
                batch = futures[fut]
                try:
                    results = fut.result()
                except Exception as e:
                    for img in batch:
                        done += 1
                        failed.append((img, str(e)))
                        logger.error("failed %s: %s", img, e)
                        print(f"{color(f'[{done}/{total}]', CYAN)} {img}  "
                              f"{color('FAILED:', RED)} {e}")
                        print()
                    continue
                for img in batch:
                    done += 1
                    if img in results:
                        handle_caption(args, img, results[img], done, total, tagged,
                                       caption_counts)
                    else:
                        # model skipped this image in the batch -> fall back to single call
                        try:
                            caption = process_one(args, img, blacklist_exact,
                                                  blacklist_prefixes, blacklist_suffixes,
                                                  blacklist_contains, header, hint, model)
                        except Exception as e:
                            failed.append((img, str(e)))
                            logger.error("failed %s: %s", img, e)
                            print(f"{color(f'[{done}/{total}]', CYAN)} {img}  "
                                  f"{color('FAILED:', RED)} {e}")
                            print()
                            continue
                        handle_caption(args, img, caption, done, total, tagged, caption_counts)
        else:
            futures = {ex.submit(process_one, args, img, blacklist_exact, blacklist_prefixes,
                                 blacklist_suffixes, blacklist_contains, header, hint,
                                 model): img for img in jobs}
            for fut in as_completed(futures):
                img = futures[fut]
                done += 1
                try:
                    caption = fut.result()
                except Exception as e:
                    failed.append((img, str(e)))
                    logger.error("failed %s: %s", img, e)
                    print(f"{color(f'[{done}/{total}]', CYAN)} {img}  "
                          f"{color('FAILED:', RED)} {e}")
                    print()
                    continue
                handle_caption(args, img, caption, done, total, tagged, caption_counts)

    print()
    print(f"tagged: {len(tagged)}  skipped (already captioned): {skipped}  failed: {len(failed)}")
    if args.dry_run:
        print("dry-run: no files written")
    if args.report or args.dry_run:
        print_tag_report(caption_counts, len(tagged))
    if failed:
        for img, err in failed:
            logger.error("%s: %s", img, err)
        sys.exit(1)


if __name__ == "__main__":
    main()
