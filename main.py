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
               "  tagger . --review                        # human review grid in the browser",
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


def make_review_handler(folder: str):
    """HTTP handler factory for the review grid (thumbnails, data, saves)."""
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
            if urllib.parse.urlparse(self.path).path != "/api/save":
                return self.send_error(404)
            length = int(self.headers.get("Content-Length", 0) or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except Exception:
                return self._json({"ok": False, "error": "bad json"}, 400)
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

        def log_message(self, fmt, *args):
            pass

    return ReviewHandler


def review_folder(folder: str, port: int, open_browser: bool):
    """Serve the human review grid until Ctrl+C."""
    if not any(os.path.splitext(f)[1].lower() in IMAGE_EXTS
               for f in os.listdir(folder)):
        sys.exit(f"no images ({', '.join(sorted(IMAGE_EXTS))}) found in {folder}")
    url = f"http://127.0.0.1:{port}"
    print(f"review grid: {url}  (Ctrl+C to stop)")
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        ThreadingHTTPServer(("127.0.0.1", port),
                            make_review_handler(folder)).serve_forever()
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
        review_folder(args.folder, args.port, not args.no_browser)
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
