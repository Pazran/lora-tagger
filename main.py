#!/usr/bin/env python3
"""lora-tagger — auto-tag dataset images for LoRA training via LM Studio.

Sends each image in a folder to a local vision model (Qwen3-VL by default),
normalizes the response into Danbooru tags, and writes <stem>.txt captions
next to each image — the standard kohya/ai-toolkit dataset layout.
"""

import argparse
import base64
import io
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import tomllib
from PIL import Image

from logger import setup_logger

logger = setup_logger("lora-tagger", level="INFO", log_file="tagger.log")

DEFAULT_BASE_URL = "http://localhost:1234/v1"

CYAN = "\033[36m"
RED = "\033[31m"
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
               "  tagger . --save-config                   # persist settings to tagger.toml",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("folder", nargs="?", default=".",
                   help="folder to scan (default: current folder)")
    p.add_argument("--config", default=None,
                   help="path to a tagger.toml config file; if omitted, tagger.toml "
                        "in the target folder is auto-loaded")
    p.add_argument("--save-config", action="store_true",
                   help="write effective settings to tagger.toml in the folder and exit")
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
    p.add_argument("--limit", type=int, default=0,
                   help="process at most N images (default: all)")
    p.add_argument("--force", action="store_true",
                   help="re-tag even if the .txt caption is already non-empty")
    p.add_argument("--dry-run", action="store_true",
                   help="tag and print captions, write nothing")
    p.add_argument("--report", action="store_true",
                   help="print tag frequency report after the run "
                        "(auto-enabled in --dry-run; use for real runs to audit color/tag spread)")
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


def main(argv=None):
    args = parse_args(argv)

    if not os.path.isdir(args.folder):
        sys.exit(f"error: not a directory: {args.folder}")

    cfg, cfg_path = load_config(args.folder, args.config)
    if cfg_path:
        logger.info("loaded config: %s", cfg_path)
    resolve_args(args, cfg)

    if args.save_config:
        save_config(args)
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
    for img in images:
        stem = os.path.splitext(img)[0]
        txt = os.path.join(args.folder, stem + ".txt")
        if (os.path.exists(txt) and open(txt, encoding="utf-8").read().strip()
                and not args.force):
            skipped += 1
            continue
        jobs.append(img)

    if args.limit:
        jobs = jobs[:args.limit]

    total = len(jobs)
    if total == 0:
        print(f"nothing to tag ({skipped} already captioned)")
        return

    print(f"tagging {total} image(s) with {model}"
          + ("  [dry-run: nothing will be written]" if args.dry_run else ""))
    tagged, failed = [], []
    done = 0
    caption_counts: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process_one, args, img, blacklist_exact, blacklist_prefixes,
                             blacklist_suffixes, blacklist_contains, header, hint, model): img
                   for img in jobs}
        for fut in as_completed(futures):
            img = futures[fut]
            done += 1
            try:
                caption = fut.result()
            except Exception as e:
                failed.append((img, str(e)))
                logger.error("failed %s: %s", img, e)
                print(f"[{done}/{total}] {img}  FAILED: {e}")
                continue
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
