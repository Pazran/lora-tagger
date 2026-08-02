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
from PIL import Image

from logger import setup_logger

logger = setup_logger("lora-tagger", level="INFO", log_file="tagger.log")

DEFAULT_BASE_URL = "http://localhost:1234/v1"
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


def split_blacklist(extra: str) -> tuple[set[str], list[str]]:
    """Split a tag list into (exact set, wildcard prefixes).

    Entries ending in '*' become prefix rules: 'fantasy_*' strips any tag starting
    with 'fantasy_'. Merged with the built-in defaults.
    """
    exact = set(DEFAULT_BLACKLIST)
    prefixes = list(DEFAULT_BLACKLIST_PREFIXES)
    for t in parse_tag_list(extra):
        if t.endswith("*"):
            prefixes.append(t[:-1])
        else:
            exact.add(t)
    return exact, prefixes


def is_blacklisted(tag: str, exact: set[str], prefixes: list[str]) -> bool:
    if tag in exact:
        return True
    return any(tag.startswith(p) for p in prefixes)


def render_banned(exact: set[str], prefixes: list[str]) -> str:
    parts = sorted(exact) + [p + "*" for p in prefixes]
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
               "  tagger . --hint \"face_paint\" --blacklist \"fantasy_*, ornate\"",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("folder", nargs="?", default=".",
                   help="folder to scan (default: current folder)")
    p.add_argument("--subject", choices=("character", "outfit", "style"), default="character",
                   help="what the model should focus on (default: character)")
    p.add_argument("--trigger", default="",
                   help="token prepended to every caption, e.g. mychar")
    p.add_argument("--character", default="",
                   help="invariant header tags true for EVERY image (e.g. \"1girl, white_hair\"); "
                        "prepended to all captions and excluded from model output")
    p.add_argument("--hint", default="",
                   help="canonical vocabulary: tags to use exactly when the feature is visible, "
                        "e.g. \"face_paint\" (never synonyms)")
    p.add_argument("--blacklist", default="",
                   help="extra tags to strip, comma-separated; 'foo_*' strips a whole family "
                        "(merged with defaults)")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL,
                   help=f"LM Studio API base URL (default: {DEFAULT_BASE_URL})")
    p.add_argument("--model", default="",
                   help="model id to use; auto-detects a qwen*-vl model if empty")
    p.add_argument("--temperature", type=float, default=0.3,
                   help="sampling temperature, low = reproducible (default: 0.3)")
    p.add_argument("--max-tags", type=int, default=40,
                   help="cap on tags per caption (default: 40)")
    p.add_argument("--max-size", type=int, default=1280,
                   help="downscale longest side before sending (default: 1280px)")
    p.add_argument("--workers", type=int, default=1,
                   help="parallel requests; LM Studio queues, keep low (default: 1)")
    p.add_argument("--limit", type=int, default=0,
                   help="process at most N images (default: all)")
    p.add_argument("--force", action="store_true",
                   help="re-tag even if the .txt caption is already non-empty")
    p.add_argument("--dry-run", action="store_true",
                   help="tag and print captions, write nothing")
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
              hint: list[str]) -> str:
    payload = {
        "model": model,
        "temperature": args.temperature,
        "max_tokens": 1024,
        "messages": [
            {"role": "system",
             "content": system_prompt(args.subject, render_banned(exact, prefixes),
                                      ", ".join(hint) or "(none)", args.max_tags)},
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
                    hint: list[str], attempts: int = 3) -> str:
    last = None
    for i in range(attempts):
        try:
            return tag_image(args, model, b64, exact, prefixes, hint)
        except Exception as e:
            last = e
            if i < attempts - 1:
                logger.warning("attempt %d failed: %s; retrying", i + 1, e)
                time.sleep(3 * (i + 1))
    raise last


def normalize_tags(raw: str, exact: set[str], prefixes: list[str], excluded: set[str],
                   max_tags: int) -> list[str]:
    out, seen = [], set()
    for chunk in re.split(r"[,;\n]", raw):
        t = chunk.strip().lower().replace(" ", "_")
        t = re.sub(r"[^a-z0-9_]", "", t)
        if not t or is_blacklisted(t, exact, prefixes) or t in excluded or t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out[:max_tags]


def build_caption(trigger: str, header: list[str], tags: list[str]) -> str:
    parts = ([trigger] if trigger else []) + header + tags
    return ", ".join(p for p in parts if p)


def process_one(args, img: str, exact: set[str], prefixes: list[str],
                header: list[str], hint: list[str], model: str) -> str:
    path = os.path.join(args.folder, img)
    b64 = image_to_b64(path, args.max_size)
    raw = call_with_retry(args, model, b64, exact, prefixes, hint)
    # The trigger and header are prepended by the script; never let the model
    # duplicate them (VL models sometimes echo the subject name back).
    excluded = set(header)
    if args.trigger:
        excluded.add(args.trigger)
    tags = normalize_tags(raw, exact, prefixes, excluded, args.max_tags)
    return build_caption(args.trigger, header, tags)


def main(argv=None):
    args = parse_args(argv)

    if not os.path.isdir(args.folder):
        sys.exit(f"error: not a directory: {args.folder}")

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

    blacklist_exact, blacklist_prefixes = split_blacklist(args.blacklist)
    header = parse_tag_list(args.character)
    hint = parse_tag_list(args.hint)

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
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process_one, args, img, blacklist_exact, blacklist_prefixes,
                             header, hint, model): img
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
            print(f"[{done}/{total}] {img}  ({len(caption.split(','))} tags)")
            if args.dry_run:
                print(f"    {caption}")
            else:
                txt = os.path.join(args.folder, os.path.splitext(img)[0] + ".txt")
                with open(txt, "w", encoding="utf-8") as f:
                    f.write(caption)

    print()
    print(f"tagged: {len(tagged)}  skipped (already captioned): {skipped}  failed: {len(failed)}")
    if args.dry_run:
        print("dry-run: no files written")
    if failed:
        for img, err in failed:
            logger.error("%s: %s", img, err)
        sys.exit(1)


if __name__ == "__main__":
    main()
