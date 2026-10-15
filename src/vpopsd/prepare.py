"""Convert explicit JSON/JSONL fields; sample without image overlap across splits."""

import argparse
import hashlib
import json
import random
from pathlib import Path

from PIL import Image

from vpopsd.data import load_records, validate_box, write_jsonl


def field(row, dotted):
    for key in dotted.split("."):
        row = row[key]
    return row


def normalize_box(box, units, fmt, width, height):
    x1, y1, a, b = map(float, box)
    x2, y2 = (x1 + a, y1 + b) if fmt == "xywh" else (a, b)
    if units == "pixels":
        x1, x2, y1, y2 = x1 / width, x2 / width, y1 / height, y2 / height
    elif units == "1000":
        x1, x2, y1, y2 = x1 / 1000, x2 / 1000, y1 / 1000, y2 / 1000
    return validate_box([x1, y1, x2, y2])


def image_digest(path):
    # Hash decoded pixels, catching duplicates with different compression/metadata.
    with Image.open(path) as source:
        image = source.convert("RGB")
    return hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()


def split_records(rows, train_size, val_size, seed, excluded=()):
    if train_size < 1 or val_size < 0:
        raise ValueError("Invalid split sizes")
    excluded = set(excluded)
    groups, cache = {}, {}
    for row in rows:
        if row["image"] not in cache:
            cache[row["image"]] = image_digest(row["image"])
        digest = cache[row["image"]]
        if digest not in excluded:
            groups.setdefault(digest, []).append(row)
    keys = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(keys)
    val, train_pool = [], []
    for key in keys:
        group = groups[key].copy()
        rng.shuffle(group)
        if len(val) < val_size:
            # Reserve the entire image for validation, even if only some rows are used.
            val.extend(group[: val_size - len(val)])
        else:
            train_pool.extend(group)
    rng.shuffle(train_pool)
    if len(val) < val_size or len(train_pool) < train_size:
        raise ValueError("Insufficient examples after image disjoint splitting / benchmark exclusions")
    return train_pool[:train_size], val


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    convert = sub.add_parser("convert")
    convert.add_argument("--input", required=True)
    convert.add_argument("--image-root", required=True)
    convert.add_argument("--output", required=True)
    convert.add_argument("--image-field", default="image")
    convert.add_argument("--instruction-field", default="instruction")
    convert.add_argument("--bbox-field", default="bbox")
    convert.add_argument("--bbox-format", choices=["xyxy", "xywh"], required=True)
    convert.add_argument("--bbox-units", choices=["normalized", "pixels", "1000"], required=True)
    convert.add_argument("--benchmark", default="scalecua")
    split = sub.add_parser("split")
    split.add_argument("--input", required=True)
    split.add_argument("--output-dir", required=True)
    split.add_argument("--train-size", type=int, default=7000)
    split.add_argument("--val-size", type=int, default=300)
    split.add_argument("--seed", type=int, default=42)
    split.add_argument("--exclude", nargs="*", default=[], help="Canonical benchmark JSONL manifests")
    args = parser.parse_args()
    if args.command == "convert":
        raw = Path(args.input).read_text(encoding="utf-8")
        source = (
            json.loads(raw)
            if raw.lstrip().startswith("[")
            else [json.loads(s) for s in raw.splitlines() if s.strip()]
        )
        result = []
        for i, row in enumerate(source):
            image = (Path(args.image_root) / field(row, args.image_field)).resolve()
            with Image.open(image) as im:
                w, h = im.size
            instruction = field(row, args.instruction_field)
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError(f"Invalid instruction at row {i}")
            result.append(
                {
                    "id": f"{args.benchmark}:{i}",
                    "image": str(image),
                    "instruction": instruction,
                    "benchmark": args.benchmark,
                    "bbox": normalize_box(
                        field(row, args.bbox_field), args.bbox_units, args.bbox_format, w, h
                    ),
                }
            )
        write_jsonl(args.output, result)
        print(f"Converted {len(result)} rows to {args.output}")
    else:
        rows = load_records(args.input)
        excluded = {image_digest(r["image"]) for path in args.exclude for r in load_records(path)}
        train, val = split_records(rows, args.train_size, args.val_size, args.seed, excluded)
        out = Path(args.output_dir)
        write_jsonl(out / "train.jsonl", train)
        write_jsonl(out / "validation.jsonl", val)
        (out / "split.json").write_text(
            json.dumps(
                {
                    "seed": args.seed,
                    "train": len(train),
                    "validation": len(val),
                    "excluded_images": len(excluded),
                    "source_sha256": hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
