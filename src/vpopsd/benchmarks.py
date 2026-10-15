"""Export ScreenSpot parquet or convert downloaded ScreenSpot-v2/Pro annotations."""

import argparse
import json
from pathlib import Path

from PIL import Image

from vpopsd.data import write_jsonl
from vpopsd.prepare import normalize_box


def convert_annotations(paths, image_root, benchmark):
    root = Path(image_root).resolve()
    rows = []
    for path in paths:
        records = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise ValueError(f"Expected annotation list: {path}")
        for row in records:
            if benchmark == "osworld-g-bbox" and row.get("box_type") != "bbox":
                continue  # Explicit bbox-only subset, never silently call it the full benchmark.
            filename = (
                row["image_path"]
                if benchmark in ("osworld-g-bbox", "ui-vision-element")
                else row["img_filename"]
            )
            image = (root / filename).resolve()
            if not image.is_file():
                # Some Pro archives organize screenshots by application subdirectory.
                candidates = list(root.rglob(Path(filename).name))
                if len(candidates) != 1:
                    raise FileNotFoundError(f"Cannot uniquely resolve {filename}")
                image = candidates[0]
            with Image.open(image) as im:
                w, h = im.size
            units = "pixels"
            fmt = "xywh" if benchmark in ("screenspot-v2", "osworld-g-bbox") else "xyxy"
            bbox = row["box_coordinates"] if benchmark == "osworld-g-bbox" else row["bbox"]
            instruction = (
                row["prompt_to_evaluate"] if benchmark == "ui-vision-element" else row["instruction"]
            )
            rows.append(
                {
                    "id": f"{benchmark}:{len(rows)}",
                    "image": str(image),
                    "instruction": instruction,
                    "benchmark": benchmark,
                    "source_annotation": Path(path).name,
                    "bbox": normalize_box(bbox, units, fmt, w, h),
                }
            )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        choices=[
            "screenspot",
            "screenspot-v2",
            "screenspot-pro",
            "groundui-1k",
            "osworld-g-bbox",
            "ui-vision-element",
        ],
        required=True,
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--annotations", nargs="+", help="Local annotation JSON files (v2/Pro)")
    parser.add_argument("--image-root", help="Extracted screenshots directory (v2/Pro)")
    parser.add_argument("--revision", default="main", help="Hugging Face revision for ScreenSpot export")
    args = parser.parse_args()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.benchmark in ("screenspot", "groundui-1k"):
        from datasets import load_dataset

        repo = "bevaya/ScreenSpot" if args.benchmark == "screenspot" else "agent-studio/GroundUI-1K"
        split = "test" if args.benchmark == "screenspot" else "train"
        dataset = load_dataset(repo, split=split, revision=args.revision)
        images = out / "images"
        images.mkdir(exist_ok=True)
        rows = []
        for i, row in enumerate(dataset):
            image = images / f"{i:06d}.png"
            screenshot = row["image"].convert("RGB")
            screenshot.save(image)
            units = "normalized" if args.benchmark == "screenspot" else "pixels"
            rows.append(
                {
                    "id": f"{args.benchmark}:{i}",
                    "image": str(image),
                    "instruction": row["instruction"],
                    "bbox": normalize_box(row["bbox"], units, "xyxy", *screenshot.size),
                    "benchmark": args.benchmark,
                }
            )
        (out / "source.json").write_text(
            json.dumps({"dataset": repo, "revision": args.revision, "fingerprint": dataset._fingerprint}),
            encoding="utf-8",
        )
    else:
        if not args.annotations or not args.image_root:
            parser.error("Local annotation conversion requires --annotations and --image-root")
        rows = convert_annotations(args.annotations, args.image_root, args.benchmark)
    write_jsonl(out / "test.jsonl", rows)
    print(f"Exported {len(rows)} rows to {out / 'test.jsonl'}")


if __name__ == "__main__":
    main()
