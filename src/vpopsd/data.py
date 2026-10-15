"""Explicit normalized xyxy schema; no coordinate-system guessing."""

import json
import math
from pathlib import Path


def validate_box(box):
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise ValueError("bbox must contain four normalized xyxy coordinates")
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in box):
        raise ValueError("bbox coordinates must be finite numbers")
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise ValueError(f"Invalid normalized xyxy bbox: {box}")
    return tuple(float(v) for v in box)


def load_records(path, require_images=True):
    path = Path(path).resolve()
    records, seen = [], set()
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            for key in ("id", "image", "instruction", "bbox"):
                if key not in row:
                    raise ValueError(f"{path}:{line_no}: missing {key}")
            row["id"] = str(row["id"])
            if row["id"] in seen:
                raise ValueError(f"Duplicate id: {row['id']}")
            seen.add(row["id"])
            if not isinstance(row["instruction"], str) or not row["instruction"].strip():
                raise ValueError("instruction must be nonempty text")
            row["bbox"] = validate_box(row["bbox"])
            image = Path(row["image"])
            row["image"] = str(image if image.is_absolute() else path.parent / image)
            if require_images and not Path(row["image"]).is_file():
                raise FileNotFoundError(row["image"])
            records.append(row)
    if not records:
        raise ValueError(f"Empty dataset: {path}")
    return records


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
