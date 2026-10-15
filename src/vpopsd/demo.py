"""Generate a synthetic GUI fixture and teacher view locally; no dataset download."""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from vpopsd.data import write_jsonl
from vpopsd.privilege import privileged_view


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/demo")
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (800, 500), "#eef2f6")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 800, 65), fill="#17243a")
    draw.text((30, 25), "SYNTHETIC GUI - Account settings", fill="white")
    draw.rounded_rectangle((100, 120, 700, 400), radius=20, fill="white")
    draw.text((140, 160), "Notifications: enabled", fill="#17243a")
    draw.rounded_rectangle((520, 310, 660, 365), radius=8, fill="#2563eb")
    draw.text((565, 330), "Save", fill="white")
    image.save(out / "student.png")
    box = [520 / 800, 310 / 500, 660 / 800, 365 / 500]
    privileged_view(image, box).save(out / "teacher.png")
    write_jsonl(
        out / "examples.jsonl",
        [
            {
                "id": "synthetic-0",
                "image": "student.png",
                "instruction": "Click Save",
                "bbox": box,
                "benchmark": "synthetic-smoke-only",
            }
        ],
    )
    write_jsonl(out / "predictions.jsonl", [{"id": "synthetic-0", "text": "(737, 675)"}])
    print(f"Synthetic demo written to {out.resolve()}")


if __name__ == "__main__":
    main()
