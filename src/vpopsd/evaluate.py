"""Offline point-in-box accuracy and student-only model inference."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from vpopsd.coordinates import parse_point
from vpopsd.data import load_records, write_jsonl


def score(records, predictions):
    by_id = {}
    for pred in predictions:
        key = str(pred["id"])
        if key in by_id:
            raise ValueError(f"Duplicate prediction: {key}")
        by_id[key] = pred
    expected = {r["id"] for r in records}
    if set(by_id) - expected:
        raise ValueError("Predictions contain unknown IDs")
    groups, details = defaultdict(lambda: {"correct": 0, "total": 0, "invalid": 0, "missing": 0}), []
    for row in records:
        pred = by_id.get(row["id"])
        point = parse_point(pred.get("text", "")) if pred else None
        x1, y1, x2, y2 = row["bbox"]
        correct = point is not None and x1 <= point[0] <= x2 and y1 <= point[1] <= y2
        group = groups[row.get("benchmark", "default")]
        group["total"] += 1
        group["correct"] += int(correct)
        group["invalid"] += int(point is None)
        group["missing"] += int(pred is None)
        details.append({"id": row["id"], "correct": correct, "point": point})
    if not groups:
        raise ValueError("No evaluation examples")
    for group in groups.values():
        group["accuracy"] = group["correct"] / group["total"]
    return {
        "benchmarks": dict(groups),
        "macro_accuracy": sum(g["accuracy"] for g in groups.values()) / len(groups),
        "micro_accuracy": sum(g["correct"] for g in groups.values()) / len(records),
        "total": len(records),
    }, details


def predict(records, cfg):
    import torch
    from PIL import Image

    from vpopsd.engine import encode_prompt
    from vpopsd.privilege import prompt
    from vpopsd.train import load_model

    model, processor = load_model(cfg, inference=True)
    model.eval()
    device = next(model.parameters()).device
    with torch.inference_mode():
        for row in records:
            with Image.open(row["image"]) as source:
                image = source.convert("RGB")
            inputs = encode_prompt(processor, image, prompt(row["instruction"]), device)
            output = model.generate(**inputs, do_sample=False, max_new_tokens=cfg.get("max_new_tokens", 32))
            text = processor.tokenizer.decode(
                output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )
            yield {"id": row["id"], "text": text}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--predictions", help="Existing JSONL id/text predictions")
    mode.add_argument("--config", help="Inference YAML; only original screenshots are used")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    records = load_records(args.data, require_images=bool(args.config))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if args.config:
        import yaml

        cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        predictions = list(predict(records, cfg))
        write_jsonl(output / "predictions.jsonl", predictions)
    else:
        predictions = [
            json.loads(s)
            for s in Path(args.predictions).read_text(encoding="utf-8").splitlines()
            if s.strip()
        ]
    summary, details = score(records, predictions)
    (output / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_jsonl(output / "details.jsonl", details)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
