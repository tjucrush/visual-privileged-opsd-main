"""Run all configured datasets with one model load and require complete results."""

import argparse
import json
from pathlib import Path

import yaml

from vpopsd.data import load_records, write_jsonl
from vpopsd.evaluate import predict, score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    suite = yaml.safe_load(Path(args.suite).read_text(encoding="utf-8"))
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    records = []
    for name, path in suite["datasets"].items():
        for row in load_records(path):
            records.append({**row, "id": f"{name}/{row['id']}", "benchmark": name})
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Use an empty output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    predictions = []
    with (out / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for pred in predict(records, cfg):
            predictions.append(pred)
            stream.write(json.dumps(pred, ensure_ascii=False) + "\n")
            stream.flush()
    summary, details = score(records, predictions)
    summary["protocol"] = "unified single-point-in-box"
    summary["config"] = cfg
    summary["suite"] = suite
    (out / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_jsonl(out / "details.jsonl", details)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
