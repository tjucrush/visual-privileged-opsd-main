"""GPU training entry point: current-policy stop-gradient self-teacher."""

import argparse
import hashlib
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from vpopsd.data import load_records
from vpopsd.engine import distillation_step, encode_prompt
from vpopsd.privilege import PrivilegeConfig, privileged_view, prompt


def load_model(cfg, inference=False):
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    device = cfg.get("device", "cuda")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use a GPU host for 8B training")
    dtype = getattr(torch, cfg.get("dtype", "bfloat16"))
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        cfg["model_name"],
        torch_dtype=dtype,
        attn_implementation="sdpa",
    ).to(device)
    processor = AutoProcessor.from_pretrained(
        cfg["model_name"],
        min_pixels=cfg.get("min_pixels", 256 * 32 * 32),
        max_pixels=cfg.get("max_pixels", 1280 * 32 * 32),
    )
    if cfg.get("adapter"):
        if not inference:
            raise ValueError("adapter is inference-only; training starts from model_name")
        model = PeftModel.from_pretrained(model, cfg["adapter"])
    elif cfg.get("lora", True) and not inference:
        model = get_peft_model(
            model,
            LoraConfig(
                r=cfg.get("lora_rank", 16),
                lora_alpha=cfg.get("lora_alpha", 32),
                lora_dropout=0.0,
                task_type="CAUSAL_LM",
                target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj)$",
            ),
        )
    if not inference and cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    return model, processor


def run(cfg):
    seed = cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cfg.get("epochs", 1) < 1 or cfg.get("gradient_accumulation_steps", 8) < 1:
        raise ValueError("epochs and gradient_accumulation_steps must be positive")
    rows = load_records(cfg["train_data"])
    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"Use an empty output directory: {output}")
    privilege = PrivilegeConfig(**cfg.get("privilege", {}))
    import peft
    import transformers

    metadata = {
        "config": cfg,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "dataset_sha256": hashlib.sha256(Path(cfg["train_data"]).read_bytes()).hexdigest(),
        "samples": len(rows),
        "teacher": "current-policy-stop-gradient",
    }
    (output / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    model, processor = load_model(cfg)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=cfg.get("learning_rate", 1e-5),
        weight_decay=0.01,
    )
    accumulation = cfg.get("gradient_accumulation_steps", 8)
    device = next(model.parameters()).device
    started, updates, samples = time.perf_counter(), 0, 0
    with (output / "metrics.jsonl").open("w", encoding="utf-8") as log:
        for epoch in range(cfg.get("epochs", 1)):
            order = list(range(len(rows)))
            random.Random(seed + epoch).shuffle(order)
            for group_start in range(0, len(order), accumulation):
                group = order[group_start : group_start + accumulation]
                optimizer.zero_grad(set_to_none=True)
                for idx in group:
                    row = rows[idx]
                    with Image.open(row["image"]) as source:
                        image = source.convert("RGB")
                    teacher_image = privileged_view(image, row["bbox"], privilege)
                    student_inputs = encode_prompt(processor, image, prompt(row["instruction"]), device)
                    teacher_inputs = encode_prompt(
                        processor,
                        teacher_image,
                        prompt(row["instruction"], teacher=True, use_hint=cfg.get("teacher_hint", True)),
                        device,
                    )
                    loss, metrics = distillation_step(model, processor, student_inputs, teacher_inputs, cfg)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Nonfinite loss for {row['id']}")
                    (loss / len(group)).backward()
                    samples += 1
                    metrics.update(
                        epoch=epoch + 1,
                        sample_id=row["id"],
                        samples=samples,
                        optimizer_step=updates + 1,
                        elapsed_seconds=time.perf_counter() - started,
                    )
                    log.write(json.dumps(metrics) + "\n")
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("max_grad_norm", 1.0))
                optimizer.step()
                updates += 1
                log.flush()
                print(json.dumps({"step": updates, **metrics}), flush=True)
                if cfg.get("max_steps") and updates >= cfg["max_steps"]:
                    break
            if cfg.get("max_steps") and updates >= cfg["max_steps"]:
                break
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    training_seconds = time.perf_counter() - started
    model.save_pretrained(output / "model")
    processor.save_pretrained(output / "model")
    (output / "summary.json").write_text(
        json.dumps(
            {
                "samples": samples,
                "optimizer_steps": updates,
                "training_seconds": training_seconds,
                "training_hours": training_seconds / 3600,
                "timing_scope": "rollout + teacher + student + optimizer; excludes model load/save",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    run(cfg)


if __name__ == "__main__":
    main()
