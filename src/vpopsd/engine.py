"""Single-example multimodal rollout and causal completion alignment."""

import torch

from vpopsd.coordinates import parse_point, token_importance
from vpopsd.loss import weighted_reverse_kl


def encode_prompt(processor, image, text, device):
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
    rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    batch = processor(text=[rendered], images=[image], return_tensors="pt")
    batch.pop("token_type_ids", None)
    return batch.to(device)


def completion_inputs(prompt_inputs, completion):
    if completion.ndim != 2 or completion.shape[0] != 1 or completion.shape[1] == 0:
        raise ValueError("Expected one nonempty completion")
    result = dict(prompt_inputs)
    result["input_ids"] = torch.cat([prompt_inputs["input_ids"], completion], dim=1)
    result["attention_mask"] = torch.cat(
        [prompt_inputs["attention_mask"], torch.ones_like(completion)], dim=1
    )
    # Regenerate multimodal position IDs for each prompt; never reuse a rollout cache.
    result.pop("position_ids", None)
    result.pop("past_key_values", None)
    return result


def completion_logits(model, prompt_inputs, completion):
    p = prompt_inputs["input_ids"].shape[1]
    indices = torch.arange(p - 1, p - 1 + completion.shape[1], device=completion.device)
    result = model(**completion_inputs(prompt_inputs, completion), logits_to_keep=indices, use_cache=False)
    return result.logits


def distillation_step(model, processor, student_inputs, teacher_inputs, cfg):
    model.eval()
    with torch.no_grad():
        output = model.generate(
            **student_inputs,
            do_sample=True,
            temperature=cfg.get("rollout_temperature", 1.0),
            top_k=0,
            top_p=1.0,
            max_new_tokens=cfg.get("max_new_tokens", 32),
            pad_token_id=processor.tokenizer.pad_token_id,
            use_cache=True,
        )
        completion = output[:, student_inputs["input_ids"].shape[1] :]
        teacher_logits = completion_logits(model, teacher_inputs, completion).detach()
    # Both distributions are conditioned on exactly the same student-generated IDs.
    model.train()
    student_logits = completion_logits(model, student_inputs, completion)
    ids = completion[0].tolist()
    weights = token_importance(processor.tokenizer, ids, digit_scale=cfg.get("digit_scale", 4.0))
    if not cfg.get("position_weighting", True):
        weights = [1.0] * len(ids)
    importance = torch.tensor([weights], device=completion.device)
    loss, metrics = weighted_reverse_kl(
        student_logits,
        teacher_logits,
        importance,
        temperature=cfg.get("distillation_temperature", 1.0),
        reliability_floor=cfg.get("reliability_floor", 0.05),
        entropy_weighting=cfg.get("entropy_weighting", True),
    )
    decoded = processor.tokenizer.decode(ids, skip_special_tokens=True)
    metrics.update(
        loss=loss.detach().item(),
        completion_tokens=len(ids),
        valid_coordinate=int(parse_point(decoded) is not None),
    )
    return loss, metrics
