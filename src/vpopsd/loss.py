"""Full-vocabulary reverse KL with detached teacher entropy weighting."""

import math

import torch
import torch.nn.functional as F


def weighted_reverse_kl(
    student_logits,
    teacher_logits,
    importance,
    mask=None,
    temperature=1.0,
    reliability_floor=0.05,
    entropy_weighting=True,
):
    if student_logits.shape != teacher_logits.shape or student_logits.ndim != 3:
        raise ValueError("Expected matching [batch, completion, vocabulary] logits")
    if temperature <= 0 or not 0 < reliability_floor <= 1:
        raise ValueError("Invalid temperature or reliability_floor")
    if student_logits.shape[-1] < 2:
        raise ValueError("Vocabulary must contain at least two tokens")
    if importance.shape != student_logits.shape[:2] or (importance <= 0).any():
        raise ValueError("importance must be positive [batch, completion]")
    student_logp = F.log_softmax(student_logits.float() / temperature, dim=-1)
    with torch.no_grad():
        teacher_logp = F.log_softmax(teacher_logits.detach().float() / temperature, dim=-1)
        entropy = -(teacher_logp.exp() * teacher_logp).sum(-1)
        reliability = (1 - entropy / math.log(teacher_logits.shape[-1])).clamp(reliability_floor, 1)
        if not entropy_weighting:
            reliability = torch.ones_like(reliability)
        valid = torch.ones_like(importance, dtype=torch.bool) if mask is None else mask.bool()
        if valid.shape != importance.shape or not valid.any():
            raise ValueError("mask must match importance and contain valid tokens")
        weights = importance.detach() * reliability * valid
    kl = (student_logp.exp() * (student_logp - teacher_logp)).sum(-1)
    # Divide by valid count: reliability remains an absolute supervision strength.
    loss = temperature**2 * (weights * kl).sum() / valid.sum()
    return loss, {
        "reverse_kl": kl.detach()[valid].mean().item(),
        "teacher_entropy": entropy[valid].mean().item(),
        "mean_weight": weights[valid].mean().item(),
    }
