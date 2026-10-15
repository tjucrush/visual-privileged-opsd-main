# Implementation Notes

This document records the image transformations, training flow, and objective calculation implemented by the package.

## Image Cue Construction

For a normalized target bounding box, snap each lower boundary down and each upper boundary up to a 12×12 grid. Expand by one cell and clip to the image. The red rectangle and Gaussian both depend only on this coarse region. For center `(cx,cy)` and spreads `(sx,sy)` equal to 0.65 times the region dimensions:

`G(x,y) = exp(-0.5 * (((x-cx)/sx)^2 + ((y-cy)/sy)^2))`

`teacher_image = original_image * (0.35 + 0.65 * G)`

Draw the coarse red rectangle afterward. No exact target center, numeric target coordinate, ground-truth answer string or example-specific coordinate hint is placed in the prompt. The instruction itself is passed through; upstream datasets must be audited if their instructions already contain answer coordinates. The teacher still receives intended approximate localization privilege. Very large targets can yield a full-image cue.

## Training Flow

For each example, sample `y ~ p_theta(. | original_image, instruction)` with temperature 1, no top-k or top-p truncation by default. Under no-grad, compute the teacher distribution on `y` with the privileged view. Compute the differentiable student distribution on the same IDs with the original view. Each predicted Token `y[t]` uses only `y[:t]` as its causal prefix. Prompt lengths may differ and are aligned independently.

The teacher shares the current student parameters, including LoRA adapters, and is recomputed for each sample. It is not a frozen initial model or an EMA teacher. Gradients flow only through the student forward. Multiple samples within one optimizer accumulation group use unchanged current parameters. Malformed outputs are retained with baseline positional weights, rather than selecting only successful trajectories.

## Objective Calculation

With temperature `T`, `p = softmax(z_student/T)`, `q = stopgrad(softmax(z_teacher/T))`:

`KL_t = sum_v p[t,v] * (log p[t,v] - log q[t,v])`

`H_t = -sum_v q[t,v] * log q[t,v]`

`R_t = max(0.05, 1 - H_t / log(V))`

Coordinate output uses integers on a 0–1000 scale, in `(x, y)` syntax. A digit with decimal place value `10^k` gets character importance `1 + alpha * 10^k / 1000`, with default `alpha=4`. A Token spanning multiple characters takes their maximum importance; punctuation, EOS and malformed answers have importance 1. Actual generated Token IDs are decoded incrementally, never replaced by a potentially different re-tokenization. For non-prefix-stable decoding, use baseline weights. Full merged coordinate Tokens cannot distinguish their constituent digits; this is an intrinsic limitation of Token-level weighting.

`w_t = stopgrad(I_t * R_t)`

`L = T^2 * sum_t mask_t * w_t * KL_t / sum_t mask_t`

Normalization is by valid Token count, not by the sum of weights, so lowering reliability also lowers absolute supervision. The trainer handles one unpadded completion at a time; EOS participates, prompt positions do not. FP32 log-softmax and reductions avoid BF16 underflow in the KL calculation. No ground-truth next-Token cross entropy is mixed into this objective.

Entropy is a heuristic for confidence, not a correctness guarantee. A confidently wrong teacher can still receive a high weight. With a very large vocabulary the normalized-entropy heuristic may have limited dynamic range; its floor and usefulness should be ablated.

## Default Configuration

The default training configuration uses LoRA rank 16, AdamW, one epoch, and gradient accumulation of 8. Configure image resolution, sampling temperature, positional weighting, entropy weighting, and visual cues in `configs/qwen3_vl_8b.yaml`. Full-parameter fine-tuning is available with `lora: false`.

