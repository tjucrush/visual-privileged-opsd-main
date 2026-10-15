# Validation record

Date: 2026-10-15. Local platform: Windows, Python 3.12.14. Validation ran on CPU.

| Check | Observed result |
|---|---|
| `pytest -q` | **24 passed**, 37.80 seconds |
| `ruff check .` | All checks passed |
| `pip check` | No broken requirements |
| Synthetic GUI demo | Original and privileged images rendered and visually inspected |
| Offline scorer | One known synthetic point inside its box, 1/1 correct; this is not a model score |
| Official Qwen3-VL-8B processor | Downloaded processor/tokenizer only; encoded a 128×96 RGB image and prompt on CPU |
| Processor smoke output | Input IDs shape `[1,32]`; image grid `[[1,6,8]]` |
| Actual Qwen tokenizer digit weights | `(123, 456)` produced per-digit weights 1.4, 1.04, 1.004 with default alpha=4 |

Tests exercise full-vocabulary reverse KL against a manual calculation, temperature scaling, teacher gradient isolation, masks, entropy confidence, absolute weight strength, merged digit Tokens, malformed outputs, coarse cue invariance, explicit coordinate conversions, image-disjoint splits, metric denominators and benchmark-specific schemas.

The model integration test constructs a randomly initialized, tiny **real Transformers Qwen3-VL** with image patch inputs. It checks generation, teacher/student completion logits, language-layer LoRA, gradient checkpointing and nonzero adapter gradients. Another tiny recurrent policy checks shared rollout IDs under different prompt lengths and an optimizer update. Neither is an accuracy experiment.

Main tested versions: PyTorch 2.8.0 CPU, torchvision 0.23.0 CPU, Transformers 4.57.1, PEFT 0.18.1, Accelerate 1.15.0, NumPy 2.4.4, Pillow 12.2.0, pytest 9.1.1. The complete local package snapshot is in `validation-requirements.txt`; this is an environment record, not a portable GPU lockfile.
