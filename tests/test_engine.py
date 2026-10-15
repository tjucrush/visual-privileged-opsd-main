from types import SimpleNamespace

import torch

from vpopsd.engine import completion_logits, distillation_step


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(16, 12)
        self.rnn = torch.nn.GRU(12, 12, batch_first=True)
        self.head = torch.nn.Linear(12, 16)
        self.prefixes = []

    def forward(self, input_ids, attention_mask, logits_to_keep, **kwargs):
        self.prefixes.append(input_ids.detach().clone())
        hidden, _ = self.rnn(self.emb(input_ids))
        return SimpleNamespace(logits=self.head(hidden[:, logits_to_keep]))

    def generate(self, input_ids, **kwargs):
        # Fixed trajectory isolates alignment and optimization, not sampling quality.
        return torch.cat([input_ids, torch.tensor([[4, 5, 6]], device=input_ids.device)], dim=1)


class TinyTokenizer:
    pad_token_id = 0

    def decode(self, ids, **kwargs):
        return "(500, 500)"


def test_causal_logits_select_preceding_positions():
    model = TinyPolicy()
    inputs = {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.ones(1, 3, dtype=torch.long)}
    completion = torch.tensor([[4, 5]])
    got = completion_logits(model, inputs, completion)
    full = model(
        input_ids=torch.tensor([[1, 2, 3, 4, 5]]), attention_mask=None, logits_to_keep=torch.arange(5)
    ).logits
    torch.testing.assert_close(got, full[:, 2:4])


def test_on_policy_alignment_with_different_prompt_lengths_and_optimizer():
    torch.manual_seed(1)
    model = TinyPolicy()
    student = {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.ones(1, 2, dtype=torch.long)}
    teacher = {
        "input_ids": torch.tensor([[7, 8, 9, 10]]),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
    }
    processor = SimpleNamespace(tokenizer=TinyTokenizer())
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    before = model.head.weight.detach().clone()
    loss, metrics = distillation_step(model, processor, student, teacher, {})
    assert torch.isfinite(loss) and metrics["completion_tokens"] == 3
    assert model.prefixes[0][0, -3:].tolist() == model.prefixes[1][0, -3:].tolist() == [4, 5, 6]
    assert model.prefixes[0].shape[1] == 7 and model.prefixes[1].shape[1] == 5
    loss.backward()
    optimizer.step()
    assert not torch.equal(before, model.head.weight)


def test_real_tiny_qwen_multimodal_forward_and_lora():
    """Exercise the actual pinned Qwen API on CPU without model downloads."""
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    cfg = Qwen3VLConfig(
        text_config={
            "vocab_size": 64,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "rope_scaling": {"rope_type": "default", "mrope_section": [1, 1, 2]},
        },
        vision_config={
            "depth": 1,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 4,
            "out_hidden_size": 32,
            "patch_size": 2,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "num_position_embeddings": 16,
            "deepstack_visual_indexes": [],
        },
        image_token_id=60,
        video_token_id=61,
        vision_start_token_id=62,
        vision_end_token_id=63,
    )
    model = Qwen3VLForConditionalGeneration(cfg)
    model = get_peft_model(
        model,
        LoraConfig(
            r=2,
            lora_alpha=4,
            task_type="CAUSAL_LM",
            target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj)$",
        ),
    )
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    inputs = {
        "input_ids": torch.tensor([[1, 62, 60, 63, 2]]),
        "attention_mask": torch.ones(1, 5, dtype=torch.long),
        "pixel_values": torch.randn(4, 24),
        "image_grid_thw": torch.tensor([[1, 2, 2]]),
    }
    completion = torch.tensor([[3, 4, 5]])
    model.eval()
    with torch.no_grad():
        teacher_inputs = dict(inputs)
        teacher_inputs["pixel_values"] = inputs["pixel_values"] * 0.5
        teacher = completion_logits(model, teacher_inputs, completion)
        generated = model.generate(
            **inputs, max_new_tokens=3, do_sample=False, pad_token_id=0, eos_token_id=59
        )
    assert generated.shape[1] > inputs["input_ids"].shape[1]
    model.train()
    student = completion_logits(model, inputs, completion)
    from vpopsd.loss import weighted_reverse_kl

    loss, _ = weighted_reverse_kl(student, teacher, torch.ones(1, 3))
    loss.backward()
    assert student.shape == (1, 3, 64)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters() if p.requires_grad)
