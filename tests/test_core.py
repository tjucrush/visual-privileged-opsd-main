import math

import numpy as np
import pytest
import torch
from PIL import Image

from vpopsd.coordinates import parse_point, token_importance
from vpopsd.data import load_records, write_jsonl
from vpopsd.evaluate import score
from vpopsd.loss import weighted_reverse_kl
from vpopsd.prepare import image_digest, normalize_box, split_records
from vpopsd.privilege import PrivilegeConfig, privileged_view, prompt


class Tokenizer:
    def __init__(self, pieces):
        self.pieces = pieces

    def decode(self, ids, **kwargs):
        return "".join(self.pieces[i] for i in ids)


def test_reverse_kl_matches_manual_and_detaches_teacher():
    student = torch.tensor([[[0.2, -0.7, 1.2]]], requires_grad=True)
    teacher = torch.tensor([[[1.8, -0.1, 0.4]]], requires_grad=True)
    loss, _ = weighted_reverse_kl(student, teacher, torch.ones(1, 1), entropy_weighting=False)
    p, q = student.softmax(-1), teacher.softmax(-1)
    expected = (p * (p.log() - q.log())).sum()
    torch.testing.assert_close(loss, expected)
    assert not torch.isclose(loss, (q * (q.log() - p.log())).sum())
    loss.backward()
    assert teacher.grad is None
    assert student.grad is not None and student.grad.abs().sum() > 0


def test_identical_distributions_have_zero_loss():
    logits = torch.randn(2, 5, 7)
    loss, _ = weighted_reverse_kl(logits, logits, torch.ones(2, 5))
    assert loss.item() == pytest.approx(0, abs=1e-7)


def test_masked_tokens_do_not_receive_gradient():
    student = torch.randn(1, 3, 5, requires_grad=True)
    teacher = torch.randn(1, 3, 5)
    loss, _ = weighted_reverse_kl(student, teacher, torch.ones(1, 3), torch.tensor([[1, 0, 1]]))
    loss.backward()
    assert student.grad[0, 1].abs().sum() == 0
    assert student.grad[0, 0].abs().sum() > 0


def test_entropy_reliability_and_absolute_weight_strength():
    student = torch.tensor([[[1.0, 0.0]]])
    uniform = torch.zeros_like(student)
    _, low = weighted_reverse_kl(student, uniform, torch.ones(1, 1))
    _, high = weighted_reverse_kl(student, torch.tensor([[[10.0, -10.0]]]), torch.ones(1, 1))
    assert low["mean_weight"] == pytest.approx(0.05)
    assert high["mean_weight"] > 0.99
    a, _ = weighted_reverse_kl(student, uniform, torch.ones(1, 1))
    b, _ = weighted_reverse_kl(student, uniform, torch.ones(1, 1) * 3)
    torch.testing.assert_close(b, a * 3)


def test_temperature_scaling():
    s, t = torch.randn(1, 2, 4), torch.randn(1, 2, 4)
    loss, _ = weighted_reverse_kl(s, t, torch.ones(1, 2), temperature=2, entropy_weighting=False)
    lp, lq = (s / 2).log_softmax(-1), (t / 2).log_softmax(-1)
    torch.testing.assert_close(loss, 4 * (lp.exp() * (lp - lq)).sum(-1).mean())


@pytest.mark.parametrize(
    "text,expected",
    [
        ("(500, 250)", (0.5, 0.25)),
        ("(0, 1000)", (0, 1)),
        ("(1001, 0)", None),
        ("(1, 2) extra", None),
        ("[500, 250]", None),
        ("(nan, 0)", None),
    ],
)
def test_coordinate_contract(text, expected):
    assert parse_point(text) == expected


def test_digit_weights_merged_tokens_and_eos():
    tok = Tokenizer(["(", "1", "2", "3", ", ", "45", "6", ")", ""])
    weights = token_importance(tok, range(9), digit_scale=4)
    assert weights[1] > weights[2] > weights[3] > weights[0]
    assert weights[5] == weights[1]
    assert weights[-1] == 1
    assert token_importance(Tokenizer(["invalid"]), [0]) == [1]


def test_privilege_coarsens_and_preserves_student():
    image = Image.new("RGB", (100, 80), "white")
    original = np.asarray(image).copy()
    a = privileged_view(image, [0.42, 0.42, 0.44, 0.44])
    b = privileged_view(image, [0.425, 0.425, 0.43, 0.43])
    # Two different exact targets within the same coarse region produce identical cues.
    assert np.array_equal(a, b)
    assert np.array_equal(image, original)
    assert not np.array_equal(a, image)
    assert a.size == image.size
    assert "0.41" not in prompt("Save", teacher=True)
    no_hints = privileged_view(
        image, [0.4, 0.4, 0.5, 0.5], PrivilegeConfig(red_box=False, gaussian_mask=False)
    )
    assert np.array_equal(no_hints, image)


def test_score_counts_invalid_missing_and_macro():
    records = [
        {"id": str(i), "bbox": [0.1, 0.1, 0.9, 0.9], "benchmark": "a" if i == 0 else "b"} for i in range(3)
    ]
    result, _ = score(records, [{"id": "0", "text": "(500, 500)"}, {"id": "1", "text": "broken"}])
    assert result["macro_accuracy"] == 0.5
    assert result["micro_accuracy"] == pytest.approx(1 / 3)
    assert result["benchmarks"]["b"]["invalid"] == 2
    assert result["benchmarks"]["b"]["missing"] == 1
    with pytest.raises(ValueError):
        score(records, [{"id": "oops", "text": "(500, 500)"}])


def test_explicit_coordinate_conversion():
    assert normalize_box([10, 20, 40, 60], "pixels", "xywh", 100, 100) == (0.1, 0.2, 0.5, 0.8)
    with pytest.raises(ValueError):
        normalize_box([10, 20, 40, 60], "normalized", "xyxy", 100, 100)
    with pytest.raises(ValueError):
        normalize_box([math.nan, 0, 1, 1], "normalized", "xyxy", 100, 100)


def test_image_disjoint_reproducible_split(tmp_path):
    rows = []
    for i in range(5):
        path = tmp_path / f"{i}.png"
        Image.new("RGB", (10, 10), (i * 20, 20, 20)).save(path)
        rows.extend({"id": f"{i}-{j}", "image": str(path)} for j in range(3))
    train, val = split_records(rows, 6, 2, 42)
    assert not ({r["image"] for r in train} & {r["image"] for r in val})
    assert (train, val) == split_records(rows, 6, 2, 42)
    excluded = image_digest(rows[0]["image"])
    train, val = split_records(rows, 6, 2, 42, [excluded])
    assert all(image_digest(r["image"]) != excluded for r in train + val)


def test_schema_rejects_duplicate_and_invalid_boxes(tmp_path):
    row = {"id": "1", "image": "x.png", "instruction": "Save", "bbox": [0, 0, 1, 1]}
    path = tmp_path / "rows.jsonl"
    write_jsonl(path, [row, row])
    with pytest.raises(ValueError, match="Duplicate"):
        load_records(path, require_images=False)
    row["bbox"] = [0, 0, 0, 1]
    write_jsonl(path, [row])
    with pytest.raises(ValueError):
        load_records(path, require_images=False)


@pytest.mark.parametrize(
    "benchmark,annotation,expected",
    [
        (
            "screenspot-v2",
            {"img_filename": "screen.png", "instruction": "Save", "bbox": [10, 20, 40, 30]},
            (0.1, 0.2, 0.5, 0.5),
        ),
        (
            "screenspot-pro",
            {"img_filename": "screen.png", "instruction": "Save", "bbox": [10, 20, 40, 30]},
            (0.1, 0.2, 0.4, 0.3),
        ),
        (
            "osworld-g-bbox",
            {
                "image_path": "screen.png",
                "instruction": "Save",
                "box_type": "bbox",
                "box_coordinates": [10, 20, 40, 30],
            },
            (0.1, 0.2, 0.5, 0.5),
        ),
        (
            "ui-vision-element",
            {"image_path": "screen.png", "prompt_to_evaluate": "Save", "bbox": [10, 20, 40, 30]},
            (0.1, 0.2, 0.4, 0.3),
        ),
    ],
)
def test_benchmark_coordinate_conventions(tmp_path, benchmark, annotation, expected):
    import json

    from vpopsd.benchmarks import convert_annotations

    Image.new("RGB", (100, 100)).save(tmp_path / "screen.png")
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps([annotation]), encoding="utf-8")
    rows = convert_annotations([path], tmp_path, benchmark)
    assert rows[0]["bbox"] == expected
    assert rows[0]["instruction"] == "Save"
