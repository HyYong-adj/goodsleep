"""Phase 4 gated KD 검증."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from psg_only.constants import CANONICAL_CLASSES, CANONICAL_LOGITS_FROM_LEGACY  # noqa: E402
from run_phase4 import gated_kd_loss  # noqa: E402

W = torch.ones(4)


def test_kd_is_zero_when_student_matches_teacher():
    teacher = torch.tensor([[3.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0]])
    targets = torch.tensor([0, 1])
    assert gated_kd_loss(teacher.clone(), teacher, targets, W, 2.0).item() == pytest.approx(0.0, abs=1e-6)


def test_kd_grows_as_student_diverges():
    teacher = torch.tensor([[4.0, 0.0, 0.0, 0.0]])
    targets = torch.tensor([0])
    near = gated_kd_loss(torch.tensor([[3.0, 0.0, 0.0, 0.0]]), teacher, targets, W, 2.0)
    far = gated_kd_loss(torch.tensor([[0.0, 0.0, 0.0, 4.0]]), teacher, targets, W, 2.0)
    assert far > near > 0


def test_gate_downweights_unconfident_teacher():
    """헷갈리는 teacher(균등 분포)는 확신하는 teacher 보다 적게 기여해야 한다."""
    student = torch.tensor([[0.0, 0.0, 0.0, 4.0]])
    targets = torch.tensor([0])
    confident = gated_kd_loss(student, torch.tensor([[8.0, 0.0, 0.0, 0.0]]), targets, W, 2.0)
    unsure = gated_kd_loss(student, torch.tensor([[0.1, 0.0, 0.0, 0.0]]), targets, W, 2.0)
    assert confident > unsure


def test_kd_ignores_masked_targets_and_returns_zero_when_all_masked():
    teacher = torch.tensor([[4.0, 0.0, 0.0, 0.0], [0.0, 4.0, 0.0, 0.0]])
    student = torch.tensor([[0.0, 0.0, 0.0, 4.0], [0.0, 4.0, 0.0, 0.0]])
    both = gated_kd_loss(student, teacher, torch.tensor([0, 1]), W, 2.0)
    only_matching = gated_kd_loss(student, teacher, torch.tensor([-100, 1]), W, 2.0)
    assert only_matching.item() == pytest.approx(0.0, abs=1e-6)
    assert both > only_matching
    allmasked = gated_kd_loss(student, teacher, torch.tensor([-100, -100]), W, 2.0)
    assert allmasked.item() == 0.0


def test_kd_is_differentiable_wrt_student_only():
    student = torch.tensor([[1.0, 0.0, 0.0, 0.0]], requires_grad=True)
    teacher = torch.tensor([[4.0, 0.0, 0.0, 0.0]], requires_grad=True)
    gated_kd_loss(student, teacher, torch.tensor([0]), W, 2.0).backward()
    assert student.grad is not None and torch.any(student.grad != 0)
    assert teacher.grad is None or torch.all(teacher.grad == 0)


def test_class_weight_scales_contribution():
    """두 epoch 의 KL 이 서로 달라야 가중치 효과가 드러난다.

    epoch0 은 student 가 teacher 와 일치(KL 약 0), epoch1 은 크게 어긋난다.
    epoch1 의 정답 클래스(REM)를 키우면 가중평균이 그쪽으로 끌려가야 한다.
    """
    student = torch.tensor([[4.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 4.0]])
    teacher = torch.tensor([[4.0, 0.0, 0.0, 0.0], [0.0, 4.0, 0.0, 0.0]])
    targets = torch.tensor([0, 1])
    flat = gated_kd_loss(student, teacher, targets, torch.ones(4), 2.0)
    rem_heavy = gated_kd_loss(student, teacher, targets, torch.tensor([1.0, 10.0, 1.0, 1.0]), 2.0)
    assert rem_heavy > flat, "소수 클래스 가중치가 KD 항에 반영되지 않는다"


def test_teacher_cache_class_order_and_axis():
    """빌드된 teacher 캐시가 canonical 순서·epoch 축인지 실제 파일로 확인한다."""
    import json
    cache = ROOT / "artifacts/teacher_canonical"
    if not (cache / "metadata.json").exists():
        pytest.skip("teacher cache not built")
    meta = json.loads((cache / "metadata.json").read_text())
    assert meta["class_order"] == list(CANONICAL_CLASSES)
    # 순서가 틀리면 일치율이 ~0.1 로 떨어진다
    assert meta["teacher_argmax_agreement_with_labels"] > 0.7
    assert CANONICAL_LOGITS_FROM_LEGACY == (0, 3, 1, 2)


def test_reorder_maps_legacy_positions_correctly():
    legacy = np.array([[10.0, 20.0, 30.0, 40.0]])  # Wake, Light, Deep, REM
    canonical = legacy[:, list(CANONICAL_LOGITS_FROM_LEGACY)]
    assert canonical.tolist() == [[10.0, 40.0, 20.0, 30.0]]  # Wake, REM, Light, Deep
