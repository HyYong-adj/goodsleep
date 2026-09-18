"""Phase 1 옵션(A1-A4)의 정확성 검증.

window_hours 가 실제 30초 recording 격자와 어긋나면 시각 특징 실험 전체가
무의미해지므로, dataset 이 돌려주는 target_indexes 로부터 직접 대조한다.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from psg_only.constants import CANONICAL_CLASSES
from psg_only.models import B1Model, window_hours
from psg_only.train import build_scheduler, class_weights


def test_window_hours_matches_real_epoch_grid():
    # target_start = 60 -> 첫 target 은 녹음 시작 후 60 * 30s = 30분
    target_indexes = torch.tensor([[60 + i for i in range(20)]], dtype=torch.long)
    input_valid = torch.ones(1, 40, dtype=torch.bool)
    hours = window_hours(target_indexes, input_valid, left_context=10, hours_scale=8.0)
    assert hours.shape == (1, 40)
    # 입력 j 번째는 epoch (60 - 10 + j)
    expected = torch.tensor([[(50 + j) * 30.0 / 3600.0 / 8.0 for j in range(40)]])
    torch.testing.assert_close(hours, expected)
    # 중앙(첫 target) 위치의 실제 시각
    assert hours[0, 10].item() * 8.0 == pytest.approx(0.5)


def test_window_hours_zeroes_padding():
    """밤 시작 경계: 앞쪽 패딩은 음수 시각이 아니라 0 이어야 한다.

    epoch 0 자체의 시각도 0.0 이므로 패딩과 값이 겹친다. 이는 참조 구현과
    같은 동작이며, 모델은 input_valid 로 패딩을 따로 구분한다(embedding 이
    0 으로 마스킹된다). 따라서 여기서는 음수가 없다는 것과 이후 단조 증가만 본다.
    """
    target_indexes = torch.tensor([[i for i in range(20)]], dtype=torch.long)
    input_valid = torch.ones(1, 40, dtype=torch.bool)
    input_valid[0, :10] = False  # 좌측 컨텍스트가 녹음 이전
    hours = window_hours(target_indexes, input_valid, left_context=10)
    assert torch.all(hours[0, :10] == 0.0)
    assert torch.all(hours >= 0.0), "패딩이 음수 시각으로 새면 안 된다"
    assert hours[0, 10].item() == 0.0  # epoch 0 = 녹음 시작
    assert torch.all(torch.diff(hours[0, 10:]) > 0)


def test_window_hours_is_monotonic_within_window():
    target_indexes = torch.tensor([[100 + i for i in range(20)]], dtype=torch.long)
    valid = torch.ones(1, 40, dtype=torch.bool)
    hours = window_hours(target_indexes, valid, left_context=10)
    assert torch.all(torch.diff(hours[0]) > 0)


def test_time_feature_changes_output_and_shape():
    """동일 embedding 이라도 밤의 앞/뒤에 놓이면 출력이 달라져야 한다."""
    torch.manual_seed(0)
    model = B1Model(time_feature=True).eval()
    x = torch.randn(1, 40, 192)
    valid = torch.ones(1, 40, dtype=torch.bool)
    early_idx = torch.arange(10, 30).unsqueeze(0)   # 녹음 초반
    late_idx = torch.arange(800, 820).unsqueeze(0)  # 약 6.7시간 후
    with torch.no_grad():
        early = model(x, valid, window_hours(early_idx, valid, 10))
        late = model(x, valid, window_hours(late_idx, valid, 10))
    assert early.shape == (1, 20, 4)
    assert late.shape == (1, 20, 4)
    assert not torch.allclose(early, late)


def test_time_feature_requires_hours():
    model = B1Model(time_feature=True).eval()
    with pytest.raises(ValueError, match="requires the hours tensor"):
        model(torch.randn(1, 40, 192), torch.ones(1, 40, dtype=torch.bool))


def test_default_model_ignores_hours_and_is_unchanged():
    """time_feature=False 는 파라미터가 늘지 않아야 기존 체크포인트와 호환된다."""
    baseline = B1Model()
    assert baseline.time_proj is None
    assert "time_proj.weight" not in baseline.state_dict()


def test_class_weight_multiplier_scales_rem_relative_to_others():
    labels = np.array([0] * 100 + [1] * 10 + [2] * 1000 + [3] * 50)
    base = class_weights(labels)
    boosted = class_weights(labels, {"REM": 2.0})
    rem = CANONICAL_CLASSES.index("REM")
    # REM 대 나머지 비율이 정확히 2배
    for other in range(4):
        if other == rem:
            continue
        ratio_base = base[rem] / base[other]
        ratio_boosted = boosted[rem] / boosted[other]
        assert (ratio_boosted / ratio_base).item() == pytest.approx(2.0)
    # 평균 정규화 유지 -> loss scale 이 parent 와 비교 가능
    assert boosted.mean().item() == pytest.approx(1.0)


def test_class_weight_rejects_unknown_class():
    labels = np.array([0, 1, 2, 3])
    with pytest.raises(ValueError, match="Unknown class"):
        class_weights(labels, {"N2": 2.0})


def test_cosine_scheduler_warms_up_then_decays():
    model = torch.nn.Linear(2, 2)
    opt = torch.optim.AdamW(model.parameters(), lr=1.0)
    sched = build_scheduler(opt, {"scheduler": "cosine", "warmup_epochs": 3}, max_epochs=10)
    seen = []
    for _ in range(10):
        seen.append(opt.param_groups[0]["lr"])
        sched.step()
    # warmup 구간은 증가
    assert seen[0] < seen[1] < seen[2]
    assert seen[2] == pytest.approx(1.0)
    # 이후 단조 감소, 끝에서 0 근처
    assert all(a > b for a, b in zip(seen[3:], seen[4:]))
    assert seen[-1] < 0.1


def test_scheduler_none_is_default_and_constant():
    opt = torch.optim.AdamW(torch.nn.Linear(2, 2).parameters(), lr=3e-4)
    assert build_scheduler(opt, {}, max_epochs=10) is None
    assert build_scheduler(opt, {"scheduler": "none"}, max_epochs=10) is None


def test_scheduler_rejects_unsupported_and_bad_warmup():
    opt = torch.optim.AdamW(torch.nn.Linear(2, 2).parameters(), lr=1.0)
    with pytest.raises(ValueError, match="Unsupported scheduler"):
        build_scheduler(opt, {"scheduler": "step"}, max_epochs=10)
    with pytest.raises(ValueError, match="warmup_epochs"):
        build_scheduler(opt, {"scheduler": "cosine", "warmup_epochs": 10}, max_epochs=10)
