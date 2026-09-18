"""Phase 2 옵션(night_norm, SpecAugment)의 정확성 검증."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from psg_only.data import night_statistics  # noqa: E402
from psg_only.train import spec_augment  # noqa: E402


@pytest.fixture
def subject(tmp_path):
    rng = np.random.default_rng(0)
    root = tmp_path / "S1"
    root.mkdir()
    mels = rng.normal(-40.0, 10.0, size=(6, 48, 32)).astype(np.float32)
    valid = np.array([True, True, False, True, True, False])
    np.save(root / "mels.npy", mels)
    np.save(root / "valid.npy", valid)
    return root, mels, valid


def test_night_statistics_uses_only_valid_epochs(subject):
    root, mels, valid = subject
    mean, std = night_statistics(root)
    block = mels[valid].astype(np.float64)
    assert mean == pytest.approx(block.mean())
    assert std == pytest.approx(block.std())
    # 무효 epoch 을 포함하면 값이 달라져야 한다(즉 제외가 실제로 동작한다)
    assert mean != pytest.approx(mels.astype(np.float64).mean())


def test_night_statistics_prefers_cached_json_and_never_writes(subject):
    root, _, _ = subject
    (root / "night_stats.json").write_text(json.dumps({"mean": -33.0, "std": 7.0}))
    before = sorted(p.name for p in root.iterdir())
    assert night_statistics(root) == (-33.0, 7.0)
    # 소스 캐시(타 연구자 소유)에 아무것도 쓰지 않아야 한다
    assert sorted(p.name for p in root.iterdir()) == before


def test_night_statistics_rejects_degenerate_cache(subject):
    root, _, _ = subject
    (root / "night_stats.json").write_text(json.dumps({"mean": -33.0, "std": 0.0}))
    with pytest.raises(ValueError, match="invalid night statistics"):
        night_statistics(root)


def test_night_statistics_rejects_all_invalid(tmp_path):
    root = tmp_path / "S2"
    root.mkdir()
    np.save(root / "mels.npy", np.zeros((2, 48, 8), dtype=np.float32))
    np.save(root / "valid.npy", np.array([False, False]))
    with pytest.raises(ValueError, match="no valid epochs"):
        night_statistics(root)


def test_night_norm_centres_each_recording():
    """서로 다른 gain 의 두 녹음이 정규화 후 같은 분포로 정렬되어야 한다."""
    rng = np.random.default_rng(1)
    base = rng.normal(0.0, 1.0, size=(4, 48, 32))
    quiet, loud = base - 60.0, base - 30.0  # 30 dB 차이
    out = []
    for block in (quiet, loud):
        mean, std = block.mean(), block.std()
        out.append((block - mean) / std)
    assert np.allclose(out[0], out[1], atol=1e-9)
    # 전역 통계 하나로는 정렬되지 않는다
    g_mean = np.concatenate([quiet, loud]).mean()
    g_std = np.concatenate([quiet, loud]).std()
    assert not np.allclose((quiet - g_mean) / g_std, (loud - g_mean) / g_std, atol=1e-6)


def test_spec_augment_preserves_shape_and_masks_to_zero():
    torch.manual_seed(0)
    import random
    random.seed(0)
    x = torch.full((4, 1, 48, 300), 5.0)
    out = spec_augment(x, freq_mask=8, time_mask=150, gain_std=0.0)
    assert out.shape == x.shape
    # gain jitter 없이는 마스킹된 위치만 0, 나머지는 원본
    assert torch.any(out == 0.0), "마스킹이 전혀 적용되지 않았다"
    assert set(out.unique().tolist()) <= {0.0, 5.0}


def test_spec_augment_does_not_mutate_input():
    import random
    random.seed(1)
    torch.manual_seed(1)
    x = torch.full((2, 1, 48, 300), 3.0)
    original = x.clone()
    spec_augment(x, gain_std=0.5)
    torch.testing.assert_close(x, original)


def test_spec_augment_gain_jitter_is_per_sample():
    import random
    random.seed(2)
    torch.manual_seed(2)
    x = torch.zeros(8, 1, 48, 300)
    out = spec_augment(x, freq_mask=0, time_mask=0, gain_std=1.0)
    # 마스킹을 끄면 각 샘플은 상수 offset 하나만 더해진다
    offsets = out.reshape(8, -1)[:, 0]
    for index in range(8):
        assert torch.allclose(out[index], torch.full_like(out[index], offsets[index].item()))
    assert offsets.std().item() > 0.1, "샘플별로 다른 gain 이어야 한다"


def test_spec_augment_rejects_bad_shape_and_oversized_mask():
    with pytest.raises(ValueError, match=r"\[batch, 1, mel, frames\]"):
        spec_augment(torch.zeros(2, 3, 48, 300))
    with pytest.raises(ValueError, match="Mask width"):
        spec_augment(torch.zeros(2, 1, 48, 300), freq_mask=48)
