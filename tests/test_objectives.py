"""focal / supcon 목적함수 검증."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from psg_only.objectives import focal_loss, supcon_loss  # noqa: E402

W = torch.tensor([1.058, 1.314, 0.322, 1.306])  # 실제 학습셋 가중치


def weighted_ce(logits, targets, weight):
    mask = targets != -100
    return F.cross_entropy(logits[mask], targets[mask], weight=weight)


def test_focal_gamma_zero_equals_weighted_ce():
    torch.manual_seed(0)
    logits = torch.randn(64, 4)
    targets = torch.randint(0, 4, (64,))
    assert focal_loss(logits, targets, W, gamma=0.0).item() == pytest.approx(
        weighted_ce(logits, targets, W).item(), abs=1e-5)


def test_focal_downweights_easy_samples():
    """이미 잘 맞히는 샘플은 gamma 가 커질수록 기여가 줄어야 한다."""
    easy = torch.tensor([[8.0, 0.0, 0.0, 0.0]])
    t = torch.tensor([0])
    assert focal_loss(easy, t, W, gamma=2.0) < focal_loss(easy, t, W, gamma=0.0)


def test_focal_preserves_hard_samples():
    hard = torch.tensor([[0.0, 0.0, 0.0, 4.0]])
    t = torch.tensor([0])
    ratio = (focal_loss(hard, t, W, gamma=2.0) / focal_loss(hard, t, W, gamma=0.0)).item()
    assert ratio > 0.9, "틀린 샘플은 거의 그대로 남아야 한다"


def test_focal_ignores_masked_and_handles_all_masked():
    logits = torch.randn(8, 4)
    t = torch.full((8,), -100)
    assert focal_loss(logits, t, W).item() == 0.0
    t2 = t.clone(); t2[0] = 1
    assert focal_loss(logits, t2, W).item() > 0


def test_supcon_zero_when_classes_perfectly_separated():
    """같은 클래스가 완전히 겹치고 다른 클래스와 직교하면 손실이 0 에 가까워야 한다."""
    z = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    y = torch.tensor([0, 0, 1, 1])
    loss, st = supcon_loss(z * 50, y, temperature=0.1)
    assert loss.item() < 0.01
    assert st["used"] == 4


def test_supcon_penalises_mixed_classes():
    torch.manual_seed(0)
    z = torch.randn(16, 8)
    y = torch.tensor([0, 1] * 8)
    mixed = supcon_loss(z, y, temperature=0.1)[0]
    separated = supcon_loss(torch.cat([torch.ones(8, 8), -torch.ones(8, 8)]) * 5,
                            torch.tensor([0] * 8 + [1] * 8), temperature=0.1)[0]
    assert mixed > separated


def test_supcon_skips_anchors_without_positives():
    z = torch.randn(3, 4)
    y = torch.tensor([0, 1, 2])  # 모두 유일 -> positive 없음
    loss, st = supcon_loss(z, y)
    assert loss.item() == 0.0 and st["used"] == 0 and st["anchors"] == 3


def test_supcon_reports_rem_anchor_count():
    """REM(canonical index 1) anchor 수를 보고해야 한다 — 희소성이 이 실험의 제약이다."""
    y = torch.tensor([2, 2, 1, 2, 2, 1, 2, -100])
    _, st = supcon_loss(torch.randn(8, 4), y)
    assert st["rem_anchors"] == 2
    assert st["anchors"] == 7  # -100 제외


def test_supcon_class_weight_shifts_toward_rare_class():
    """희소 클래스 anchor 가 잘 뭉쳐 있으면, 가중치를 주었을 때 총손실이 내려가야 한다."""
    torch.manual_seed(0)
    light = torch.randn(10, 8)                       # 흩어진 다수 클래스
    rem = torch.ones(4, 8) + 0.001 * torch.randn(4, 8)  # 잘 뭉친 희소 클래스
    z = torch.cat([light, rem])
    y = torch.tensor([2] * 10 + [1] * 4)
    flat = supcon_loss(z, y, weight=torch.ones(4), temperature=0.1)[0]
    weighted = supcon_loss(z, y, weight=W, temperature=0.1)[0]
    assert weighted < flat


def test_supcon_is_differentiable():
    z = torch.randn(8, 4, requires_grad=True)
    y = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    supcon_loss(z, y, W)[0].backward()
    assert z.grad is not None and torch.any(z.grad != 0)


# ---- BiLSTM 뒤 Transformer encoder 적층 (choihy-e021) ----

from psg_only.endtoend import EndToEndModel  # noqa: E402


def test_post_transformer_absent_by_default_keeps_parent_parameters():
    """기본값 0 이면 기존 parent 와 파라미터 수가 같아야 비교가 성립한다."""
    assert EndToEndModel().post_transformer is None
    assert sum(p.numel() for p in EndToEndModel().parameters()) == 2_631_528


def test_post_transformer_adds_capacity_and_preserves_output_shape():
    m = EndToEndModel(post_transformer_layers=2)
    assert m.post_transformer is not None
    x = torch.randn(2, 40, 1, 48, 1499)
    v = torch.ones(2, 40, dtype=torch.bool)
    assert m(x, v).shape == (2, 20, 4)


def _run_post_transformer(model, sequence, valid):
    """forward 의 post-transformer 구간만 재현한다 (모듈 수준 검증용)."""
    x = sequence + model.post_position.unsqueeze(0)
    return model.post_transformer(x, src_key_padding_mask=~valid)


def test_post_transformer_padding_mask_blocks_leakage():
    """패딩 위치를 바꿔도 유효 위치의 출력은 그대로여야 한다.

    end-to-end 로는 검증할 수 없다 — 학습 안 된 encoder 가 어떤 입력이든 거의 같은
    embedding 을 내므로(epoch 간 std 0.002) 마스크가 없어도 출력이 안 변해 테스트가 공허해진다.
    그래서 BiLSTM 출력 자리에 직접 값을 넣어 transformer 구간만 시험한다.
    """
    torch.manual_seed(0)
    m = EndToEndModel(post_transformer_layers=2).eval()
    sequence = torch.randn(1, 40, 256)
    valid = torch.ones(1, 40, dtype=torch.bool)
    valid[0, :10] = False
    perturbed = sequence.clone()
    perturbed[0, :10] = torch.randn(10, 256) * 10  # 패딩 구간만 크게 교란
    with torch.no_grad():
        a = _run_post_transformer(m, sequence, valid)
        b = _run_post_transformer(m, perturbed, valid)
    torch.testing.assert_close(a[:, 10:], b[:, 10:], rtol=1e-4, atol=1e-4)


def test_padding_mask_test_is_not_vacuous():
    """위 테스트가 의미 있으려면, 마스크를 빼면 같은 교란이 출력을 바꿔야 한다."""
    torch.manual_seed(0)
    m = EndToEndModel(post_transformer_layers=2).eval()
    sequence = torch.randn(1, 40, 256)
    perturbed = sequence.clone()
    perturbed[0, :10] = torch.randn(10, 256) * 10
    with torch.no_grad():
        a = m.post_transformer(sequence + m.post_position.unsqueeze(0))
        b = m.post_transformer(perturbed + m.post_position.unsqueeze(0))
    assert not torch.allclose(a[:, 10:], b[:, 10:], rtol=1e-3, atol=1e-3), \
        "마스크 없이도 출력이 같다면 누수 테스트가 공허하다"


def test_post_transformer_mixes_across_valid_positions():
    """유효 위치끼리는 실제로 정보가 섞여야 한다(전역 문맥이 쓰이는지)."""
    torch.manual_seed(0)
    m = EndToEndModel(post_transformer_layers=2).eval()
    sequence = torch.randn(1, 40, 256)
    valid = torch.ones(1, 40, dtype=torch.bool)
    perturbed = sequence.clone()
    perturbed[0, 38:] = torch.randn(2, 256) * 5  # target 창(10:30) 밖의 먼 위치
    with torch.no_grad():
        a = _run_post_transformer(m, sequence, valid)
        b = _run_post_transformer(m, perturbed, valid)
    assert not torch.allclose(a[:, 10:30], b[:, 10:30], rtol=1e-3, atol=1e-3)


def test_post_transformer_heads_must_divide_width():
    with pytest.raises(ValueError, match="post_transformer_heads"):
        EndToEndModel(hidden_dim=128, post_transformer_layers=1, post_transformer_heads=7)


# ---- front-end 변환 (choihy-e023 PCEN / e024 delta) ----

import numpy as np  # noqa: E402
from psg_only.data import PCEN_DEFAULT, logmel_delta, pcen  # noqa: E402


def _block(seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(-40.0, 8.0, size=(48, 1499)).astype(np.float32)


def test_pcen_shape_finite_nonnegative():
    out = pcen(_block(), **PCEN_DEFAULT)
    assert out.shape == (48, 1499) and out.dtype == np.float32
    assert np.isfinite(out).all() and (out >= -1e-6).all()


def test_pcen_is_gain_invariant():
    """PCEN 의 핵심 성질: 녹음 전체 이득이 바뀌어도 출력은 거의 같아야 한다 (alpha≈1)."""
    b = _block()
    a, c = pcen(b, **PCEN_DEFAULT), pcen(b + 20.0, **PCEN_DEFAULT)  # +20 dB = 100x power
    rel = np.abs(a - c).mean() / (np.abs(a).mean() + 1e-9)
    assert rel < 0.15, f"gain invariance broken: rel diff {rel:.3f}"


def test_pcen_suppresses_stationary_background_emphasises_onset():
    """정지 배경은 눌리고, 갑작스러운 onset 은 살아남아야 한다."""
    b = np.full((48, 1499), -40.0, np.float32)
    b[:, 800:830] += 20.0  # 0.6 s 짜리 사건
    out = pcen(b, **PCEN_DEFAULT)
    steady, onset = out[:, 700:790].mean(), out[:, 800:806].mean()
    assert onset > 3 * steady


def test_pcen_no_startup_transient():
    """첫 프레임으로 상태를 초기화했으므로 상수 입력은 시간에 걸쳐 평평해야 한다."""
    out = pcen(np.full((48, 1499), -40.0, np.float32), **PCEN_DEFAULT)
    assert np.allclose(out[:, 0], out[:, -1], atol=1e-4)


def test_logmel_delta_shape_and_semantics():
    b = _block()
    d = logmel_delta(b)
    assert d.shape == (2, 48, 1499)
    np.testing.assert_allclose(d[0], b)
    np.testing.assert_allclose(d[1][:, 1:], b[:, 1:] - b[:, :-1], rtol=1e-6, atol=1e-5)
    assert np.all(d[1][:, 0] == 0.0)  # 첫 프레임 복제 패딩 → 차분 0


def test_delta_channel_is_zero_mean_ish():
    d = logmel_delta(_block(3))
    assert abs(d[1].mean()) < 0.5


def test_conformer_accepts_two_input_channels():
    from psg_only.conformer import ConformerEpochModel
    m = ConformerEpochModel(in_channels=2).eval()
    with torch.no_grad():
        logits, emb = m(torch.randn(2, 2, 48, 1499))
    assert logits.shape == (2, 4) and emb.shape == (2, 192)
    with pytest.raises(ValueError, match="fixed Mel shape"):
        m(torch.randn(2, 1, 48, 1499))


def test_endtoend_two_channel_forward():
    m = EndToEndModel(in_channels=2).eval()
    with torch.no_grad():
        out = m(torch.randn(1, 40, 2, 48, 1499), torch.ones(1, 40, dtype=torch.bool))
    assert out.shape == (1, 20, 4)


def test_default_single_channel_parameter_count_unchanged():
    assert sum(p.numel() for p in EndToEndModel().parameters()) == 2_631_528
