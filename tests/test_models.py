import torch

from psg_only.models import B0Model, B1Model


def test_b0_shapes():
    model = B0Model()
    logits, embedding = model(torch.randn(2, 1, 48, 1499))
    assert logits.shape == (2, 4)
    assert embedding.shape == (2, 192)


def test_b1_shapes():
    model = B1Model()
    logits = model(torch.randn(3, 40, 192), torch.ones(3, 40, dtype=torch.bool))
    assert logits.shape == (3, 20, 4)


def test_b1_zeros_invalid_embedding_values():
    model = B1Model().eval()
    baseline = torch.zeros(1, 40, 192)
    contaminated = baseline.clone()
    contaminated[:, :10] = 999.0
    valid = torch.ones(1, 40, dtype=torch.bool)
    valid[:, :10] = False
    with torch.no_grad():
        assert torch.allclose(model(baseline, valid), model(contaminated, valid))
