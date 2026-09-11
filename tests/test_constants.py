import torch

from psg_only.constants import CANONICAL_CLASSES, remap_legacy_labels, reorder_legacy_logits


def test_legacy_labels_map_to_canonical_order():
    assert CANONICAL_CLASSES == ("Wake", "REM", "Light", "Deep")
    assert remap_legacy_labels(torch.tensor([0, 1, 2, 3])).tolist() == [0, 2, 3, 1]


def test_legacy_logits_reorder_columns():
    logits = torch.tensor([[10.0, 20.0, 30.0, 40.0]])
    assert reorder_legacy_logits(logits).tolist() == [[10.0, 40.0, 20.0, 30.0]]
