from __future__ import annotations

import torch

CANONICAL_CLASSES = ("Wake", "REM", "Light", "Deep")
CLASS_ORDER_VERSION = "wake-rem-light-deep-v1"
LEGACY_CLASSES = ("Wake", "Light", "Deep", "REM")
LEGACY_LABEL_TO_CANONICAL = (0, 2, 3, 1)
CANONICAL_LOGITS_FROM_LEGACY = (0, 3, 1, 2)
NAME_TO_INDEX = {name: index for index, name in enumerate(CANONICAL_CLASSES)}


def remap_legacy_labels(labels: torch.Tensor) -> torch.Tensor:
    mapping = torch.tensor(LEGACY_LABEL_TO_CANONICAL, device=labels.device, dtype=torch.long)
    labels = labels.to(torch.long)
    valid = labels >= 0
    output = labels.clone()
    if valid.any():
        if (labels[valid] >= len(mapping)).any():
            raise ValueError("Legacy labels must be in [0, 3].")
        output[valid] = mapping[labels[valid]]
    return output


def reorder_legacy_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.shape[-1] != 4:
        raise ValueError("Expected four legacy logit columns.")
    return logits[..., list(CANONICAL_LOGITS_FROM_LEGACY)]
