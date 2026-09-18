from __future__ import annotations

import numpy as np


def stitch_predictions(rows: list[dict], expected_keys: set | None = None) -> list[dict]:
    seen = set()
    ordered = []
    for row in sorted(rows, key=lambda item: (item["subject_id"], item["epoch_index"])):
        key = (row["subject_id"], int(row["epoch_index"]))
        if key in seen:
            raise ValueError(f"Duplicate stitched prediction: {key}.")
        seen.add(key)
        ordered.append(row)
    if expected_keys is not None and seen != expected_keys:
        raise ValueError(f"Prediction coverage mismatch: missing={len(expected_keys - seen)}, extra={len(seen - expected_keys)}")
    return ordered


def context_options(data):
    """Resolve symmetric context while preserving the 20-target evaluation grid."""
    size = data.get("input_epochs", 40)
    target = data.get("target_epochs", 20)
    stride = data.get("stride", 20)
    if type(size) is not int or size < 20 or (size - 20) % 2:
        raise ValueError("input_epochs must be an integer >=20 with symmetric context.")
    if type(target) is not int or target != 20 or type(stride) is not int or stride != 20:
        raise ValueError("target_epochs and stride must remain 20.")
    left = (size - target) // 2
    if "left_context" in data and data["left_context"] != left:
        raise ValueError("left_context must match symmetric input/target lengths.")
    return {"input_epochs": size, "target_epochs": target, "stride": stride}
