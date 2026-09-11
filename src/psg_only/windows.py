from __future__ import annotations

import numpy as np


def stitch_predictions(rows: list[dict]) -> list[dict]:
    seen = set()
    ordered = []
    for row in sorted(rows, key=lambda item: (item["subject_id"], item["epoch_index"])):
        key = (row["subject_id"], int(row["epoch_index"]))
        if key in seen:
            raise ValueError(f"Duplicate stitched prediction: {key}.")
        seen.add(key)
        ordered.append(row)
    return ordered
