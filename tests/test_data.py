import json
from pathlib import Path

import numpy as np
import pandas as pd

from psg_only.data import CachedMelEpochDataset, validate_inputs


def make_fixture(tmp_path: Path):
    root = tmp_path / "cache"
    rows = []
    for split, subject in (("train", "s1"), ("val", "s2")):
        path = root / subject
        path.mkdir(parents=True)
        np.save(path / "mels.npy", np.ones((4, 48, 1499), np.float16))
        np.save(path / "labels.npy", np.array([0, 1, 2, 3], np.int64))
        np.save(path / "valid.npy", np.ones(4, bool))
        rows.extend({"subject_id": subject, "split": split, "label_index": label, "subject_epoch_index": label} for label in range(4))
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    stats = root / "train_stats.json"
    stats.write_text(json.dumps({"mean": 1.0, "std": 2.0}))
    return manifest, root, stats


def test_cached_dataset_remaps_legacy_label(tmp_path):
    manifest, root, stats = make_fixture(tmp_path)
    dataset = CachedMelEpochDataset(manifest, root, stats, "train")
    x, y, subject, epoch, valid = dataset[1]
    assert x.shape == (1, 48, 1499)
    assert int(y) == 2
    assert subject == "s1" and epoch == 1 and valid


def test_validate_inputs_reports_no_test_split(tmp_path):
    manifest, root, stats = make_fixture(tmp_path)
    report = validate_inputs(manifest, root, stats)
    assert report["test_available"] is False
