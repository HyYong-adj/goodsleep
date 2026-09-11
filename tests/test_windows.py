import numpy as np
import pandas as pd

from psg_only.data import EmbeddingWindowDataset
from psg_only.windows import stitch_predictions


def test_stitch_orders_and_rejects_duplicates():
    rows = [
        {"subject_id": "b", "epoch_index": 0},
        {"subject_id": "a", "epoch_index": 1},
        {"subject_id": "a", "epoch_index": 0},
    ]
    assert [(row["subject_id"], row["epoch_index"]) for row in stitch_predictions(rows)] == [("a", 0), ("a", 1), ("b", 0)]
    try:
        stitch_predictions(rows + [{"subject_id": "a", "epoch_index": 0}])
        assert False
    except ValueError:
        pass


def test_windows_cover_each_epoch_once_across_boundary_lengths(tmp_path):
    root = tmp_path / "embeddings"
    rows = []
    lengths = (1, 19, 20, 21, 39, 40, 41)
    for length in lengths:
        subject = f"s{length}"
        directory = root / subject
        directory.mkdir(parents=True)
        np.save(directory / "embeddings.npy", np.zeros((length, 192), np.float16))
        np.save(directory / "labels.npy", np.arange(length, dtype=np.int64) % 4)
        np.save(directory / "valid.npy", np.ones(length, dtype=bool))
        rows.extend({"subject_id": subject, "split": "train", "label_index": index % 4, "subject_epoch_index": index} for index in range(length))
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)

    dataset = EmbeddingWindowDataset(manifest, root, "train")
    for length in lengths:
        subject = f"s{length}"
        target_indexes = []
        for index, window in enumerate(dataset.windows):
            if window.subject_id == subject:
                _, _, _, target_valid, _, indexes = dataset[index]
                target_indexes.extend(indexes[target_valid].tolist())
        assert target_indexes == list(range(length))
