import pytest
import json
from psg_only.constants import CANONICAL_CLASSES, CLASS_ORDER_VERSION
from psg_only.provenance import file_hash
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


@pytest.mark.parametrize("input_epochs", [40, 80])
def test_windows_cover_each_epoch_once_across_boundary_lengths(tmp_path, input_epochs):
    root = tmp_path / "embeddings"
    rows = []
    lengths = (1, 19, 20, 21, 39, 40, 41, 79, 80, 81)
    for length in lengths:
        subject = f"s{length}"
        directory = root / subject
        directory.mkdir(parents=True)
        np.save(directory / "embeddings.npy", np.broadcast_to(np.arange(length, dtype=np.float16)[:, None], (length, 192)).copy())
        np.save(directory / "labels.npy", np.arange(length, dtype=np.int64) % 4)
        np.save(directory / "valid.npy", np.ones(length, dtype=bool))
        rows.extend({"subject_id": subject, "split": "train", "label_index": (0, 3, 1, 2)[index % 4], "subject_epoch_index": index} for index in range(length))
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)

    for directory in root.iterdir():
        metadata = dict(checkpoint_sha256="fixture", config_hash="fixture", stats_hash="fixture", source_cache="fixture",
                        class_order=list(CANONICAL_CLASSES), class_order_version=CLASS_ORDER_VERSION,
                        subject_id=directory.name, manifest_hash=file_hash(manifest),
                        files={n:file_hash(directory/n) for n in ('embeddings.npy','labels.npy','valid.npy')})
        (directory/'metadata.json').write_text(json.dumps(metadata))
    dataset = EmbeddingWindowDataset(manifest, root, "train", input_epochs=input_epochs)
    for length in lengths:
        subject = f"s{length}"
        target_indexes = []
        for index, window in enumerate(dataset.windows):
            if window.subject_id == subject:
                x, _, input_valid, target_valid, _, indexes = dataset[index]
                left = (input_epochs - 20) // 2
                assert x.shape == (input_epochs, 192)
                assert input_valid.shape == (input_epochs,)
                assert x[left:left+20, 0][target_valid].tolist() == indexes[target_valid].tolist()
                assert (x[~input_valid] == 0).all()
                target_indexes.extend(indexes[target_valid].tolist())
        assert target_indexes == list(range(length))
