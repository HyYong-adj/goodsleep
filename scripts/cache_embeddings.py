from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from psg_only.constants import CLASS_ORDER_VERSION, remap_legacy_labels
from psg_only.data import read_manifest, train_stats
from psg_only.evaluate import load_model

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--train-subject-limit", type=int, default=None)
parser.add_argument("--val-subject-limit", type=int, default=None)
args = parser.parse_args()
if args.batch_size < 1:
    raise ValueError("--batch-size must be positive.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, checkpoint = load_model(args.checkpoint, device)
if checkpoint["model_type"] != "b0":
    raise ValueError("Embedding cache requires a B0 checkpoint.")
digest = hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
root = Path(args.output)
data = checkpoint["config"]["data"]
frame = read_manifest(data["manifest"])
mean, std = train_stats(data["stats"])

for split in ("train", "val"):
    subjects = sorted(frame.loc[frame.split == split, "subject_id"].unique())
    limit = args.train_subject_limit if split == "train" else args.val_subject_limit
    if limit is not None:
        subjects = subjects[:limit]
    for subject_id in subjects:
        out = root / str(subject_id)
        metadata = out / "metadata.json"
        if metadata.exists():
            old = json.loads(metadata.read_text())
            if old.get("checkpoint_sha256") != digest:
                raise FileExistsError(f"Refusing mixed checkpoint cache: {out}")
            continue
        source = Path(data["cache_root"]) / str(subject_id)
        mels = np.load(source / "mels.npy", mmap_mode="r")
        source_labels = np.load(source / "labels.npy", mmap_mode="r")
        source_valid = np.load(source / "valid.npy", mmap_mode="r")
        if mels.ndim != 3 or mels.shape[1:] != (48, 1499) or not (len(mels) == len(source_labels) == len(source_valid)):
            raise ValueError(f"{subject_id}: invalid source cache shape.")
        manifest_indexes = frame.loc[(frame.split == split) & (frame.subject_id == subject_id), "subject_epoch_index"].to_numpy(dtype=np.int64)
        if len(manifest_indexes) != len(np.unique(manifest_indexes)) or manifest_indexes.min() < 0 or manifest_indexes.max() >= len(mels):
            raise ValueError(f"{subject_id}: invalid manifest epoch indexes.")

        # Keep cache indices identical to the source night.  Manifest gaps remain
        # invalid, which prevents B1 from treating distant epochs as neighbours.
        embeddings = np.zeros((len(mels), 192), dtype=np.float16)
        labels = remap_legacy_labels(torch.from_numpy(np.asarray(source_labels, dtype=np.int64).copy())).numpy()
        valid = np.zeros(len(mels), dtype=bool)
        valid[manifest_indexes] = np.asarray(source_valid[manifest_indexes], dtype=bool)
        for start in range(0, len(mels), args.batch_size):
            stop = min(start + args.batch_size, len(mels))
            x = torch.from_numpy(np.asarray(mels[start:stop], dtype=np.float32).copy()).unsqueeze(1)
            x = (x - mean) / std
            with torch.no_grad():
                _, embedding = model(x.to(device))
            embeddings[start:stop] = embedding.cpu().numpy().astype(np.float16)
        labels[~valid] = -100
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "embeddings.npy", embeddings)
        np.save(out / "labels.npy", labels.astype(np.int64))
        np.save(out / "valid.npy", valid)
        metadata.write_text(json.dumps({
            "checkpoint_sha256": digest,
            "config_hash": checkpoint["config_hash"],
            "class_order_version": CLASS_ORDER_VERSION,
            "source_cache": Path(data["cache_root"]).name,
            "subject_id": str(subject_id),
            "shape": [len(embeddings), 192],
            "dtype": "float16",
        }, indent=2))
