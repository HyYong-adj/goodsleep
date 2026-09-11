from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .constants import CANONICAL_CLASSES, remap_legacy_labels

REQUIRED_COLUMNS = {
    "subject_id", "split", "label_index", "subject_epoch_index",
}


def read_manifest(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"subject_id": str})
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    allowed = {"train", "val", "test"}
    unknown = set(frame["split"].unique()) - allowed
    if unknown:
        raise ValueError(f"Unexpected split values: {sorted(unknown)}")
    return frame


def train_stats(path: str | Path) -> tuple[float, float]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    mean, std = float(payload["mean"]), float(payload["std"])
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
        raise ValueError("Invalid train normalization statistics.")
    return mean, std


def _subject_rows(frame: pd.DataFrame, split: str, subject_limit: int | None = None) -> pd.DataFrame:
    selected = frame.loc[frame["split"] == split].copy()
    if selected.empty:
        raise ValueError(f"Manifest has no rows for split={split!r}.")
    subjects = sorted(selected["subject_id"].unique())
    if subject_limit is not None:
        subjects = subjects[:subject_limit]
        selected = selected.loc[selected["subject_id"].isin(subjects)].copy()
    selected.sort_values(["subject_id", "subject_epoch_index"], inplace=True)
    return selected.reset_index(drop=True)


def validate_inputs(manifest_path: str | Path, cache_root: str | Path, stats_path: str | Path) -> dict:
    frame = read_manifest(manifest_path)
    root = Path(cache_root)
    train_stats(stats_path)
    groups = {name: set(frame.loc[frame.split == name, "subject_id"]) for name in ("train", "val", "test")}
    if groups["train"] & groups["val"]:
        raise ValueError("Subject leakage between train and val.")
    if groups["train"] & groups["test"] or groups["val"] & groups["test"]:
        raise ValueError("Subject leakage involving test.")
    report = {"subjects": {name: len(values) for name, values in groups.items()}, "rows": int(len(frame))}
    report["class_counts"] = {}
    for split in ("train", "val"):
        values = frame.loc[frame.split == split, "label_index"].to_numpy(dtype=np.int64)
        canonical = np.asarray([ (0, 2, 3, 1)[value] for value in values ], dtype=np.int64)
        counts = np.bincount(canonical, minlength=4)
        if np.any(counts == 0):
            raise ValueError(f"Missing canonical class in {split}: {counts.tolist()}")
        report["class_counts"][split] = dict(zip(CANONICAL_CLASSES, counts.tolist()))
    for subject_id, rows in frame.groupby("subject_id", sort=False):
        base = root / str(subject_id)
        mels, labels, valid = base / "mels.npy", base / "labels.npy", base / "valid.npy"
        if not (mels.is_file() and labels.is_file() and valid.is_file()):
            raise FileNotFoundError(f"Missing cache files for {subject_id}.")
        mel_array = np.load(mels, mmap_mode="r")
        label_array = np.load(labels, mmap_mode="r")
        valid_array = np.load(valid, mmap_mode="r")
        if mel_array.ndim != 3 or mel_array.shape[1:] != (48, 1499):
            raise ValueError(f"{subject_id}: expected [T,48,1499], got {mel_array.shape}.")
        if len(mel_array) != len(label_array) or len(mel_array) != len(valid_array):
            raise ValueError(f"{subject_id}: cache array length mismatch.")
        indexes = rows.subject_epoch_index.to_numpy(dtype=np.int64)
        if indexes.min() < 0 or indexes.max() >= len(mel_array) or len(np.unique(indexes)) != len(indexes):
            raise ValueError(f"{subject_id}: invalid manifest epoch indexes.")
        sample = np.asarray(mel_array[indexes[: min(4, len(indexes))]], dtype=np.float32)
        if not np.isfinite(sample).all():
            raise ValueError(f"{subject_id}: non-finite Mel values.")
    report["test_available"] = bool(groups["test"])
    return report


class CachedMelEpochDataset(Dataset):
    def __init__(self, manifest_path: str | Path, cache_root: str | Path, stats_path: str | Path, split: str, subject_limit: int | None = None):
        if split not in {"train", "val"}:
            raise ValueError("Only train and val are permitted.")
        self.frame = _subject_rows(read_manifest(manifest_path), split, subject_limit)
        self.root = Path(cache_root)
        self.mean, self.std = train_stats(stats_path)
        self._mels: dict[str, np.ndarray] = {}
        self._labels: dict[str, np.ndarray] = {}
        self._valid: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.frame)

    def _arrays(self, subject_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if subject_id not in self._mels:
            base = self.root / subject_id
            self._mels[subject_id] = np.load(base / "mels.npy", mmap_mode="r")
            self._labels[subject_id] = np.load(base / "labels.npy", mmap_mode="r")
            self._valid[subject_id] = np.load(base / "valid.npy", mmap_mode="r")
        return self._mels[subject_id], self._labels[subject_id], self._valid[subject_id]

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        subject_id, epoch = str(row.subject_id), int(row.subject_epoch_index)
        mels, labels, valid = self._arrays(subject_id)
        if epoch >= len(mels):
            raise IndexError(f"{subject_id}: epoch {epoch} outside cache.")
        x = torch.from_numpy(np.asarray(mels[epoch], dtype=np.float32).copy()).unsqueeze(0)
        x = (x - self.mean) / self.std
        y = torch.tensor(int(labels[epoch]), dtype=torch.long)
        y = remap_legacy_labels(y)
        is_valid = bool(valid[epoch])
        if not is_valid:
            y = torch.tensor(-100, dtype=torch.long)
        return x, y, subject_id, epoch, is_valid

    def labels(self) -> np.ndarray:
        values = []
        for index in range(len(self)):
            _, label, _, _, valid = self[index]
            if valid:
                values.append(int(label))
        return np.asarray(values, dtype=np.int64)


@dataclass(frozen=True)
class WindowRef:
    subject_id: str
    target_start: int


class EmbeddingWindowDataset(Dataset):
    def __init__(self, manifest_path: str | Path, embedding_root: str | Path, split: str, input_epochs: int = 40, target_epochs: int = 20, stride: int = 20, subject_limit: int | None = None):
        if input_epochs != 40 or target_epochs != 20:
            raise ValueError("B1 v1 is intentionally fixed at 40 input / 20 target epochs.")
        self.frame = _subject_rows(read_manifest(manifest_path), split, subject_limit)
        self.root, self.split = Path(embedding_root), split
        self._arrays: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self.groups: dict[str, int] = {}
        for subject_id, rows in self.frame.groupby("subject_id", sort=True):
            sid = str(subject_id)
            indexes = rows.subject_epoch_index.to_numpy(dtype=np.int64)
            if len(indexes) != len(np.unique(indexes)) or np.any(np.diff(indexes) <= 0):
                raise ValueError(f"{sid}: subject_epoch_index must be strictly increasing and unique.")
            embeddings, _, valid = self._load(sid)
            if indexes.min() < 0 or indexes.max() >= len(embeddings):
                raise ValueError(f"{sid}: manifest epoch indexes outside embedding cache.")
            # Retain the original epoch axis: absent manifest epochs stay invalid,
            # rather than being compressed into artificial temporal neighbours.
            # A manifest row can itself be unscored, so its cache validity remains
            # the authority for target masking.
            self.groups[sid] = len(embeddings)
        self.windows = [WindowRef(sid, start) for sid, total in self.groups.items() for start in range(0, total, stride)]

    def __len__(self) -> int:
        return len(self.windows)

    def _load(self, subject_id: str):
        if subject_id not in self._arrays:
            base = self.root / subject_id
            embeddings = np.load(base / "embeddings.npy", mmap_mode="r")
            labels = np.load(base / "labels.npy", mmap_mode="r")
            valid = np.load(base / "valid.npy", mmap_mode="r")
            if embeddings.ndim != 2 or embeddings.shape[1] != 192:
                raise ValueError(f"{subject_id}: embedding shape must be [T,192].")
            if not (len(embeddings) == len(labels) == len(valid)):
                raise ValueError(f"{subject_id}: embedding cache length mismatch.")
            if not np.isfinite(np.asarray(embeddings[: min(4, len(embeddings))], dtype=np.float32)).all():
                raise ValueError(f"{subject_id}: non-finite embedding values.")
            self._arrays[subject_id] = embeddings, labels, valid
        return self._arrays[subject_id]

    def __getitem__(self, index: int):
        ref = self.windows[index]
        embeddings, labels, valid = self._load(ref.subject_id)
        total = len(embeddings)
        x = np.zeros((40, 192), np.float32)
        y = np.full(20, -100, np.int64)
        input_valid = np.zeros(40, bool)
        target_valid = np.zeros(20, bool)
        source_start, source_end = max(0, ref.target_start - 10), min(total, ref.target_start + 30)
        destination = source_start - (ref.target_start - 10)
        count = source_end - source_start
        x[destination:destination + count] = embeddings[source_start:source_end]
        input_valid[destination:destination + count] = valid[source_start:source_end]
        target_end = min(total, ref.target_start + 20)
        count = target_end - ref.target_start
        y[:count] = labels[ref.target_start:target_end]
        target_valid[:count] = valid[ref.target_start:target_end]
        target_indexes = np.full(20, -1, np.int64)
        target_indexes[:count] = np.arange(ref.target_start, target_end)
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(input_valid), torch.from_numpy(target_valid), ref.subject_id, torch.from_numpy(target_indexes)
