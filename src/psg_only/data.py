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

from .constants import CANONICAL_CLASSES, CLASS_ORDER_VERSION, remap_legacy_labels
from .provenance import file_hash
from .windows import context_options

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
    if frame[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Manifest contains null required values.")
    for column in ("subject_epoch_index", "label_index"):
        values = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(values).all() or not (values == np.floor(values)).all():
            raise ValueError(f"Non-integer {column}.")
        frame[column] = values.astype(np.int64)
    if (frame.subject_epoch_index < 0).any() or not frame.label_index.between(0, 3).all():
        raise ValueError("Invalid manifest index or legacy label.")
    if frame.groupby("subject_id").split.nunique().gt(1).any():
        raise ValueError("Subject leakage across splits.")
    if frame.duplicated(["subject_id", "subject_epoch_index"]).any():
        raise ValueError("Duplicate manifest epoch.")
    # Cache lookup index and real 30-second recording index are different axes.
    if "recording_start_seconds" in frame:
        epoch = pd.to_numeric(frame.recording_start_seconds) / 30
        if not np.isfinite(epoch).all() or (epoch < 0).any() or not np.allclose(epoch, np.rint(epoch), rtol=0, atol=1e-7):
            raise ValueError("Recording timestamps must lie on the 30-second grid.")
        frame["epoch_index"] = np.rint(epoch).astype(np.int64)
    else:
        frame["epoch_index"] = frame.subject_epoch_index
    for _, rows in frame.groupby("subject_id"):
        if np.any(np.diff(rows.sort_values("subject_epoch_index").epoch_index) <= 0):
            raise ValueError("Non-monotonic recording timeline.")
    return frame


def train_stats(path: str | Path) -> tuple[float, float]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    mean, std = float(payload["mean"]), float(payload["std"])
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
        raise ValueError("Invalid train normalization statistics.")
    return mean, std


def night_statistics(subject_root: str | Path) -> tuple[float, float]:
    """피험자(밤) 단위 log-Mel mean/std.

    마이크 감도·방 소음·환자 거리 차이를 제거한다. 본 데이터의 피험자별 평균은
    -68.4 ~ -26.8 (범위 41.6 dB)로, 전역 통계 하나로는 흡수되지 않는 변량이다.

    소스 캐시에 미리 계산된 ``night_stats.json``이 있으면 재사용한다. 없으면
    유효 epoch 에서 직접 계산하되 **소스 캐시에 쓰지 않는다** (타 연구자 소유).
    라벨을 쓰지 않는 입력 신호만의 정규화이며, 밤 전체를 요구하므로 실시간
    경로에는 쓸 수 없다. 현재 제품 정의(아침 밤 전체 그래프, 비인과 허용)에서는 허용된다.
    """
    root = Path(subject_root)
    cached = root / "night_stats.json"
    if cached.is_file():
        payload = json.loads(cached.read_text())
        mean, std = float(payload["mean"]), float(payload["std"])
    else:
        mels = np.load(root / "mels.npy", mmap_mode="r")
        valid = np.load(root / "valid.npy")
        total = squares = count = 0.0
        for index in np.flatnonzero(valid):
            block = np.asarray(mels[index], dtype=np.float64)
            total += block.sum()
            squares += np.square(block).sum()
            count += block.size
        if not count:
            raise ValueError(f"{root.name}: no valid epochs for night statistics.")
        mean = total / count
        std = float(np.sqrt(max(squares / count - mean * mean, 0.0)))
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
        raise ValueError(f"{root.name}: invalid night statistics.")
    return mean, std


def _subject_rows(frame: pd.DataFrame, split: str, subject_limit: int | None = None) -> pd.DataFrame:
    if split not in {"train", "val"}:
        raise ValueError("Only train and val are permitted.")
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
        if label_array.shape != (len(mel_array),) or valid_array.shape != (len(mel_array),) or valid_array.dtype != np.bool_:
            raise ValueError("Labels/valid must be 1D and valid must be bool.")
        indexes = rows.subject_epoch_index.to_numpy(dtype=np.int64)
        if indexes.min() < 0 or indexes.max() >= len(mel_array) or len(np.unique(indexes)) != len(indexes):
            raise ValueError(f"{subject_id}: invalid manifest epoch indexes.")
        if not np.array_equal(label_array[indexes], rows.label_index.to_numpy()):
            raise ValueError("Manifest/cache label mismatch.")
        sample = np.asarray(mel_array[indexes[: min(4, len(indexes))]], dtype=np.float32)
        if not np.isfinite(sample).all():
            raise ValueError(f"{subject_id}: non-finite Mel values.")
    report["valid_class_counts"] = {}
    for split in ("train", "val"):
        counts = np.zeros(4, dtype=np.int64)
        for sid, rows in frame.loc[frame.split == split].groupby("subject_id"):
            idx = rows.subject_epoch_index.to_numpy()
            valid = np.load(root / sid / "valid.npy", mmap_mode="r")[idx]
            canonical = np.asarray((0, 2, 3, 1))[rows.label_index.to_numpy()[valid]]
            counts += np.bincount(canonical, minlength=4)
        if (counts == 0).any():
            raise ValueError(f"Missing valid training/evaluation class in {split}.")
        report["valid_class_counts"][split] = dict(zip(CANONICAL_CLASSES, counts.tolist()))
    report["test_available"] = bool(groups["test"])
    return report


class CachedMelEpochDataset(Dataset):
    def __init__(self, manifest_path: str | Path, cache_root: str | Path, stats_path: str | Path, split: str, subject_limit: int | None = None, night_norm: bool = False):
        if split not in {"train", "val"}:
            raise ValueError("Only train and val are permitted.")
        self.frame = _subject_rows(read_manifest(manifest_path), split, subject_limit)
        self.root = Path(cache_root)
        self.night_norm = bool(night_norm)
        self._night_stats: dict[str, tuple[float, float]] = {}
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
        if epoch < 0 or epoch >= len(mels):
            raise IndexError(f"{subject_id}: epoch {epoch} outside cache.")
        if int(labels[epoch]) != int(row.label_index):
            raise ValueError("Manifest/cache label mismatch.")
        x = torch.from_numpy(np.asarray(mels[epoch], dtype=np.float32).copy()).unsqueeze(0)
        mean, std = self.norm_stats(subject_id)
        x = (x - mean) / std
        y = torch.tensor(int(labels[epoch]), dtype=torch.long)
        y = remap_legacy_labels(y)
        is_valid = bool(valid[epoch])
        if not is_valid:
            y = torch.tensor(-100, dtype=torch.long)
        return x, y, subject_id, int(row.epoch_index), is_valid

    def norm_stats(self, subject_id: str) -> tuple[float, float]:
        """C3(night_norm) 이면 피험자별, 아니면 기존 train 전역 통계."""
        if not self.night_norm:
            return self.mean, self.std
        if subject_id not in self._night_stats:
            self._night_stats[subject_id] = night_statistics(self.root / subject_id)
        return self._night_stats[subject_id]

    def labels(self) -> np.ndarray:
        values = []
        for sid, rows in self.frame.groupby("subject_id"):
            _, labels, valid = self._arrays(sid)
            idx = rows.subject_epoch_index.to_numpy()
            if not np.array_equal(labels[idx], rows.label_index.to_numpy()):
                raise ValueError("Manifest/cache label mismatch.")
            values.extend(np.asarray((0, 2, 3, 1))[labels[idx][valid[idx]]].tolist())
        return np.asarray(values, dtype=np.int64)

    def expected_keys(self):
        return {(sid, int(epoch)) for sid, rows in self.frame.groupby("subject_id")
                for epoch in rows.epoch_index.to_numpy()[self._arrays(sid)[2][rows.subject_epoch_index.to_numpy()]]}



@dataclass(frozen=True)
class WindowRef:
    subject_id: str
    target_start: int


class MelWindowDataset(Dataset):
    """Phase 3: 동결 embedding 대신 **Mel 시퀀스**를 직접 돌려준다.

    윈도우 규칙은 EmbeddingWindowDataset 과 같다(실제 30초 격자, 중앙 20 target, stride 20).
    다만 Mel 캐시는 ``subject_epoch_index``(캐시 축)로 색인되고 시간축은 ``epoch_index`` 이므로,
    피험자마다 epoch_index -> cache_index 매핑을 만들어 둔다. manifest 에 없는 epoch 은
    -1 로 남겨 항상 invalid 로 취급한다.
    """

    def __init__(self, manifest_path, cache_root, stats_path, split, input_epochs=40,
                 target_epochs=20, stride=20, subject_limit=None, night_norm=False):
        context_options(dict(input_epochs=input_epochs, target_epochs=target_epochs, stride=stride))
        self.input_epochs, self.target_epochs = input_epochs, target_epochs
        self.left_context = (input_epochs - target_epochs) // 2
        self.frame = _subject_rows(read_manifest(manifest_path), split, subject_limit)
        self.root = Path(cache_root)
        self.night_norm = bool(night_norm)
        self.mean, self.std = train_stats(stats_path)
        self._night_stats: dict[str, tuple[float, float]] = {}
        self._mels: dict[str, np.ndarray] = {}
        self.index: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for subject_id, rows in self.frame.groupby("subject_id", sort=True):
            sid = str(subject_id)
            epoch_idx = rows.epoch_index.to_numpy(dtype=np.int64)
            cache_idx = rows.subject_epoch_index.to_numpy(dtype=np.int64)
            if np.any(np.diff(epoch_idx) <= 0):
                raise ValueError(f"{sid}: epoch_index must be strictly increasing.")
            source_valid = np.load(self.root / sid / "valid.npy", mmap_mode="r")
            total = int(epoch_idx.max()) + 1
            mapping = np.full(total, -1, dtype=np.int64)
            labels = np.full(total, -100, dtype=np.int64)
            valid = np.zeros(total, dtype=bool)
            mapping[epoch_idx] = cache_idx
            valid[epoch_idx] = np.asarray(source_valid)[cache_idx]
            labels[epoch_idx] = np.asarray((0, 2, 3, 1))[rows.label_index.to_numpy()]
            labels[~valid] = -100
            self.index[sid] = (mapping, labels, valid)
        self.windows = [WindowRef(sid, start)
                        for sid, (mapping, _, _) in self.index.items()
                        for start in range(0, len(mapping), stride)]

    def __len__(self):
        return len(self.windows)

    def norm_stats(self, subject_id):
        if not self.night_norm:
            return self.mean, self.std
        if subject_id not in self._night_stats:
            self._night_stats[subject_id] = night_statistics(self.root / subject_id)
        return self._night_stats[subject_id]

    def _mel_array(self, subject_id):
        if subject_id not in self._mels:
            self._mels[subject_id] = np.load(self.root / subject_id / "mels.npy", mmap_mode="r")
        return self._mels[subject_id]

    def expected_keys(self):
        return {(sid, int(i)) for sid, (_, _, valid) in self.index.items() for i in np.flatnonzero(valid)}

    def labels_flat(self):
        values = []
        for _, labels, valid in self.index.values():
            values.extend(labels[valid].tolist())
        return np.asarray(values, dtype=np.int64)

    def __getitem__(self, position):
        ref = self.windows[position]
        mapping, labels, valid = self.index[ref.subject_id]
        total = len(mapping)
        mels = self._mel_array(ref.subject_id)
        mean, std = self.norm_stats(ref.subject_id)

        x = np.zeros((self.input_epochs, 1, 48, 1499), np.float32)
        input_valid = np.zeros(self.input_epochs, bool)
        source_start = max(0, ref.target_start - self.left_context)
        source_end = min(total, ref.target_start + self.target_epochs + self.left_context)
        destination = source_start - (ref.target_start - self.left_context)
        for offset in range(source_end - source_start):
            epoch = source_start + offset
            if not valid[epoch]:
                continue
            block = np.asarray(mels[mapping[epoch]], dtype=np.float32)
            x[destination + offset, 0] = (block - mean) / std
            input_valid[destination + offset] = True

        y = np.full(self.target_epochs, -100, np.int64)
        target_valid = np.zeros(self.target_epochs, bool)
        target_indexes = np.full(self.target_epochs, -1, np.int64)
        target_end = min(total, ref.target_start + self.target_epochs)
        count = target_end - ref.target_start
        y[:count] = labels[ref.target_start:target_end]
        target_valid[:count] = valid[ref.target_start:target_end]
        target_indexes[:count] = np.arange(ref.target_start, target_end)
        return (torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(input_valid),
                torch.from_numpy(target_valid), ref.subject_id, torch.from_numpy(target_indexes))




class EmbeddingWindowDataset(Dataset):
    def __init__(self, manifest_path: str | Path, embedding_root: str | Path, split: str, input_epochs: int = 40, target_epochs: int = 20, stride: int = 20, subject_limit: int | None = None):
        context_options(dict(input_epochs=input_epochs, target_epochs=target_epochs, stride=stride))
        self.input_epochs = input_epochs
        self.target_epochs = target_epochs
        self.left_context = (input_epochs - target_epochs) // 2
        self.frame = _subject_rows(read_manifest(manifest_path), split, subject_limit)
        self.root, self.split = Path(embedding_root), split
        self._arrays: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self.groups: dict[str, int] = {}
        self.provenance = None
        self.manifest_hash = file_hash(manifest_path)
        for subject_id, rows in self.frame.groupby("subject_id", sort=True):
            sid = str(subject_id)
            indexes = rows.epoch_index.to_numpy(dtype=np.int64)
            if len(indexes) != len(np.unique(indexes)) or np.any(np.diff(indexes) <= 0):
                raise ValueError(f"{sid}: subject_epoch_index must be strictly increasing and unique.")
            embeddings, labels, valid = self._load(sid)
            if indexes.min() < 0 or indexes.max() >= len(embeddings):
                raise ValueError(f"{sid}: manifest epoch indexes outside embedding cache.")
            membership = np.zeros(len(valid), dtype=bool)
            membership[indexes] = True
            if np.any(valid & ~membership):
                raise ValueError("Embedding cache includes epochs outside manifest.")
            expected = np.asarray((0, 2, 3, 1))[rows.label_index.to_numpy()]
            if not np.array_equal(labels[indexes][valid[indexes]], expected[valid[indexes]]):
                raise ValueError("Embedding labels disagree with canonical manifest labels.")
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
            metadata = json.loads((base / "metadata.json").read_text())
            if metadata.get("class_order_version") != CLASS_ORDER_VERSION or metadata.get("class_order") != list(CANONICAL_CLASSES):
                raise ValueError("Embedding class order is missing or incompatible; regenerate cache.")
            if metadata.get("manifest_hash") != self.manifest_hash or metadata.get("subject_id") != subject_id:
                raise ValueError("Embedding manifest/subject mismatch.")
            identity = {k: metadata[k] for k in ("checkpoint_sha256", "config_hash", "manifest_hash", "stats_hash", "source_cache")}
            if self.provenance is not None and identity != self.provenance:
                raise ValueError("Mixed embedding provenance.")
            self.provenance = identity
            for name in ("embeddings.npy", "labels.npy", "valid.npy"):
                if metadata.get("files", {}).get(name) != file_hash(base / name):
                    raise ValueError("Embedding file hash mismatch; regenerate cache.")
            embeddings = np.load(base / "embeddings.npy", mmap_mode="r")
            labels = np.load(base / "labels.npy", mmap_mode="r")
            valid = np.load(base / "valid.npy", mmap_mode="r")
            if embeddings.ndim != 2 or embeddings.shape[1] != 192:
                raise ValueError(f"{subject_id}: embedding shape must be [T,192].")
            if labels.shape != (len(embeddings),) or valid.shape != (len(embeddings),) or valid.dtype != np.bool_:
                raise ValueError("Invalid embedding labels/mask shape or dtype.")
            if not (len(embeddings) == len(labels) == len(valid)):
                raise ValueError(f"{subject_id}: embedding cache length mismatch.")
            if not np.isfinite(np.asarray(embeddings, dtype=np.float32)).all():
                raise ValueError(f"{subject_id}: non-finite embedding values.")
            self._arrays[subject_id] = embeddings, labels, valid
        return self._arrays[subject_id]

    def expected_keys(self):
        return {(sid, int(i)) for sid in self.groups for i in np.flatnonzero(self._load(sid)[2])}

    def __getitem__(self, index: int):
        ref = self.windows[index]
        embeddings, labels, valid = self._load(ref.subject_id)
        total = len(embeddings)
        x = np.zeros((self.input_epochs, 192), np.float32)
        y = np.full(20, -100, np.int64)
        input_valid = np.zeros(self.input_epochs, bool)
        target_valid = np.zeros(20, bool)
        source_start, source_end = max(0, ref.target_start - self.left_context), min(total, ref.target_start + self.target_epochs + self.left_context)
        destination = source_start - (ref.target_start - self.left_context)
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
