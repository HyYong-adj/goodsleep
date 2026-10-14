"""변환 front-end(pcen / logmel_delta)의 채널별 정규화 통계를 train 유효 epoch 에서 계산한다.

log-Mel 용 train_stats.json 은 분포가 달라 재사용할 수 없다. 원칙은 같다 — train split 만 사용.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from psg_only.data import PCEN_DEFAULT, logmel_delta, pcen, read_manifest  # noqa: E402

CACHE = Path("/home/sleep/researchers/byoungjun/cache/audio_compact_full")
MANIFEST = ROOT / "artifacts/experiments/20260914T062144930455Z/manifest.csv"


def main() -> None:
    kind = sys.argv[1]
    out = Path(sys.argv[2])
    stride = int(sys.argv[3]) if len(sys.argv) > 3 else 4  # 매 4번째 유효 epoch 만 사용(통계 목적엔 충분)
    frame = read_manifest(MANIFEST)
    train = frame[frame.split == "train"]
    channels = 2 if kind == "logmel_delta" else 1
    total = np.zeros(channels, np.float64); squares = np.zeros(channels, np.float64); count = 0
    subjects = 0
    for sid, sub in train.groupby("subject_id", sort=True):
        sid = str(sid)
        mels = np.load(CACHE / sid / "mels.npy", mmap_mode="r")
        valid = np.load(CACHE / sid / "valid.npy")
        idx = sub.subject_epoch_index.to_numpy()
        idx = idx[valid[idx]][::stride]
        for i in idx:
            block = np.asarray(mels[i], dtype=np.float32)
            x = pcen(block, **PCEN_DEFAULT)[None] if kind == "pcen" else logmel_delta(block)
            total += x.reshape(channels, -1).sum(1); squares += np.square(x.reshape(channels, -1)).sum(1)
            count += x.shape[1] * x.shape[2]
        subjects += 1
        if subjects % 40 == 0:
            print(json.dumps({"subjects": subjects}), flush=True)
    mean = total / count
    std = np.sqrt(np.maximum(squares / count - mean ** 2, 1e-12))
    payload = {"frontend": kind, "channels": channels, "mean": mean.tolist(), "std": std.tolist(),
               "train_subjects": subjects, "epoch_stride": stride, "elements": int(count),
               "pcen_params": PCEN_DEFAULT if kind == "pcen" else None, "source_cache": CACHE.name}
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"stage": "complete", "mean": mean.tolist(), "std": std.tolist()}))


if __name__ == "__main__":
    main()
