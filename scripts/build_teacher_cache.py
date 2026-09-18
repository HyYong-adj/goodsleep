"""PSG teacher logits 를 canonical 클래스 순서 + 실제 epoch 시간축으로 옮긴다.

원본(`byoungjun/cache/teacher_targets`)은 legacy 순서(Wake, Light, Deep, REM)이고
**캐시 축**(subject_epoch_index)에 색인돼 있다. B1 은 canonical 순서(Wake, REM, Light, Deep)와
**실제 30초 시간축**(epoch_index)을 쓰므로 둘 다 변환해야 한다. 변환 없이 쓰면
KD 가 엉뚱한 클래스를 가르치게 되고 조용히 망가진다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from psg_only.constants import CANONICAL_CLASSES, CANONICAL_LOGITS_FROM_LEGACY, CLASS_ORDER_VERSION  # noqa: E402
from psg_only.data import read_manifest  # noqa: E402

SOURCE = Path("/home/sleep/researchers/byoungjun/cache/teacher_targets")
MANIFEST = ROOT / "artifacts/experiments/20260914T062144930455Z/manifest.csv"
MEL_CACHE = Path("/home/sleep/researchers/byoungjun/cache/audio_compact_full")


def main() -> None:
    output = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "artifacts/teacher_canonical")
    output.mkdir(parents=True, exist_ok=True)
    frame = read_manifest(MANIFEST)
    agreements, total = [], 0
    for sid, rows in frame.groupby("subject_id", sort=True):
        sid = str(sid)
        legacy = np.load(SOURCE / sid / "logits.npy").astype(np.float32)
        source_valid = np.load(MEL_CACHE / sid / "valid.npy")
        cache_idx = rows.subject_epoch_index.to_numpy()
        epoch_idx = rows.epoch_index.to_numpy()
        if legacy.shape[0] != len(source_valid):
            raise ValueError(f"{sid}: teacher axis {legacy.shape[0]} != cache {len(source_valid)}")
        canonical_logits = legacy[:, list(CANONICAL_LOGITS_FROM_LEGACY)]

        span = int(epoch_idx.max()) + 1
        out_logits = np.zeros((span, 4), dtype=np.float32)
        out_valid = np.zeros(span, dtype=bool)
        out_logits[epoch_idx] = canonical_logits[cache_idx]
        out_valid[epoch_idx] = source_valid[cache_idx]

        # 무결성: teacher argmax 가 canonical 정답과 얼마나 맞는지. 순서를 틀리면 급락한다.
        targets = np.asarray((0, 2, 3, 1))[rows.label_index.to_numpy()]
        keep = source_valid[cache_idx]
        if keep.any():
            agreements.append(float((canonical_logits[cache_idx][keep].argmax(1) == targets[keep]).mean()))
            total += int(keep.sum())

        subject_dir = output / sid
        subject_dir.mkdir(exist_ok=True)
        np.save(subject_dir / "logits.npy", out_logits)
        np.save(subject_dir / "valid.npy", out_valid)

    agreement = float(np.mean(agreements))
    (output / "metadata.json").write_text(json.dumps({
        "source": str(SOURCE), "class_order": list(CANONICAL_CLASSES),
        "class_order_version": CLASS_ORDER_VERSION, "axis": "epoch_index (real 30s grid)",
        "reorder": "legacy(Wake,Light,Deep,REM) -> canonical via CANONICAL_LOGITS_FROM_LEGACY",
        "subjects": len(agreements), "valid_epochs": total,
        "teacher_argmax_agreement_with_labels": agreement,
    }, indent=2))
    print(json.dumps({"subjects": len(agreements), "valid_epochs": total,
                      "teacher_agreement": round(agreement, 4)}))
    if agreement < 0.7:
        raise SystemExit(f"Teacher agreement {agreement:.3f} is implausibly low — class order is probably wrong.")


if __name__ == "__main__":
    main()
