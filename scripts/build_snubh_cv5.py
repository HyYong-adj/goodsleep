"""SNUBH 기본 뷰(66 녹화 / 58명) 참가자 단위 5-fold 분할 + fold-train log-mel 통계.

- 분할 단위는 participant(split-night PSG/CPAP, 재검사 녹화가 같은 fold). 진단 AHI 등급
  (`participants.csv` ahi_class) 층화, 등급 안에서 participant_id 정렬 후 순환 배정 → 결정적.
- fold k 는 PSG-Audio `byoungjun/manifests/cv5/fold{k}.csv` 와 짝을 이룬다.
- 매니페스트는 내 로더 계약에 맞춰 `recording_start_seconds` 를 뺀다(ANALYSIS-START 격자).
- 정규화 통계는 fold train 녹화의 `stats.json`(sum/sq_sum/count)만으로 계산한다.
공유 번들·캐시는 읽기만 한다.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path("/home/sleep/data/marg_psg_audio/_derived/snubh_psg/cache/audio_compact_v1")
PARTICIPANTS = Path("/home/sleep/researchers/pbj/external_manifests/snubh_psg/participants.csv")
OUT = ROOT / "artifacts/manifests/snubh_cv5"
K = 5


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(CACHE / "manifest.csv", dtype={"subject_id": str}).drop(columns=["recording_start_seconds"])
    people = pd.read_csv(PARTICIPANTS).set_index("participant_id")
    ids = sorted(frame.participant_id.unique())
    fold_of = {}
    offset = 0
    for grade in ("normal", "mild", "moderate", "severe"):
        members = sorted((p for p in ids if people.loc[p, "ahi_class"] == grade),
                         key=lambda p: hashlib.sha256(f"snubh-cv5-{p}".encode()).hexdigest())
        for i, p in enumerate(members):
            fold_of[p] = (offset + i) % K
        offset += len(members)
    assert set(fold_of) == set(ids)
    summary = {}
    for k in range(K):
        part = frame.assign(split=np.where(frame.participant_id.map(fold_of) == k, "val", "train"))
        part.to_csv(OUT / f"fold{k}.csv", index=False)
        total = squares = count = 0.0
        for sid in part[part.split == "train"].subject_id.unique():
            s = json.loads((CACHE / sid / "stats.json").read_text())
            total += s["sum"]; squares += s["sq_sum"]; count += s["count"]
        mean = total / count
        std = float(np.sqrt(squares / count - mean * mean))
        (OUT / f"fold{k}_train_stats.json").write_text(json.dumps({"mean": mean, "std": std, "source": f"snubh cv5 fold{k} train"}))
        val = part[part.split == "val"]
        summary[k] = {"val_participants": int(val.participant_id.nunique()), "val_recordings": int(val.subject_id.nunique()),
                      "val_epochs": int(len(val)), "val_rem_frac": round(float((val.label_index == 3).mean()), 4),
                      "val_ahi": people.loc[val.participant_id.unique(), "ahi_class"].value_counts().to_dict(),
                      "train_mean": round(mean, 3), "train_std": round(std, 3)}
    (OUT / "folds.json").write_text(json.dumps({"fold_of_participant": fold_of, "summary": summary}, indent=2))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
