"""choihy-e026: e022 의 피험자 내 REM/Light/Deep 호흡특징 분리 검사를 SNUBH 에 AHI 층별로 재적용한다 (CPU).

e022 는 PSG-Audio(AHI 중앙값 54)에서 REM-Light 가 어떤 포락선 특징으로도 갈라지지 않음을 보였고,
"중증 OSA 에서 Light 도 무호흡으로 조각나기 때문" 이라고 읽었다. SNUBH 는 AHI normal 22 / mild ~11 녹화를
포함하므로, 이 읽기가 맞다면 **경증·정상 층에서 REM-Light 효과가 커져야** 한다.

평가 전용 범위: 라벨별 기술통계(피험자 내 paired 효과)만 낸다. SNUBH 에 어떤 모델·probe 도 적합하지 않는다.
PSG-Audio 쪽은 e022 의 features.csv 를 그대로 쓰고 byoungjun `manifests/severity/design.json` 의
AHI<30 목록(32명)으로 층을 나눈다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from frontend_probe import CLASSES, FEATURES, epoch_features  # noqa: E402

SNUBH = Path("/home/sleep/data/marg_psg_audio/_derived/snubh_psg")
CACHE = SNUBH / "cache/audio_compact_v1"
BUNDLE = SNUBH / "bundles/snubh_psg_external_v1"
PSG_FEATURES = ROOT / "artifacts/experiments/e022_frontend_probe/features.csv"
SEVERITY = Path("/home/sleep/researchers/byoungjun/manifests/severity/design.json")
OUT = ROOT / "artifacts/experiments/e026_snubh_feature_probe"
PAIRS = (("REM", "Light"), ("Deep", "Light"), ("REM", "Deep"), ("Wake", "Light"))
HEADLINE = ("mod_peak_ratio", "mod_breath_frac", "acf_peak", "env_cv", "env_dyn_range", "centroid_bin")


def snubh_features() -> pd.DataFrame:
    path = OUT / "snubh_features.csv"
    if path.is_file():
        return pd.read_csv(path, dtype={"subject_id": str})
    manifest = pd.read_csv(CACHE / "manifest.csv", dtype={"subject_id": str})
    rows = []
    for sid, sub in manifest.groupby("subject_id", sort=True):
        mels = np.load(CACHE / sid / "mels.npy", mmap_mode="r")
        valid = np.load(CACHE / sid / "valid.npy")
        idx = sub.subject_epoch_index.to_numpy()
        keep = valid[idx]
        labels = np.asarray((0, 2, 3, 1))[sub.label_index.to_numpy()[keep]]
        idx = idx[keep]
        feats = np.concatenate([epoch_features(np.asarray(mels[idx[i:i + 200]])) for i in range(0, len(idx), 200)])
        frame = pd.DataFrame(feats, columns=FEATURES)
        frame["subject_id"], frame["label"] = sid, labels
        rows.append(frame)
        print(json.dumps({"stage": "features", "subject": sid, "epochs": int(len(idx))}), flush=True)
    data = pd.concat(rows, ignore_index=True)
    data.to_csv(path, index=False)
    return data


def paired(data: pd.DataFrame, subjects) -> dict:
    data = data[data.subject_id.isin(set(subjects))]
    result = {}
    for target, ref in PAIRS:
        ti, ri = CLASSES.index(target), CLASSES.index(ref)
        per_feature = {}
        for feature in FEATURES:
            deltas = []
            for _, sub in data.groupby("subject_id"):
                a, b = sub[sub.label == ti][feature], sub[sub.label == ri][feature]
                if len(a) >= 10 and len(b) >= 10:
                    spread = np.median(np.abs(b - b.median())) * 1.4826 + 1e-9
                    deltas.append((a.median() - b.median()) / spread)
            deltas = np.asarray(deltas)
            if not len(deltas):
                continue
            positive = int((deltas > 0).sum())
            per_feature[feature] = {"subjects": int(len(deltas)), "median_effect": float(np.median(deltas)),
                                    "positive": positive, "sign_p": float(binomtest(positive, len(deltas), 0.5).pvalue)}
        result[f"{target}-{ref}"] = per_feature
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    snubh = snubh_features()
    meta = pd.read_csv(BUNDLE / "epochs.csv", dtype={"subject_id": str},
                       usecols=["subject_id", "ahi_class", "age_band"]).groupby("subject_id").first()
    adult = meta[meta.age_band != "<18"]
    snubh_groups = {"all": meta.index,
                    "adult_ahi_lt15 (normal+mild)": adult.index[adult.ahi_class.isin(["normal", "mild"])],
                    "adult_ahi_15_30 (moderate)": adult.index[adult.ahi_class == "moderate"],
                    "adult_ahi_ge30 (severe)": adult.index[adult.ahi_class == "severe"]}
    psg = pd.read_csv(PSG_FEATURES, dtype={"subject_id": str})
    mild = set(json.loads(SEVERITY.read_text())["mild_ahi_lt30"])
    psg_ids = psg.subject_id.unique()
    psg_groups = {"all": psg_ids, "ahi_lt30": [s for s in psg_ids if s in mild],
                  "ahi_ge30": [s for s in psg_ids if s not in mild]}

    results = {"features": FEATURES, "classes": CLASSES,
               "snubh": {name: {"recordings": int(len(ids)), "paired": paired(snubh, ids)} for name, ids in snubh_groups.items()},
               "psg_audio_e022_all_splits": {name: {"subjects": int(len(ids)), "paired": paired(psg, ids)} for name, ids in psg_groups.items()},
               "class_fraction": {"snubh": snubh.label.value_counts(normalize=True).sort_index().tolist(),
                                  "psg_audio": psg.label.value_counts(normalize=True).sort_index().tolist()},
               "note": "PSG-Audio AHI<30 list = byoungjun severity design (32 subj); SNUBH ahi_class from bundle (ahi_est, manual events / TST). Pediatric excluded from SNUBH strata."}
    (OUT / "results.json").write_text(json.dumps(results, indent=2))

    lines = []
    for dataset, groups in (("SNUBH", results["snubh"]), ("PSG-Audio", results["psg_audio_e022_all_splits"])):
        for name, block in groups.items():
            for pair in ("REM-Light", "Deep-Light"):
                cells = [f"{f} {block['paired'][pair][f]['median_effect']:+.2f} ({block['paired'][pair][f]['positive']}/{block['paired'][pair][f]['subjects']})"
                         for f in HEADLINE if f in block["paired"].get(pair, {})]
                lines.append(f"{dataset:9s} | {name:30s} | {pair:10s} | " + "; ".join(cells))
    (OUT / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
