"""choihy-e026 보조: REM-Light 효과의 밤 중 시간대 교란 점검 (CPU, 기술통계만).

REM 은 밤 후반에 몰려 있어, 밤새 서서히 변하는 레벨·코덱 상태가 REM-Light 차이로 보일 수 있다.
각 REM epoch 를 같은 피험자의 ±10 epoch(±5 분) 안 Light epoch 들의 중앙값과만 비교한다.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from frontend_probe import FEATURES, MANIFEST, CACHE as PSG_CACHE  # noqa: E402
from snubh_feature_probe import CACHE, OUT, PSG_FEATURES, SEVERITY  # noqa: E402

WINDOW = 10


def epoch_axis(manifest: pd.DataFrame, cache: Path) -> dict:
    axis = {}
    for sid, sub in manifest.groupby("subject_id", sort=True):
        valid = np.load(cache / str(sid) / "valid.npy")
        idx = sub.subject_epoch_index.to_numpy()
        axis[str(sid)] = idx[valid[idx]]
    return axis


def matched(features: pd.DataFrame, axis: dict, subjects) -> dict:
    out = {}
    for feature in FEATURES:
        deltas = []
        for sid in subjects:
            sub = features[features.subject_id == sid]
            t = axis[sid]
            if len(t) != len(sub):
                raise ValueError(f"{sid}: feature rows {len(sub)} != valid epochs {len(t)}")
            lab, val = sub.label.to_numpy(), sub[feature].to_numpy()
            rem, light = np.flatnonzero(lab == 1), np.flatnonzero(lab == 2)
            diffs = []
            for i in rem:
                near = light[np.abs(t[light] - t[i]) <= WINDOW]
                if len(near) >= 3:
                    diffs.append(val[i] - np.median(val[near]))
            if len(diffs) >= 10:
                ref = val[light]
                spread = np.median(np.abs(ref - np.median(ref))) * 1.4826 + 1e-9
                deltas.append(np.median(diffs) / spread)
        deltas = np.asarray(deltas)
        positive = int((deltas > 0).sum())
        out[feature] = {"subjects": len(deltas), "median_effect": float(np.median(deltas)) if len(deltas) else None,
                        "positive": positive, "sign_p": float(binomtest(positive, len(deltas), 0.5).pvalue) if len(deltas) else None}
    return out


def main():
    snubh = pd.read_csv(OUT / "snubh_features.csv", dtype={"subject_id": str})
    s_axis = epoch_axis(pd.read_csv(CACHE / "manifest.csv", dtype={"subject_id": str}), CACHE)
    psg = pd.read_csv(PSG_FEATURES, dtype={"subject_id": str})
    from psg_only.data import read_manifest
    p_axis = epoch_axis(read_manifest(MANIFEST), PSG_CACHE)
    mild = set(json.loads(SEVERITY.read_text())["mild_ahi_lt30"])
    pids = list(psg.subject_id.unique())
    result = {"window_epochs": WINDOW,
              "snubh_all": matched(snubh, s_axis, list(snubh.subject_id.unique())),
              "psg_all": matched(psg, p_axis, pids),
              "psg_ahi_lt30": matched(psg, p_axis, [s for s in pids if s in mild])}
    (OUT / "rem_light_time_matched.json").write_text(json.dumps(result, indent=2))
    for name, block in result.items():
        if name == "window_epochs":
            continue
        print(name, "; ".join(f"{f} {v['median_effect']:+.2f} ({v['positive']}/{v['subjects']}, p {v['sign_p']:.1e})"
                              for f, v in block.items() if v["subjects"]))


if __name__ == "__main__":
    main()
