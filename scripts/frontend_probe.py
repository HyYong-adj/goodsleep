"""choihy-e022: front-end 병목 진단 (CPU).

기존 log-Mel 캐시에서 호흡 리듬·포락선·대역 에너지 특징 12개를 뽑아
(1) 같은 피험자 안에서 REM/Light/Deep 이 갈라지는지, (2) 그 특징만으로 선형 분류기가
REM 을 얼마나 맞히는지 본다. GPU 실험(e023) 을 할지 말지 여기서 결정한다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from psg_only.data import read_manifest  # noqa: E402

CACHE = Path("/home/sleep/researchers/byoungjun/cache/audio_compact_full")
MANIFEST = ROOT / "artifacts/experiments/20260914T062144930455Z/manifest.csv"
OUT = ROOT / "artifacts/experiments/e022_frontend_probe"
FS = 1499 / 30.0  # envelope frame rate, Hz
CLASSES = ("Wake", "REM", "Light", "Deep")
FEATURES = ["env_cv", "env_dyn_range", "mod_peak_hz", "mod_peak_ratio", "mod_breath_frac",
            "acf_peak", "acf_peak_lag_s", "low_frac", "mid_frac", "high_frac", "high_env_cv", "centroid_bin"]


def epoch_features(logmel: np.ndarray) -> np.ndarray:
    """logmel [T,48,1499] -> [T,12]. 벡터화."""
    power = np.power(10.0, logmel.astype(np.float32) / 10.0)          # linear energy
    env = power.mean(1)                                                # [T,1499] broadband envelope
    mean = env.mean(1, keepdims=True) + 1e-12
    env_cv = env.std(1) / mean[:, 0]
    p5, p95 = np.percentile(env, [5, 95], axis=1)
    env_dyn_range = np.log10((p95 + 1e-12) / (p5 + 1e-12))

    # modulation spectrum of the mean-removed, Hann-windowed envelope
    centred = (env - mean) * np.hanning(env.shape[1])[None, :]
    spectrum = np.abs(np.fft.rfft(centred, axis=1)) ** 2
    freqs = np.fft.rfftfreq(env.shape[1], d=1.0 / FS)
    band = (freqs >= 0.1) & (freqs <= 2.0)
    breath = (freqs >= 0.15) & (freqs <= 0.6)
    band_power = spectrum[:, band]
    mod_peak_hz = freqs[band][band_power.argmax(1)]
    mod_peak_ratio = band_power.max(1) / (band_power.mean(1) + 1e-12)
    mod_breath_frac = spectrum[:, breath].sum(1) / (band_power.sum(1) + 1e-12)

    # envelope autocorrelation, lag 1.5-8 s (breath period / apnea cycle)
    n = env.shape[1]
    fft = np.fft.rfft(env - mean, n=2 * n, axis=1)
    acf = np.fft.irfft(fft * np.conj(fft), n=2 * n, axis=1)[:, :n]
    acf = acf / (acf[:, :1] + 1e-12)
    lo, hi = int(1.5 * FS), int(8.0 * FS)
    acf_peak = acf[:, lo:hi].max(1)
    acf_peak_lag_s = (acf[:, lo:hi].argmax(1) + lo) / FS

    # band energy fractions (thirds of the 48 mel bins) and high-band burstiness
    total = power.sum((1, 2)) + 1e-12
    low_frac = power[:, :16].sum((1, 2)) / total
    mid_frac = power[:, 16:32].sum((1, 2)) / total
    high_frac = power[:, 32:].sum((1, 2)) / total
    high_env = power[:, 32:].mean(1)
    high_env_cv = high_env.std(1) / (high_env.mean(1) + 1e-12)
    bins = np.arange(48, dtype=np.float32)
    centroid_bin = (power.sum(2) * bins).sum(1) / (power.sum((1, 2)) + 1e-12)

    return np.stack([env_cv, env_dyn_range, mod_peak_hz, mod_peak_ratio, mod_breath_frac,
                     acf_peak, acf_peak_lag_s, low_frac, mid_frac, high_frac, high_env_cv, centroid_bin], 1)


def main() -> None:
    frame = read_manifest(MANIFEST)
    rows = []
    for sid, sub in frame.groupby("subject_id", sort=True):
        sid = str(sid)
        mels = np.load(CACHE / sid / "mels.npy", mmap_mode="r")
        valid = np.load(CACHE / sid / "valid.npy")
        idx = sub.subject_epoch_index.to_numpy()
        keep = valid[idx]
        idx = idx[keep]
        if len(idx) == 0:
            continue
        feats = epoch_features(np.asarray(mels[idx]))
        labels = np.asarray((0, 2, 3, 1))[sub.label_index.to_numpy()[keep]]
        df = pd.DataFrame(feats, columns=FEATURES)
        df["subject_id"] = sid
        df["split"] = sub.split.iloc[0]
        df["label"] = labels
        rows.append(df)
        print(json.dumps({"stage": "features", "subject": sid, "epochs": int(len(idx))}), flush=True)
    data = pd.concat(rows, ignore_index=True)
    data.to_csv(OUT / "features.csv", index=False)

    # (1) within-subject paired separation, train split only
    train = data[data.split == "train"]
    paired = {}
    for target, ref in (("REM", "Light"), ("Deep", "Light"), ("REM", "Deep"), ("Wake", "Light")):
        ti, ri = CLASSES.index(target), CLASSES.index(ref)
        per_feature = {}
        for feature in FEATURES:
            deltas = []
            for sid, sub in train.groupby("subject_id"):
                a, b = sub[sub.label == ti][feature], sub[sub.label == ri][feature]
                if len(a) >= 10 and len(b) >= 10:
                    spread = np.median(np.abs(b - b.median())) * 1.4826 + 1e-9  # robust sd of ref
                    deltas.append((a.median() - b.median()) / spread)
            deltas = np.asarray(deltas)
            if len(deltas) == 0:
                continue
            positive = int((deltas > 0).sum())
            from scipy.stats import binomtest
            p = binomtest(positive, len(deltas), 0.5).pvalue
            per_feature[feature] = {"subjects": int(len(deltas)), "median_effect": float(np.median(deltas)),
                                    "positive": positive, "sign_p": float(p)}
        paired[f"{target}-{ref}"] = per_feature

    # (2) class-balanced linear probe, train -> val, features z-scored on train
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import cohen_kappa_score, f1_score
    from sklearn.preprocessing import StandardScaler
    val = data[data.split == "val"]
    scaler = StandardScaler().fit(train[FEATURES])
    probe = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    probe.fit(scaler.transform(train[FEATURES]), train.label)
    pred = probe.predict(scaler.transform(val[FEATURES]))
    per_class = f1_score(val.label, pred, labels=[0, 1, 2, 3], average=None, zero_division=0)
    probe_result = {"macro_f1": float(per_class.mean()), "kappa": float(cohen_kappa_score(val.label, pred)),
                    "accuracy": float((pred == val.label).mean()),
                    "per_class_f1": dict(zip(CLASSES, map(float, per_class))),
                    "n_val_epochs": int(len(val))}

    # (2b) same probe with per-subject z-scoring of features (night-normalised features)
    def subject_z(df):
        return df.groupby("subject_id")[FEATURES].transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
    tz, vz = subject_z(train), subject_z(val)
    probe2 = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0).fit(tz, train.label)
    pred2 = probe2.predict(vz)
    pc2 = f1_score(val.label, pred2, labels=[0, 1, 2, 3], average=None, zero_division=0)
    probe_subject_z = {"macro_f1": float(pc2.mean()), "kappa": float(cohen_kappa_score(val.label, pred2)),
                       "per_class_f1": dict(zip(CLASSES, map(float, pc2)))}

    results = {"features": FEATURES, "train_epochs": int(len(train)), "val_epochs": int(len(val)),
               "paired_within_subject": paired, "linear_probe_val": probe_result,
               "linear_probe_val_subject_znorm": probe_subject_z,
               "reference_deep_model_val": {"macro_f1": 0.5310, "REM_f1": 0.268, "Deep_f1": 0.399,
                                            "note": "choihy-e012, same fixed val-44"}}
    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({"stage": "complete"}), flush=True)


if __name__ == "__main__":
    main()
