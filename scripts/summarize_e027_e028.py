"""choihy-e027/e028 집계: 같은 fold 기준선과 paired 비교.

기준선
  PSG-Audio val: e018 fold k (PSG-Audio 단독 학습, PSG val 선택)
  SNUBH val    : e025 N0 제로샷 예측을 fold k val 녹화로 제한 (같은 e018 fold k 모델)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "artifacts/experiments"
FOLDS = ROOT / "artifacts/manifests/snubh_cv5"


def score(frame):
    t, p = frame.target.to_numpy(), frame.prediction.to_numpy()
    per = f1_score(t, p, labels=range(4), average=None, zero_division=0)
    return {"macro_f1": float(per.mean()), "kappa": float(cohen_kappa_score(t, p)), "rem_f1": float(per[1]),
            "deep_f1": float(per[3]), "pred_rem_frac": float((p == 1).mean())}


def baseline(k):
    psg = pd.read_csv(EXP / f"e018_cv_fold{k}/evaluation/predictions.csv")
    val = set(pd.read_csv(FOLDS / f"fold{k}.csv").query("split == 'val'").subject_id)
    zs = pd.read_csv(EXP / f"e025_snubh_external/e018_cv_fold{k}__train/predictions.csv")
    return {"psg": score(psg), "snubh": score(zs[zs.subject_id.isin(val)])}


def main():
    rows = []
    for k in range(5):
        base = baseline(k)
        for name in ("psg", "snubh"):
            rows.append({"run": "e018/e025 (PSG only, zero-shot)", "selection": "-", "fold": k, "val": name, **base[name]})
        for exp in ("e027", "e028"):
            path = EXP / f"{exp}_fold{k}/evaluation"
            if not (path / "results.json").is_file():
                continue
            result = json.loads((path / "results.json").read_text())
            for selection in result["selections"]:
                for name in ("psg", "snubh"):
                    pred = pd.read_csv(path / f"pred_{selection}_{name}.csv")
                    rows.append({"run": exp, "selection": selection, "fold": k, "val": name, **score(pred)})
    frame = pd.DataFrame(rows)
    frame.to_csv(EXP / "e027_e028_summary.csv", index=False)
    base = frame[frame.run.str.startswith("e018")].set_index(["fold", "val"])
    out = []
    for (run, selection, val), part in frame[~frame.run.str.startswith("e018")].groupby(["run", "selection", "val"]):
        delta = part.set_index("fold").macro_f1 - base.xs(val, level="val").macro_f1.reindex(part.fold)
        out.append({"run": run, "selection": selection, "val": val, "folds": len(part),
                    "macro_f1": f"{part.macro_f1.mean():.4f} ± {part.macro_f1.std():.4f}",
                    "kappa": round(part.kappa.mean(), 4), "rem_f1": round(part.rem_f1.mean(), 4),
                    "delta_vs_baseline": f"{delta.mean():+.4f} ({int((delta > 0).sum())}/{len(delta)} +)"})
    b = frame[frame.run.str.startswith("e018")].groupby("val")[["macro_f1", "kappa", "rem_f1"]].agg(["mean", "std"]).round(4)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(b); print(pd.DataFrame(out).to_string(index=False))


if __name__ == "__main__":
    main()
