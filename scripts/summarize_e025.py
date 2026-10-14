"""choihy-e025 집계: fold × 정규화 → 평균 ± sd, in-domain(PSG-Audio fold val) 대비 paired 차이, 층별 macro-F1."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/experiments/e025_snubh_external"
FLAGS = Path("/home/sleep/data/marg_psg_audio/_derived/snubh_psg/cache/audio_compact_v1/manifest.csv")
CLASSES = ("Wake", "REM", "Light", "Deep")


def main():
    records, strata = [], []
    for path in sorted(BASE.glob("*__*/results.json")):
        r = json.loads(path.read_text())
        fold, norm = path.parent.name.split("__")
        m = r["metrics"]
        pred = pd.read_csv(path.parent / "predictions.csv")
        records.append({"fold": fold, "norm": norm, "macro_f1": m["macro_f1"], "kappa": m["cohen_kappa"],
                        "acc": m["accuracy"], **{f"f1_{c}": m["per_class"][c]["f1"] for c in CLASSES},
                        "pred_rem_frac": float((pred.prediction == 1).mean()),
                        "in_domain": r["in_domain_val_macro_f1"]})
        for column, groups in r["strata"].items():
            for value, g in groups.items():
                strata.append({"fold": fold, "norm": norm, "stratum": column, "value": value,
                               "n": g["subjects"], "macro_f1": g["macro_f1"], "rem_f1": g["per_class_f1"]["REM"]})
    frame = pd.DataFrame(records)
    frame["drop_vs_in_domain"] = frame.macro_f1 - frame.in_domain
    summary = frame.drop(columns="fold").groupby("norm").agg(["mean", "std"]).round(4)
    st = pd.DataFrame(strata).groupby(["norm", "stratum", "value"]).agg(n=("n", "first"), macro_f1=("macro_f1", "mean"),
                                                                       rem_f1=("rem_f1", "mean")).round(3)
    # low_dynamic_range 사후 층화 (pbj §7.2)
    flags = pd.read_csv(FLAGS, usecols=["subject_id", "quality_flags"]).groupby("subject_id").quality_flags.first().fillna("")
    from sklearn.metrics import f1_score
    ldr = []
    for path in sorted(BASE.glob("*__*/predictions.csv")):
        fold, norm = path.parent.name.split("__")
        p = pd.read_csv(path)
        p["ldr"] = p.subject_id.map(flags).str.contains("low_dynamic_range")
        for value, part in p.groupby("ldr"):
            ldr.append({"norm": norm, "low_dynamic_range": bool(value), "subjects": part.subject_id.nunique(),
                        "macro_f1": f1_score(part.target, part.prediction, labels=range(4), average="macro", zero_division=0)})
    ldr = pd.DataFrame(ldr).groupby(["norm", "low_dynamic_range"]).agg(subjects=("subjects", "first"), macro_f1=("macro_f1", "mean")).round(3)
    out = {"per_run": frame.round(4).to_dict("records")}
    (BASE / "summary.json").write_text(json.dumps(out, indent=2))
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.max_rows", 200):
        print(frame.round(3).to_string(index=False)); print(); print(summary.T); print(); print(st); print(); print(ldr)


if __name__ == "__main__":
    main()
