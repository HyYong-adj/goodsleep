"""choihy-e025: PSG-Audio 로 학습한 동결 체크포인트를 SNUBH PSG 외부 번들에서 평가한다.

평가 전용(A 등급). SNUBH 로 학습·체크포인트 선택·보정을 하지 않는다. 입력은 pbj 가 만든
`snubh_psg_external_v1` 번들의 log-mel 캐시(`audio_compact_v1`, PSG-Audio 와 같은 48×1499 계약)다.

정규화 조건 (전부 라벨 미사용):
  N0 train  — PSG-Audio train 전역 통계(-40.9/11.8). 모델을 학습 그대로 쓴다. **주 조건.**
  N1 night  — 밤별 z-score. 아침 밤 전체 추론이라 제품 정의상 가능한 test-time 정규화.
  N2 domain — SNUBH 데이터셋 전역 통계(-25.4/10.0). 표적 도메인 unlabeled 통계를 쓰는 transductive 참고값.

SNUBH 매니페스트의 `recording_start_seconds` 는 ANALYSIS-START 격자라 30 s 배수가 아니다.
내 로더는 그 열이 있으면 30 s 격자를 강제하므로 열을 빼고 `subject_epoch_index`(연속 확인됨)를
시간축으로 쓰는 파생 매니페스트를 출력 디렉터리에 만든다. 공유 번들은 건드리지 않는다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from psg_only.data import MelWindowDataset  # noqa: E402
from psg_only.endtoend import endtoend_from_config  # noqa: E402
from psg_only.metrics import evaluate_predictions  # noqa: E402
from psg_only.provenance import file_hash  # noqa: E402
from psg_only.transformer import configure_precision  # noqa: E402
from psg_only.windows import stitch_predictions  # noqa: E402
from run_phase3 import _predict  # noqa: E402

SNUBH = Path("/home/sleep/data/marg_psg_audio/_derived/snubh_psg")
CACHE = SNUBH / "cache/audio_compact_v1"
BUNDLE = SNUBH / "bundles/snubh_psg_external_v1"
PSG_TRAIN_STATS = Path("/home/sleep/researchers/byoungjun/cache/audio_compact_full/train_stats.json")
LEGACY_MODEL_DEFAULTS = {"post_transformer_layers": 0, "post_transformer_heads": 4,
                         "post_transformer_ff": 1024, "in_channels": 1}
STRATA = ("ahi_class", "age_band", "study_type", "sync_quality", "sex_code")


def build_manifest(out: Path) -> Path:
    frame = pd.read_csv(CACHE / "manifest.csv", dtype={"subject_id": str})
    bundle = pd.read_csv(BUNDLE / "official_subjects.csv", dtype={"subject_id": str})
    if set(frame.subject_id) != set(bundle.subject_id):
        raise ValueError("cache manifest and bundle subjects differ")
    frame = frame.drop(columns=["recording_start_seconds"]).assign(split="val")
    path = out / "snubh_external_manifest.csv"
    frame.to_csv(path, index=False)
    return path


def subject_meta() -> pd.DataFrame:
    epochs = pd.read_csv(BUNDLE / "epochs.csv", dtype={"subject_id": str}, usecols=["subject_id", *STRATA])
    return epochs.groupby("subject_id").first()


def domain_stats(out: Path) -> Path:
    payload = json.loads((CACHE / "dataset_stats.json").read_text())
    path = out / "snubh_domain_stats.json"
    path.write_text(json.dumps({"mean": payload["mean"], "std": payload["std"], "source": "snubh audio_compact_v1 dataset_stats"}))
    return path


def stratified(rows: list[dict], meta: pd.DataFrame) -> dict:
    frame = pd.DataFrame(rows)
    report = {}
    for column in STRATA:
        groups = {}
        for value, part in frame.groupby(frame.subject_id.map(meta[column])):
            metrics = evaluate_predictions(part.to_dict("records"))
            groups[str(value)] = {"subjects": int(part.subject_id.nunique()), "epochs": len(part),
                                  "macro_f1": metrics["macro_f1"], "kappa": metrics["cohen_kappa"],
                                  "per_class_f1": {k: v["f1"] for k, v in metrics["per_class"].items()}}
        report[column] = groups
    return report


def per_subject(rows: list[dict]) -> list[dict]:
    from sklearn.metrics import cohen_kappa_score, f1_score
    out = []
    for sid, part in pd.DataFrame(rows).groupby("subject_id"):
        t, p = part.target.to_numpy(), part.prediction.to_numpy()
        out.append({"subject_id": sid, "epochs": len(part),
                    "macro_f1": float(f1_score(t, p, labels=np.arange(4), average="macro", zero_division=0)),
                    "kappa": float(cohen_kappa_score(t, p)) if len(set(t)) > 1 else None,
                    "acc": float((t == p).mean()), "true_rem_frac": float((t == 1).mean()),
                    "pred_rem_frac": float((p == 1).mean())})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, nargs="+")
    parser.add_argument("--norm", choices=["train", "night", "domain"], default="train")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--subjects", type=int, help="smoke: 앞 N 녹화만")
    args = parser.parse_args()

    configure_precision()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(args.output)
    stats = domain_stats(args.output) if args.norm == "domain" else PSG_TRAIN_STATS
    meta = subject_meta()
    device = torch.device(args.device)
    dataset = MelWindowDataset(manifest, CACHE, stats, split="val", input_epochs=40, target_epochs=20, stride=20,
                               subject_limit=args.subjects, night_norm=args.norm == "night")
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=6)

    for checkpoint in args.checkpoint:
        name = checkpoint.parent.name
        target = args.output / f"{name}__{args.norm}"
        if (target / "results.json").is_file():
            print(f"skip {target}", flush=True)
            continue
        target.mkdir(parents=True, exist_ok=True)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload["config"]["data"].get("night_norm") or payload["config"]["data"].get("frontend", "logmel") != "logmel":
            raise SystemExit(f"{name}: 학습 정규화/front-end 가 N0 가정과 다르다")
        config = json.loads(json.dumps(payload["config"]))
        # e018 등은 post_transformer/in_channels 옵션 도입 전 체크포인트다. 기본값이 당시 구조와 같다.
        for key, value in LEGACY_MODEL_DEFAULTS.items():
            config["model"].setdefault(key, value)
        model = endtoend_from_config(config).to(device)
        model.load_state_dict(payload["state_dict"])
        rows = stitch_predictions(_predict(model, loader, device), dataset.expected_keys())
        pd.DataFrame(rows).to_csv(target / "predictions.csv", index=False)
        metrics = evaluate_predictions(rows)
        result = {"experiment": "choihy-e025", "checkpoint": str(checkpoint), "checkpoint_sha256": file_hash(checkpoint),
                  "norm": args.norm, "stats_path": str(stats), "manifest_sha256": file_hash(manifest),
                  "bundle": BUNDLE.name, "smoke": bool(args.subjects),
                  "subjects": len({r["subject_id"] for r in rows}), "epochs": len(rows),
                  "in_domain_val_macro_f1": payload.get("validation_macro_f1"),
                  "evaluation_scope": "external-only (no fitting/selection on SNUBH)",
                  "metrics": metrics, "strata": stratified(rows, meta)}
        (target / "results.json").write_text(json.dumps(result, indent=2, default=float))
        pd.DataFrame(per_subject(rows)).to_csv(target / "per_subject.csv", index=False)
        print(json.dumps({"run": target.name, "macro_f1": metrics["macro_f1"], "kappa": metrics["cohen_kappa"],
                          "rem_f1": metrics["per_class"]["REM"]["f1"] if "REM" in metrics["per_class"] else None}), flush=True)


if __name__ == "__main__":
    main()
