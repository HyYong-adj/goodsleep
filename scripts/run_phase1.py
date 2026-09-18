"""Phase 1: byoungjun 트랙의 부가 요소를 동결 Conformer embedding 위에서 단일 변수로 검증한다.

A0 는 parent(Conformer + BiLSTM40, Macro-F1 0.483261)와 동일 설정이다. 코드 변경이
학습 경로를 건드리지 않았다면 seed 20260910 에서 parent 값을 정확히 재현해야 하며,
이를 무결성 검사로 강제한다.

주의: 참조 트랙에서 네 요소는 각각 단독으로는 무효(±0.01)였고 결합해야 효과가 있었다.
따라서 A1-A4 단독 런은 귀속(attribution)용이고, 채택 판정은 A5(결합)로 한다.
"""
from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from psg_only.evaluate import evaluate_b1, write_evaluation  # noqa: E402
from psg_only.train import device_for_run, load_config, train_b1  # noqa: E402

PARENT = ROOT / "artifacts/experiments/conformer_epoch_20260915T054210274368058"
PARENT_MACRO_F1 = 0.48326135281572924
PARENT_SEED = 20260910

# 각 변형은 parent 대비 정확히 한 가지만 바꾼다 (A5 만 결합).
VARIANTS = {
    "A0": {"label": "parent 재현 (변경 없음)", "model": {}, "train": {}},
    "A1": {"label": "시각 특징 [h, h^2]", "model": {"time_feature": True}, "train": {}},
    "A2": {"label": "cosine LR + warmup 3", "model": {}, "train": {"scheduler": "cosine", "warmup_epochs": 3}},
    "A3": {"label": "label smoothing 0.05", "model": {}, "train": {"label_smoothing": 0.05}},
    "A4": {"label": "REM class weight x2", "model": {}, "train": {"class_weight_multipliers": {"REM": 2.0}}},
    "A5": {
        "label": "A1-A4 결합",
        "model": {"time_feature": True},
        "train": {
            "scheduler": "cosine",
            "warmup_epochs": 3,
            "label_smoothing": 0.05,
            "class_weight_multipliers": {"REM": 2.0},
        },
    },
}


def build_config(variant: str, seed: int) -> dict:
    config = load_config(PARENT / "b1.yaml")
    spec = VARIANTS[variant]
    config["seed"] = int(seed)
    config["model"].update(copy.deepcopy(spec["model"]))
    config["train"].update(copy.deepcopy(spec["train"]))
    config["experiment"] = {
        "phase": "phase1",
        "variant": variant,
        "variant_label": spec["label"],
        "parent_run": str(PARENT),
        "changed": sorted(list(spec["model"]) + list(spec["train"])) or ["none"],
    }
    return config


def run_one(variant: str, seed: int, output: Path) -> dict:
    config = build_config(variant, seed)
    output.mkdir(parents=True, exist_ok=False)
    (output / "b1.yaml").write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
    started = time.perf_counter()
    checkpoint_path = train_b1(config, output / "b1")
    rows, checkpoint = evaluate_b1(checkpoint_path)
    result = write_evaluation(rows, checkpoint, output / "b1/evaluation")
    if result["status"] != "complete":
        raise ValueError(f"{variant} s{seed}: evaluation missing classes.")
    metrics = result["metrics"]
    record = {
        "variant": variant,
        "variant_label": VARIANTS[variant]["label"],
        "seed": seed,
        "macro_f1": metrics["macro_f1"],
        "accuracy": metrics["accuracy"],
        "cohen_kappa": metrics["cohen_kappa"],
        "per_class_f1": {name: metrics["per_class"][name]["f1"] for name in metrics["per_class"]},
        "per_class_recall": {name: metrics["per_class"][name]["recall"] for name in metrics["per_class"]},
        "transition_macro_f1": metrics.get("transition_macro_f1"),
        "stable_macro_f1": metrics.get("stable_macro_f1"),
        "predicted_transition_rate": metrics.get("predicted_transition_rate"),
        "epochs": result["epochs"],
        "wall_seconds": time.perf_counter() - started,
        "output": str(output),
    }
    (output / "record.json").write_text(json.dumps(record, indent=2))
    return record


def neighbour_inflation(history_path: Path) -> dict:
    """best epoch 과 인접 epoch 평균의 차이. 참조 트랙 실측치는 +0.049 였다."""
    scores = [json.loads(line)["validation_macro_f1"] for line in history_path.read_text().splitlines() if line.strip()]
    if len(scores) < 3:
        return {"best": max(scores) if scores else None, "neighbour_mean": None, "inflation": None}
    best_at = max(range(len(scores)), key=lambda i: scores[i])
    neighbours = [scores[i] for i in (best_at - 1, best_at + 1) if 0 <= i < len(scores)]
    mean = statistics.fmean(neighbours)
    return {"best": scores[best_at], "best_epoch": best_at, "neighbour_mean": mean, "inflation": scores[best_at] - mean}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", default="A0,A1,A2,A3,A4,A5")
    parser.add_argument("--seeds", default="20260910,20260911,20260912")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    unknown = set(variants) - set(VARIANTS)
    if unknown:
        raise SystemExit(f"Unknown variants: {sorted(unknown)}")

    out = (args.output or ROOT / "artifacts/experiments" / datetime.now(timezone.utc).strftime("phase1_%Y%m%dT%H%M%SZ")).resolve()
    out.mkdir(parents=True, exist_ok=False)
    device = device_for_run()
    state = {
        "phase": "phase1",
        "status": "RUNNING",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "parent_run": str(PARENT),
        "parent_macro_f1": PARENT_MACRO_F1,
        "evaluation_scope": "validation-only",
        "variants": {name: VARIANTS[name]["label"] for name in variants},
        "seeds": seeds,
        "records": [],
        "note": "단독 런은 귀속용, 판정은 A5(결합). 참조 트랙에서 네 요소는 단독 무효였다.",
    }

    def save() -> None:
        (out / "phase1.json").write_text(json.dumps(state, indent=2, ensure_ascii=False))

    save()
    try:
        for variant in variants:
            for seed in seeds:
                tag = f"{variant}_s{seed}"
                print(json.dumps({"stage": "start", "run": tag}, ensure_ascii=False), flush=True)
                record = run_one(variant, seed, out / tag)
                record["selection_inflation"] = neighbour_inflation(out / tag / "b1/history.jsonl")
                if variant == "A0" and seed == PARENT_SEED:
                    delta = abs(record["macro_f1"] - PARENT_MACRO_F1)
                    record["parent_reproduction_delta"] = record["macro_f1"] - PARENT_MACRO_F1
                    if delta > 1e-9:
                        state["status"] = "FAILED_PARENT_REPRODUCTION"
                        state["records"].append(record)
                        save()
                        raise SystemExit(
                            f"A0 s{PARENT_SEED} did not reproduce parent: "
                            f"{record['macro_f1']!r} vs {PARENT_MACRO_F1!r} (delta {delta:.3e}). "
                            "코드 변경이 학습 경로를 바꿨다는 뜻이므로 Phase 1 결과를 신뢰할 수 없다."
                        )
                state["records"].append(record)
                save()
                print(json.dumps({"stage": "done", "run": tag, "macro_f1": record["macro_f1"]}, ensure_ascii=False), flush=True)
        state["status"] = "COMPLETE"
        state["completed_at"] = datetime.now(timezone.utc).isoformat()
        save()
        print(json.dumps({"stage": "complete", "output": str(out)}, ensure_ascii=False), flush=True)
    except BaseException as error:
        if not state["status"].startswith("FAILED"):
            state["status"] = "FAILED"
        state["error_type"] = type(error).__name__
        state["error"] = str(error)
        save()
        raise


if __name__ == "__main__":
    main()
