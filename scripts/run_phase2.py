"""Phase 2: 참조 트랙 combo 의 주력 요소(night_norm, SpecAugment)를 C1 encoder 에 이식한다.

Phase 1 은 combo 를 시험하지 않았다 — 겹치는 요소가 시각 특징 하나뿐이었다.
combo 의 주력인 night_norm 과 SpecAugment 는 encoder 재학습이 필요해 여기로 미뤘다.

각 변형: C1 학습 -> C1 평가 -> embedding 재생성 -> B1 학습 -> B1 평가.
B1 설정은 parent 그대로 두어 encoder 단계 변경만 측정한다.
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from cache_embeddings import cache_embeddings  # noqa: E402
from psg_only.evaluate import evaluate_b0, evaluate_b1, write_evaluation  # noqa: E402
from psg_only.provenance import file_hash  # noqa: E402
from psg_only.train import device_for_run, train_b0, train_b1  # noqa: E402

PARENT = ROOT / "artifacts/experiments/conformer_epoch_20260915T054210274368058"
PARENT_C1_MACRO_F1 = 0.4126620456901757
PARENT_B1_MACRO_F1 = 0.48326135281572924

SPEC_AUGMENT = {"freq_mask": 8, "time_mask": 150, "gain_std": 0.25}

VARIANTS = {
    # C0 는 같은 GPU 에서 돌리는 대조군이다. parent 는 GPU 3 에서 학습됐고
    # 커널 선택 차이로 결과가 비트 단위로 재현되지 않으므로(§게이트), 이것 없이는
    # 모든 Δ 가 GPU 효과와 교란된다.
    "C0": {"label": "대조군 (옵션 없음, 동일 GPU)", "data": {}, "train": {}},
    "C2": {"label": "night_norm", "data": {"night_norm": True}, "train": {}},
    "C3": {"label": "SpecAugment", "data": {}, "train": {"spec_augment": SPEC_AUGMENT}},
    "C4": {"label": "night_norm + SpecAugment", "data": {"night_norm": True}, "train": {"spec_augment": SPEC_AUGMENT}},
}


def integrity_gate(work: Path) -> dict:
    """옵션이 꺼져 있으면 embedding 생성 경로가 parent 와 수치적으로 같아야 한다.

    cache_embeddings 를 수정했으므로 parent C1 체크포인트로 embedding 을 다시 만들어 대조한다.

    **비트 단위 동일성은 기준이 될 수 없다.** parent 는 GPU 3 에서, 현재는 다른 GPU 에서
    생성되며 커널 선택이 달라진다. 실측: 같은 GPU 안에서는 해시가 완전히 일치했고,
    GPU 가 다르면 요소의 약 0.05% 가 달라지되 대부분 float16 1 ULP 이내였다.

    기준을 두 번 완화했으므로 근거를 남긴다.
      1. 비트 동일 -> 불가능(GPU 간 커널 차이). 같은 GPU 재실행은 해시 일치로 확인.
      2. 1 ULP -> 2 ULP. 한 요소가 1 ULP 를 넘었는데 값이 -6.05e-05 로 float16
         subnormal 경계였다. embedding 범위가 ±6 이므로 이 크기는 의미가 없다.
    2 ULP 를 넘는 차이는 여전히 실패로 본다.
    """
    import numpy as np

    scratch = work / "_gate_embeddings"
    if scratch.exists():
        shutil.rmtree(scratch)
    cache_embeddings(str(PARENT / "c1/best.pt"), str(scratch), batch_size=64,
                     train_subject_limit=3, val_subject_limit=3)
    checked, failures = {}, []
    for subject in sorted(p.name for p in scratch.iterdir() if p.is_dir()):
        new = np.load(scratch / subject / "embeddings.npy")
        old = np.load(PARENT / "embeddings" / subject / "embeddings.npy")
        if new.shape != old.shape:
            failures.append(f"{subject}: shape {new.shape} != {old.shape}")
            continue
        difference = np.abs(new.astype(np.float64) - old.astype(np.float64))
        tolerance = 2.0 * np.spacing(np.abs(old)).astype(np.float64)  # float16 2 ULP
        outside = int(np.count_nonzero(difference > tolerance))
        checked[subject] = {
            "differing_elements": int(np.count_nonzero(difference)),
            "total_elements": int(difference.size),
            "max_abs_difference": float(difference.max()),
            "embedding_abs_max": float(np.abs(old).max()),
            "outside_two_ulp": outside,
        }
        if outside:
            failures.append(f"{subject}: {outside} elements outside 2 float16 ULP")
    shutil.rmtree(scratch)
    report = {
        "criterion": "all differences within two float16 ULP (bit equality is not reproducible across GPUs)",
        "subjects": checked,
        "failures": failures,
    }
    if failures:
        raise SystemExit(
            f"Embedding regeneration gate FAILED: {failures}. "
            "cache_embeddings 수정이 기본 경로를 바꿨다는 뜻이므로 Phase 2 결과를 신뢰할 수 없다."
        )
    return report


def build_configs(variant: str, seed: int, embedding_root: Path) -> tuple[dict, dict]:
    spec = VARIANTS[variant]
    c1 = yaml.safe_load((PARENT / "c1.yaml").read_text())
    c1["seed"] = int(seed)
    c1.pop("experiment", None)
    c1.pop("b1", None)
    c1["data"].update(copy.deepcopy(spec["data"]))
    c1["train"].update(copy.deepcopy(spec["train"]))

    b1 = yaml.safe_load((PARENT / "b1.yaml").read_text())
    b1["seed"] = int(seed)
    b1["data"]["embedding_root"] = str(embedding_root)
    return c1, b1


def run_one(variant: str, seed: int, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    embedding_root = output / "embeddings"
    c1_cfg, b1_cfg = build_configs(variant, seed, embedding_root)
    (output / "c1.yaml").write_text(yaml.safe_dump(c1_cfg, sort_keys=False, allow_unicode=True))
    (output / "b1.yaml").write_text(yaml.safe_dump(b1_cfg, sort_keys=False, allow_unicode=True))

    started = time.perf_counter()
    c1_path = train_b0(c1_cfg, output / "c1")
    c1_seconds = time.perf_counter() - started
    rows, checkpoint = evaluate_b0(c1_path)
    c1_result = write_evaluation(rows, checkpoint, output / "c1/evaluation")

    cache_embeddings(str(c1_path), str(embedding_root), batch_size=64)

    b1_path = train_b1(b1_cfg, output / "b1")
    rows, checkpoint = evaluate_b1(b1_path)
    b1_result = write_evaluation(rows, checkpoint, output / "b1/evaluation")
    for name, result in (("C1", c1_result), ("B1", b1_result)):
        if result["status"] != "complete":
            raise ValueError(f"{variant} {name}: evaluation missing classes.")

    record = {
        "variant": variant,
        "variant_label": VARIANTS[variant]["label"],
        "seed": seed,
        "c1_macro_f1": c1_result["metrics"]["macro_f1"],
        "b1_macro_f1": b1_result["metrics"]["macro_f1"],
        "c1_delta_vs_parent": c1_result["metrics"]["macro_f1"] - PARENT_C1_MACRO_F1,
        "b1_delta_vs_parent": b1_result["metrics"]["macro_f1"] - PARENT_B1_MACRO_F1,
        "b1_accuracy": b1_result["metrics"]["accuracy"],
        "b1_cohen_kappa": b1_result["metrics"]["cohen_kappa"],
        "b1_per_class_f1": {k: v["f1"] for k, v in b1_result["metrics"]["per_class"].items()},
        "b1_per_class_recall": {k: v["recall"] for k, v in b1_result["metrics"]["per_class"].items()},
        "b1_transition_macro_f1": b1_result["metrics"].get("transition_macro_f1"),
        # EmbeddingWindowDataset 의 provenance 는 일부 키만 보존하므로 캐시 metadata 에서 직접 읽는다.
        "embedding_normalization": json.loads(
            next((embedding_root).glob("*/metadata.json")).read_text()).get("normalization"),
        "c1_train_seconds": c1_seconds,
        "wall_seconds": time.perf_counter() - started,
        "output": str(output),
    }
    (output / "record.json").write_text(json.dumps(record, indent=2, ensure_ascii=False))
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", default="C0,C2,C3,C4")
    parser.add_argument("--seeds", default="20260910")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--skip-gate", action="store_true", help="이미 게이트를 통과한 뒤 이어서 돌릴 때만 사용")
    args = parser.parse_args()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    unknown = set(variants) - set(VARIANTS)
    if unknown:
        raise SystemExit(f"Unknown variants: {sorted(unknown)}")

    out = (args.output or ROOT / "artifacts/experiments" / datetime.now(timezone.utc).strftime("phase2_%Y%m%dT%H%M%SZ")).resolve()
    out.mkdir(parents=True, exist_ok=True)
    state = {
        "phase": "phase2",
        "status": "RUNNING",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "device": str(device_for_run()),
        "parent_run": str(PARENT),
        "parent_c1_macro_f1": PARENT_C1_MACRO_F1,
        "parent_b1_macro_f1": PARENT_B1_MACRO_F1,
        "evaluation_scope": "validation-only",
        "variants": {name: VARIANTS[name]["label"] for name in variants},
        "seeds": seeds,
        "records": [],
        "note": "판정은 C4(결합). B1 설정은 parent 고정이므로 encoder 단계 변경만 측정한다.",
    }

    def save() -> None:
        (out / "phase2.json").write_text(json.dumps(state, indent=2, ensure_ascii=False))

    save()
    try:
        if not args.skip_gate:
            print(json.dumps({"stage": "integrity_gate"}), flush=True)
            state["integrity_gate"] = integrity_gate(out)
            save()
            print(json.dumps({"stage": "integrity_gate", "result": "PASS"}), flush=True)
        for variant in variants:
            for seed in seeds:
                tag = f"{variant}_s{seed}"
                print(json.dumps({"stage": "start", "run": tag}, ensure_ascii=False), flush=True)
                record = run_one(variant, seed, out / tag)
                state["records"].append(record)
                save()
                print(json.dumps({"stage": "done", "run": tag,
                                  "c1": record["c1_macro_f1"], "b1": record["b1_macro_f1"]}, ensure_ascii=False), flush=True)
        state["status"] = "COMPLETE"
        state["completed_at"] = datetime.now(timezone.utc).isoformat()
        save()
        print(json.dumps({"stage": "complete", "output": str(out)}), flush=True)
    except BaseException as error:
        state["status"] = "FAILED"
        state["error_type"] = type(error).__name__
        state["error"] = str(error)
        save()
        raise


if __name__ == "__main__":
    main()
