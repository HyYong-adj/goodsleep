"""Controlled C1 Conformer -> fresh frozen embeddings -> unchanged BiLSTM40."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def read_config(path):
    import yaml
    from psg_only.conformer import conformer_from_config
    cfg = yaml.safe_load(Path(path).read_text())
    if set(cfg) != {"seed", "experiment", "data", "model", "train", "b1"}:
        raise ValueError("Unknown/missing Conformer experiment sections.")
    if set(cfg["experiment"]) != {"parent_run", "name", "hypothesis"}:
        raise ValueError("Unknown/missing experiment options.")
    if set(cfg["data"]) != {"manifest", "cache_root", "stats"}:
        raise ValueError("Retain the parent manifest, Mel cache and train stats.")
    for key in cfg["data"]:
        cfg["data"][key] = str(resolve(cfg["data"][key]))
    cfg["experiment"]["parent_run"] = str(resolve(cfg["experiment"]["parent_run"]))
    if set(cfg["b1"]) != {"data", "model", "train"}:
        raise ValueError("Unknown/missing downstream B1 options.")
    conformer_from_config(cfg)
    return cfg


def resolve(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def dump(path, payload):
    path.write_text(json.dumps(payload, indent=2) + "\n")


def reference_rows(parent, model, dataset):
    import pandas as pd
    import torch
    from psg_only.metrics import evaluate_predictions
    from psg_only.provenance import file_hash
    from psg_only.train import config_hash
    from psg_only.constants import CANONICAL_CLASSES, CLASS_ORDER_VERSION
    from psg_only.windows import stitch_predictions
    from run_transformer_pair import check_reference_metrics

    base = parent / model
    checkpoint = torch.load(base / "best.pt", map_location="cpu", weights_only=False)
    if checkpoint["model_type"] != model or checkpoint["config"].get("smoke"):
        raise ValueError("Parent must contain full B0/B1 checkpoints.")
    if checkpoint.get("class_order_version") != CLASS_ORDER_VERSION or tuple(checkpoint.get("class_order", ())) != CANONICAL_CLASSES:
        raise ValueError("Parent class order mismatch.")
    if config_hash(checkpoint["config"]) != checkpoint["config_hash"]:
        raise ValueError("Parent config hash mismatch.")
    saved = json.loads((base / "evaluation/results.json").read_text())
    if saved.get("status") != "complete" or saved.get("smoke"):
        raise ValueError("Parent evaluation is not complete/full.")
    if saved["checkpoint_sha256"] != file_hash(base / "best.pt"):
        raise ValueError("Parent evaluation/checkpoint hash mismatch.")
    rows = pd.read_csv(base / "evaluation/predictions.csv", dtype={"subject_id": str}).to_dict("records")
    rows = stitch_predictions(rows)
    check_reference_metrics(evaluate_predictions(rows), saved["metrics"])
    selected_ids = set(dataset.frame.subject_id)
    selected = [r for r in rows if str(r["subject_id"]) in selected_ids]
    expected = dataset.expected_keys()
    stitch_predictions(selected, expected)
    expected_labels = {}
    for sid, frame in dataset.frame.groupby("subject_id"):
        _, labels, valid = dataset._arrays(sid)
        for row in frame.itertuples():
            if valid[row.subject_epoch_index]:
                expected_labels[(sid, int(row.epoch_index))] = (0, 2, 3, 1)[int(labels[row.subject_epoch_index])]
    if any(expected_labels[(r["subject_id"], r["epoch_index"])] != r["target"] for r in selected):
        raise ValueError("Parent and candidate target labels differ.")
    return selected, checkpoint, {
        "checkpoint_sha256": file_hash(base / "best.pt"),
        "predictions_sha256": file_hash(base / "evaluation/predictions.csv"),
        "source": "Stored predictions, checksummed and re-scored by the current evaluator.",
    }


def preflight(cfg, smoke=False):
    import numpy as np
    from psg_only.data import CachedMelEpochDataset, validate_inputs
    from psg_only.provenance import file_hash
    from psg_only.train import class_weights
    from psg_only.windows import context_options

    data = cfg["data"]
    audit = validate_inputs(data["manifest"], data["cache_root"], data["stats"])
    parent = Path(cfg["experiment"]["parent_run"])
    train = CachedMelEpochDataset(**dict(manifest_path=data["manifest"], cache_root=data["cache_root"],
                                       stats_path=data["stats"], split="train", subject_limit=8 if smoke else None))
    val = CachedMelEpochDataset(data["manifest"], data["cache_root"], data["stats"], "val", 2 if smoke else None)
    refs = {}
    for kind in ("b0", "b1"):
        rows, cp, identity = reference_rows(parent, kind, val)
        refs[kind] = {"rows": rows, "checkpoint": cp, "identity": identity}
        if cp["config"]["seed"] != cfg["seed"] or cp["config"]["manifest_hash"] != file_hash(data["manifest"]):
            raise ValueError("Parent seed/manifest differs.")
    b0 = refs["b0"]["checkpoint"]["config"]
    b1 = refs["b1"]["checkpoint"]["config"]
    if cfg["train"] != b0["train"]:
        raise ValueError("Conformer must keep the parent B0 training settings.")
    if any(str(resolve(data[k])) != str(resolve(b0["data"][k])) for k in data):
        raise ValueError("Do not change Mel source, manifest or normalization.")
    if cfg["model"]["dropout"] != b0["model"]["dropout"]:
        raise ValueError("Retain parent epoch-classifier dropout.")
    if file_hash(data["stats"]) != b0["stats_hash"]:
        raise ValueError("Parent train stats changed.")
    if cfg["b1"]["model"] != b1["model"] or cfg["b1"]["train"] != b1["train"]:
        raise ValueError("Keep the parent B1 model and training settings.")
    if set(cfg["b1"]["data"]) != {"input_epochs", "target_epochs", "left_context", "stride"}:
        raise ValueError("Unknown B1 context fields.")
    context_options(cfg["b1"]["data"])
    if any(b1["data"].get(k) != v for k, v in cfg["b1"]["data"].items()) or cfg["b1"]["data"]["input_epochs"] != 40:
        raise ValueError("Downstream comparison must use parent 40-to-20 context.")
    if b1["embedding_provenance"]["checkpoint_sha256"] != refs["b0"]["identity"]["checkpoint_sha256"]:
        raise ValueError("Parent B1 did not use this B0.")
    counts = np.bincount(train.labels(), minlength=4)
    return dict(
        parent_run=str(parent), smoke=smoke, evaluation_scope="validation-only", seed=cfg["seed"],
        manifest_hash=file_hash(data["manifest"]), stats_hash=file_hash(data["stats"]),
        full_data_audit=audit,
        train=dict(subjects=int(train.frame.subject_id.nunique()), rows=len(train), valid_epochs=int(counts.sum())),
        val=dict(subjects=int(val.frame.subject_id.nunique()), rows=len(val), valid_epochs=len(val.expected_keys())),
        train_class_counts=counts.tolist(), class_weights=class_weights(train.labels()).tolist(),
        references={k: ref["identity"] for k, ref in refs.items()},
        budget=dict(c1_epochs=cfg["train"]["epochs"], b1_epochs=cfg["b1"]["train"]["epochs"],
                    c1_max_optimizer_steps=((len(train) + cfg["train"]["batch_size"] - 1) // cfg["train"]["batch_size"]) * cfg["train"]["epochs"]),
        fixed_feature_shape=[1, 48, 1499], conformer_token_length=188,
    ), refs


def wait_for_gpu(gpu, wait, on_wait):
    while True:
        result = subprocess.run(
            ["nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            capture_output=True, text=True,
        )
        if result.returncode:
            raise RuntimeError("GPU usage check failed: " + result.stderr.strip())
        if not result.stdout.strip():
            return
        if not wait:
            raise RuntimeError("GPU is busy. Retry later or use --wait-for-gpu; no processes were stopped.")
        on_wait()
        print(json.dumps(dict(status="WAITING_FOR_GPU", gpu=gpu, check_seconds=30)), flush=True)
        time.sleep(30)


def verify_source_snapshot(output):
    from psg_only.provenance import file_hash
    environment = json.loads((output / "environment.json").read_text())
    for relative, expected in environment["source_hashes"].items():
        if file_hash(ROOT / relative) != expected:
            raise RuntimeError("Source changed while preparing/waiting: " + relative + "; start a new run.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/c1_conformer_epoch.yaml")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--wait-for-gpu", action="store_true")
    device = parser.add_mutually_exclusive_group()
    device.add_argument("--gpu", type=int)
    device.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if args.run and args.gpu is None and not args.cpu:
        parser.error("--run requires --gpu INDEX or --cpu.")
    if args.run and not args.smoke and args.gpu is None:
        parser.error("Full training requires an assigned GPU; use --smoke for CPU.")
    if args.gpu is not None and args.gpu < 0:
        parser.error("GPU index must be nonnegative.")
    if args.wait_for_gpu and (not args.run or args.gpu is None):
        parser.error("--wait-for-gpu requires --run --gpu INDEX.")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu) if args.gpu is not None else ""
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    import torch
    import yaml
    from psg_only.evaluate import evaluate_b0, evaluate_b1, write_evaluation
    from psg_only.metrics import evaluate_predictions
    from psg_only.train import train_b0, train_b1
    from psg_only.transformer import configure_precision
    from run_transformer_pair import capture_sources, target_set
    from cache_embeddings import cache_embeddings
    configure_precision()

    output = resolve(args.output) if args.output else ROOT / "artifacts/experiments" / datetime.now(timezone.utc).strftime("conformer_epoch_%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    state = dict(status="PREPARING", stage="preflight", smoke=args.smoke,
                 evaluation_scope="validation-only", gpu=args.gpu, output=str(output),
                 started_at=datetime.now(timezone.utc).isoformat(), training_started=False)
    def save():
        dump(output / "experiment.json", state)
    save()
    try:
        cfg = read_config(resolve(args.config))
        state.update(seed=cfg["seed"], parent_run=cfg["experiment"]["parent_run"],
                     hypothesis=cfg["experiment"]["hypothesis"],
                     budget={"c1_epochs": cfg["train"]["epochs"], "b1_epochs": cfg["b1"]["train"]["epochs"]})
        save()
        capture_sources(output, Path(__file__).resolve())
        from psg_only.provenance import file_hash
        environment = json.loads((output / "environment.json").read_text())
        for script in ("cache_embeddings.py", "run_transformer_pair.py"):
            import shutil
            shutil.copy2(ROOT / "scripts" / script, output / "code/scripts" / script)
            environment["source_hashes"]["scripts/" + script] = file_hash(ROOT / "scripts" / script)
        dump(output / "environment.json", environment)
        (output / "experiment.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        report, refs = preflight(cfg, args.smoke)
        dump(output / "preflight.json", report)
        state.update(status="PREPARED")
        save()
        print(json.dumps(dict(output=str(output), status=state["status"], preflight=report)), flush=True)
        if not args.run:
            return
        if args.gpu is not None:
            if not torch.cuda.is_available():
                raise RuntimeError("Assigned CUDA device is unavailable; no CPU fallback.")
            def on_wait():
                state.update(status="WAITING_FOR_GPU")
                save()
            wait_for_gpu(args.gpu, args.wait_for_gpu, on_wait)
            # A long GPU wait must not bypass hash/source checks.
            report, refs = preflight(cfg, args.smoke)
            dump(output / "preflight.json", report)
        verify_source_snapshot(output)
        c1_cfg = {k: copy.deepcopy(cfg[k]) for k in ("seed", "data", "model", "train")}
        (output / "c1.yaml").write_text(yaml.safe_dump(c1_cfg, sort_keys=False))
        state.update(status="RUNNING", stage="train_c1", training_started=True)
        save()
        print(json.dumps(dict(stage="train_c1", token_length=188)), flush=True)
        c1_checkpoint = train_b0(c1_cfg, output / "c1", args.smoke)
        state["stage"] = "evaluate_c1"
        save()
        rows, checkpoint = evaluate_b0(c1_checkpoint)
        c1_result = write_evaluation(rows, checkpoint, output / "c1/evaluation")
        if c1_result["status"] != "complete" or abs(c1_result["metrics"]["macro_f1"] - checkpoint["validation_macro_f1"]) > 1e-8:
            raise ValueError("C1 best checkpoint evaluation mismatch or invalid metrics.")
        if target_set(rows) != target_set(refs["b0"]["rows"]):
            raise ValueError("C1/parent targets differ.")
        state["stage"] = "cache_embeddings"
        save()
        started = time.perf_counter()
        cache_embeddings(c1_checkpoint, output / "embeddings", batch_size=cfg["train"]["batch_size"])
        dump(output / "cache_metadata.json", dict(wall_seconds=time.perf_counter() - started,
                                                 encoder_type="conformer_epoch", embedding_dim=192))
        b1_cfg = copy.deepcopy(cfg["b1"])
        b1_cfg["seed"] = cfg["seed"]
        b1_cfg["data"].update(manifest=cfg["data"]["manifest"], embedding_root=str(output / "embeddings"))
        (output / "b1.yaml").write_text(yaml.safe_dump(b1_cfg, sort_keys=False))
        state["stage"] = "train_b1"
        save()
        print(json.dumps(dict(stage="train_b1", input_epochs=40, target_epochs=20)), flush=True)
        b1_checkpoint = train_b1(b1_cfg, output / "b1", args.smoke)
        state["stage"] = "evaluate_b1"
        save()
        b1_rows, b1_payload = evaluate_b1(b1_checkpoint)
        b1_result = write_evaluation(b1_rows, b1_payload, output / "b1/evaluation")
        if b1_result["status"] != "complete" or abs(b1_result["metrics"]["macro_f1"] - b1_payload["validation_macro_f1"]) > 1e-8:
            raise ValueError("B1 best checkpoint evaluation mismatch or invalid metrics.")
        if target_set(b1_rows) != target_set(refs["b1"]["rows"]) or target_set(b1_rows) != target_set(rows):
            raise ValueError("Downstream B1/parent/C1 evaluated targets differ.")
        comparison = dict(smoke=args.smoke, seed=cfg["seed"], evaluation_scope="validation-only",
                          identical_validation_epochs=True, valid_epochs=len(rows),
                          parent_run=cfg["experiment"]["parent_run"],
                          note="Technical smoke only." if args.smoke else "Single-seed encoder replacement screening; same BiLSTM40, no superiority claim.")
        for key, parent_kind, result in (("single_epoch", "b0", c1_result), ("bilstm40", "b1", b1_result)):
            parent_metrics = evaluate_predictions(refs[parent_kind]["rows"])
            comparison[key] = dict(parent=parent_metrics, candidate=result["metrics"],
                                   macro_f1_delta=result["metrics"]["macro_f1"] - parent_metrics["macro_f1"],
                                   reference_identity=refs[parent_kind]["identity"])
        dump(output / "comparison.json", comparison)
        state.update(status="SMOKE_COMPLETE" if args.smoke else "COMPLETE", stage="complete",
                     finished_at=datetime.now(timezone.utc).isoformat())
        save()
        print(json.dumps(state), flush=True)
    except BaseException as exc:
        state.update(status="STOPPED" if isinstance(exc, KeyboardInterrupt) else "FAILED",
                     error=str(exc), finished_at=datetime.now(timezone.utc).isoformat())
        save()
        (output / "traceback.txt").write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
