"""Preflight, then sequentially run the controlled T40/T80 Transformer pair."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def save_json(path, payload):
    path.write_text(json.dumps(payload, indent=2) + "\n")


def resolve(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def read_configs(paths):
    import yaml
    from psg_only.windows import context_options
    configs = [yaml.safe_load(resolve(path).read_text()) for path in paths]
    for expected_context, cfg in zip((40, 80), configs):
        if set(cfg) != {"seed", "experiment", "data", "model", "train"}:
            raise ValueError("Expected seed/experiment/data/model/train configuration sections.")
        data = cfg["data"]
        if set(data) != {"manifest", "embedding_root", "input_epochs", "target_epochs", "left_context", "stride"}:
            raise ValueError("Unknown/missing data options.")
        context_options(data)
        if data["input_epochs"] != expected_context:
            raise ValueError("Provide T40 followed by T80.")
        for key in ("manifest", "embedding_root"):
            data[key] = str(resolve(data[key]))
        for key in ("parent_run", "baseline_bilstm_run"):
            cfg["experiment"][key] = str(resolve(cfg["experiment"][key]))
        model = cfg["model"]
        fixed = dict(type="temporal_transformer", input_dim=192, d_model=192,
                     activation="gelu", norm_first=True, batch_first=True,
                     input_norm=True, final_norm=True, position_encoding="sinusoidal",
                     position_source="recording_epoch_index", causal=False)
        if set(model) != set(fixed) | {"layers", "num_heads", "ff_dim", "dropout"}:
            raise ValueError("Unknown/missing Transformer model options.")
        if any(model[k] != value for k, value in fixed.items()):
            raise ValueError("Unsupported Transformer option; no config fields may be silently ignored.")
        for key in ("layers", "num_heads", "ff_dim"):
            if type(model[key]) is not int or model[key] < 1:
                raise ValueError("Transformer dimensions must be positive integers.")
        if 192 % model["num_heads"] or not 0 <= model["dropout"] < 1:
            raise ValueError("Invalid heads/dropout.")
        fixed_train = dict(optimizer="adamw", loss="class_weighted_cross_entropy",
                           class_weight_power=-0.5, ignore_index=-100,
                           scheduler="none", selection_metric="validation_macro_f1")
        parent_keys = {"epochs", "batch_size", "learning_rate", "weight_decay",
                       "early_stopping_patience", "grad_clip_norm", "amp"}
        if set(cfg["train"]) != parent_keys | set(fixed_train):
            raise ValueError("Unknown/missing training options.")
        if any(cfg["train"][k] != v for k, v in fixed_train.items()):
            raise ValueError("Retain weighted CE/AdamW and the registered selection policy.")
    a, b = configs
    for key in ("model", "train", "seed"):
        if a[key] != b[key]:
            raise ValueError("T40/T80 must share model, train and seed.")
    for key in ("manifest", "embedding_root"):
        if a["data"][key] != b["data"][key]:
            raise ValueError("T40/T80 must share frozen embeddings and manifest.")
    if a["experiment"]["parent_run"] != b["experiment"]["parent_run"]:
        raise ValueError("The context pair must use one parent.")
    if a["experiment"]["alias"] != "T40" or b["experiment"]["alias"] != "T80":
        raise ValueError("Expected aliases T40 and T80.")
    return configs


def metric_summary(metrics):
    return {key: metrics[key] for key in ("macro_f1", "accuracy", "cohen_kappa",
                                         "transition_macro_f1", "stable_macro_f1",
                                         "transition_rate_error", "per_class", "confusion_matrix")}


def check_reference_metrics(actual, saved):
    for key in ("macro_f1", "accuracy", "cohen_kappa", "transition_macro_f1",
                "stable_macro_f1", "transition_rate_error"):
        if actual.get(key) is None or saved.get(key) is None:
            if actual.get(key) != saved.get(key):
                raise ValueError("Reference metric mismatch: " + key)
        elif abs(actual[key] - saved[key]) > 1e-8:
            raise ValueError("Reference metric mismatch: " + key)
    if actual["confusion_matrix"] != saved["confusion_matrix"]:
        raise ValueError("Reference confusion matrix mismatch.")


def target_set(rows):
    return {(str(row["subject_id"]), int(row["epoch_index"]), int(row["target"])) for row in rows}


def expected_targets(dataset):
    import numpy as np
    return {(sid, int(i), int(dataset._load(sid)[1][i]))
            for sid in dataset.groups for i in np.flatnonzero(dataset._load(sid)[2])}


def preflight(configs, smoke):
    import numpy as np
    import pandas as pd
    import torch
    from psg_only.constants import CANONICAL_CLASSES, CLASS_ORDER_VERSION
    from psg_only.metrics import evaluate_predictions
    from psg_only.provenance import file_hash
    from psg_only.train import class_weights, config_hash
    from psg_only.transformer import datasets
    from psg_only.windows import stitch_predictions

    report, prepared, references = {}, {}, {}
    common_targets = None
    for cfg in configs:
        alias = cfg["experiment"]["alias"]
        train_set, val_set = datasets(cfg, smoke)
        prepared[alias] = (train_set, val_set)
        parent = Path(cfg["experiment"]["parent_run"])
        reference = Path(cfg["experiment"]["baseline_bilstm_run"]) / "b1"
        checkpoint_path = reference / "best.pt"
        cp = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        old = cp["config"]
        if cp["model_type"] != "b1" or old.get("smoke"):
            raise ValueError("Reference must be a completed full BiLSTM run.")
        if tuple(cp.get("class_order", ())) != CANONICAL_CLASSES or cp.get("class_order_version") != CLASS_ORDER_VERSION:
            raise ValueError("Reference class order mismatch.")
        if config_hash(old) != cp["config_hash"]:
            raise ValueError("Reference checkpoint config hash mismatch.")
        if cfg["seed"] != old["seed"]:
            raise ValueError("Paired seed differs from the available BiLSTM comparator.")
        if old["data"]["input_epochs"] != cfg["data"]["input_epochs"]:
            raise ValueError("BiLSTM reference context differs.")
        if old["manifest_hash"] != train_set.manifest_hash or old["embedding_provenance"] != train_set.provenance:
            raise ValueError("Reference manifest/embedding provenance differs.")
        if Path(old["data"]["embedding_root"]).resolve() != Path(cfg["data"]["embedding_root"]):
            raise ValueError("Use the reference's exact frozen embedding directory.")
        if file_hash(parent / "b0/best.pt") != train_set.provenance["checkpoint_sha256"]:
            raise ValueError("Parent B0 checkpoint no longer matches cache.")
        b0 = torch.load(parent / "b0/best.pt", map_location="cpu", weights_only=False)
        if file_hash(b0["config"]["data"]["stats"]) != train_set.provenance["stats_hash"]:
            raise ValueError("Parent train normalization changed.")
        if any(cfg["train"].get(k) != v for k, v in old["train"].items()):
            raise ValueError("Keep the reference B1 training budget and optimizer settings.")
        saved = json.loads((reference / "evaluation/results.json").read_text())
        if saved.get("smoke") or saved.get("status") != "complete":
            raise ValueError("Reference evaluation is not a completed full result.")
        if saved["checkpoint_sha256"] != file_hash(checkpoint_path):
            raise ValueError("Reference results do not identify this checkpoint.")
        rows = pd.read_csv(reference / "evaluation/predictions.csv", dtype={"subject_id": str}).to_dict("records")
        rows = stitch_predictions(rows)
        check_reference_metrics(evaluate_predictions(rows), saved["metrics"])
        selected = [r for r in rows if str(r["subject_id"]) in val_set.groups] if smoke else rows
        expected = expected_targets(val_set)
        if target_set(selected) != expected or len(selected) != len(expected):
            raise ValueError("Reference/candidate target coverage or labels differ.")
        if common_targets is not None and expected != common_targets:
            raise ValueError("T40/T80 evaluated target sets differ.")
        common_targets = expected
        counts = np.bincount(train_set.canonical_labels(), minlength=4)
        weights = class_weights(train_set.canonical_labels())
        lengths = np.asarray(list(train_set.groups.values()) + list(val_set.groups.values()))
        report[alias] = dict(
            parent_run=str(parent), baseline_checkpoint=str(checkpoint_path),
            baseline_checkpoint_sha256=file_hash(checkpoint_path),
            baseline_predictions_sha256=file_hash(reference / "evaluation/predictions.csv"),
            seed=cfg["seed"], input_epochs=cfg["data"]["input_epochs"],
            target_epochs=20, stride=20, manifest_hash=train_set.manifest_hash,
            embedding_provenance=train_set.provenance, train_class_counts=counts.tolist(),
            class_weights=weights.tolist(),
            train=dict(subjects=len(train_set.groups), windows=len(train_set), valid_epochs=int(counts.sum())),
            val=dict(subjects=len(val_set.groups), windows=len(val_set), valid_epochs=len(expected)),
            max_optimizer_steps=((len(train_set) + cfg["train"]["batch_size"] - 1) // cfg["train"]["batch_size"]) * cfg["train"]["epochs"],
            recording_epochs=dict(min=int(lengths.min()), median=float(np.median(lengths)),
                                  p95=float(np.percentile(lengths, 95)), max=int(lengths.max())),
            reference_metrics=metric_summary(evaluate_predictions(selected)),
        )
        references[alias] = dict(rows=selected, checkpoint=str(checkpoint_path), saved=saved)
    return report, prepared, references


def capture_sources(output, runner_path=None):
    import torch
    from psg_only.provenance import file_hash
    files = sorted((ROOT / "src/psg_only").glob("*.py")) + [Path(runner_path or __file__).resolve()]
    hashes = {}
    for path in files:
        relative = path.relative_to(ROOT)
        target = output / "code" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[str(relative)] = file_hash(path)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    diff = subprocess.run(["git", "diff", "--", "src/psg_only", "scripts", "configs", "tests"],
                          cwd=ROOT, capture_output=True, text=True)
    (output / "tracked_changes.patch").write_text(diff.stdout)
    save_json(output / "environment.json", dict(
        python=sys.version, executable=sys.executable, torch=torch.__version__, cuda=torch.version.cuda,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        float32_precision="ieee_tf32_disabled",
        git_revision=commit.stdout.strip(), source_hashes=hashes,
        note="Code snapshots include untracked model/runner files; dirty patch preserves tracked changes.",
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs=2, default=["configs/t40_transformer.yaml", "configs/t80_transformer.yaml"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--run", action="store_true", help="Execute after preflight; otherwise only prepare.")
    parser.add_argument("--smoke", action="store_true", help="8 train/2 val subjects; at most 2 updates per model.")
    device = parser.add_mutually_exclusive_group()
    device.add_argument("--gpu", type=int)
    device.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if args.run and args.gpu is None and not args.cpu:
        parser.error("--run requires --gpu INDEX or --cpu.")
    if args.run and not args.smoke and args.gpu is None:
        parser.error("Full experiment requires --gpu INDEX; CPU is for smoke.")
    if args.gpu is not None and args.gpu < 0:
        parser.error("GPU index must be nonnegative.")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu) if args.gpu is not None else ""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("OMP_NUM_THREADS", "4")

    import torch
    import yaml
    from psg_only.evaluate import evaluate_b1, write_evaluation
    from psg_only.metrics import evaluate_predictions
    from psg_only.transformer import configure_precision, evaluate_transformer, train_transformer
    configure_precision()

    output = resolve(args.output) if args.output else ROOT / "artifacts/experiments" / datetime.now(timezone.utc).strftime("transformer_pair_%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    state = dict(status="PREPARING", started_at=datetime.now(timezone.utc).isoformat(),
                 smoke=args.smoke, seed=None, evaluation_scope="validation-only",
                 hypothesis="Transformer versus BiLSTM at matched 40/80-epoch context on identical frozen B0 embeddings.",
                 sequence=["T40", "T80"], gpu=args.gpu,
                 output=str(output), training_started=False)
    def save():
        save_json(output / "experiment.json", state)
    save()
    try:
        configs = read_configs(args.configs)
        state["seed"] = configs[0]["seed"]
        for cfg in configs:
            (output / (cfg["experiment"]["alias"] + ".yaml")).write_text(yaml.safe_dump(cfg, sort_keys=False))
        capture_sources(output)
        report, prepared, references = preflight(configs, args.smoke)
        save_json(output / "preflight.json", report)
        state.update(status="PREPARED", stage="preflight", parent_run=configs[0]["experiment"]["parent_run"])
        save()
        print(json.dumps(dict(output=str(output), status=state["status"], preflight=report)), flush=True)
        if not args.run:
            return
        if args.gpu is not None:
            if not torch.cuda.is_available():
                raise RuntimeError("Requested GPU is unavailable; refusing silent CPU fallback.")
            probe = subprocess.run(
                ["nvidia-smi", "-i", str(args.gpu), "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                capture_output=True, text=True,
            )
            if probe.returncode:
                raise RuntimeError("Cannot check assigned GPU: " + probe.stderr.strip())
            if probe.stdout.strip():
                raise RuntimeError("GPU already has compute processes; retry when it is free. No processes were stopped.")
        state.update(status="RUNNING", stage="baseline_evaluation")
        save()
        baseline = {}
        for cfg in configs:
            alias = cfg["experiment"]["alias"]
            ref = references[alias]
            if args.smoke:
                rows = ref["rows"]
                note = "Stored full-run predictions restricted to smoke subjects; no baseline inference rerun."
            else:
                rows, cp = evaluate_b1(ref["checkpoint"])
                check_reference_metrics(evaluate_predictions(rows), ref["saved"]["metrics"])
                write_evaluation(rows, cp, output / ("R" + alias[1:]) / "evaluation")
                note = "Baseline checkpoint re-evaluated with the current evaluator; saved metrics reproduced."
            baseline["R" + alias[1:]] = metric_summary(evaluate_predictions(rows))
            ref["rows"] = rows
            save_json(output / ("R" + alias[1:] + "_reference.json"), dict(metrics=baseline["R" + alias[1:]], note=note))

        candidate = {}
        for cfg in configs:
            alias = cfg["experiment"]["alias"]
            state.update(stage="train_" + alias, training_started=True)
            save()
            cp_path = train_transformer(cfg, output / alias, args.smoke, prepared[alias])
            state["stage"] = "evaluate_" + alias
            save()
            rows, cp = evaluate_transformer(cp_path)
            if target_set(rows) != target_set(references[alias]["rows"]):
                raise ValueError("Candidate/reference evaluated targets differ.")
            result = write_evaluation(rows, cp, output / alias / "evaluation")
            if result["status"] != "complete":
                raise RuntimeError("Candidate evaluator did not produce valid metrics.")
            save_json(output / alias / "evaluation/timing.json", cp["evaluation_timing"])
            candidate[alias] = metric_summary(result["metrics"])
            save_json(output / alias / "baseline_comparison.json", dict(
                baseline=baseline["R" + alias[1:]], candidate=candidate[alias],
                macro_f1_delta=candidate[alias]["macro_f1"] - baseline["R" + alias[1:]]["macro_f1"],
                matched_targets=len(rows), smoke=args.smoke,
            ))
            print(json.dumps(dict(alias=alias, status="COMPLETE", metrics=candidate[alias])), flush=True)
        transformer_delta = candidate["T80"]["macro_f1"] - candidate["T40"]["macro_f1"]
        recurrent_delta = baseline["R80"]["macro_f1"] - baseline["R40"]["macro_f1"]
        save_json(output / "comparison.json", dict(
            baseline=baseline, transformer=candidate, smoke=args.smoke,
            transformer_80_minus_40=transformer_delta, bilstm_80_minus_40=recurrent_delta,
            architecture_context_interaction=transformer_delta - recurrent_delta,
            interpretation="Technical smoke only." if args.smoke else "Single-seed validation screening; no statistical winner claim.",
        ))
        state.update(status="SMOKE_COMPLETE" if args.smoke else "COMPLETE",
                     stage="complete", finished_at=datetime.now(timezone.utc).isoformat())
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
