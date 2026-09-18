from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .constants import CANONICAL_CLASSES, CLASS_ORDER_VERSION
from .data import CachedMelEpochDataset, EmbeddingWindowDataset
from .metrics import evaluate_predictions
from .models import B0Model, B1Model, window_hours
from .windows import stitch_predictions, context_options
from .provenance import file_hash
from .train import device_for_run


def load_model(checkpoint_path: str | Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("class_order_version") != CLASS_ORDER_VERSION or tuple(checkpoint.get("class_order", ())) != CANONICAL_CLASSES:
        raise ValueError("Checkpoint class order is not canonical.")
    if checkpoint["model_type"] == "b0":
        model = B0Model(float(checkpoint["config"]["model"]["dropout"]))
    elif checkpoint["model_type"] == "conformer_epoch":
        from .conformer import conformer_from_config
        from .transformer import configure_precision
        configure_precision()
        model = conformer_from_config(checkpoint["config"])
    elif checkpoint["model_type"] == "b1":
        cfg = checkpoint["config"]["model"]
        model = B1Model(int(cfg["hidden_dim"]), int(cfg["layers"]), float(cfg["dropout"]),
                        input_epochs=context_options(checkpoint["config"].get("data", {}))["input_epochs"],
                        time_feature=bool(cfg.get("time_feature", False)))
    elif checkpoint["model_type"] == "temporal_transformer":
        from .transformer_model import transformer_from_config
        model = transformer_from_config(checkpoint["config"])
    else:
        raise ValueError("Unknown checkpoint model type.")
    model.load_state_dict(checkpoint["state_dict"])
    checkpoint["checkpoint_sha256"] = hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest()
    return model.to(device).eval(), checkpoint


@torch.no_grad()
def evaluate_b0(checkpoint_path: str | Path, split: str = "val") -> tuple[list[dict], dict]:
    device = device_for_run()
    model, checkpoint = load_model(checkpoint_path, device)
    data = checkpoint["config"]["data"]
    if checkpoint["config"].get("manifest_hash") != file_hash(data["manifest"]) or checkpoint["config"].get("stats_hash") != file_hash(data["stats"]):
        raise ValueError("B0 manifest/stats changed since training.")
    # 학습에 쓴 정규화를 평가에서도 그대로 써야 한다. night_norm 으로 학습한
    # encoder 를 전역 통계로 평가하면 입력 분포가 어긋나 결과가 무의미해진다.
    dataset = CachedMelEpochDataset(data["manifest"], data["cache_root"], data["stats"], split,
                                    2 if checkpoint["config"].get("smoke") else None,
                                    night_norm=bool(data.get("night_norm", False)))
    rows = []
    batch_size = checkpoint["config"]["train"]["batch_size"] if checkpoint["model_type"] == "conformer_epoch" else 32
    for x, y, subjects, epochs, valid in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0):
        logits, _ = model(x.to(device))
        probabilities = torch.softmax(logits, dim=1).cpu()
        predictions = probabilities.argmax(1)
        for i, is_valid in enumerate(valid):
            if not bool(is_valid):
                continue
            rows.append(_row(subjects[i], int(epochs[i]), int(y[i]), int(predictions[i]), probabilities[i], checkpoint["model_type"]))
    return stitch_predictions(rows, dataset.expected_keys()), checkpoint


@torch.no_grad()
def evaluate_b1(checkpoint_path: str | Path, split: str = "val") -> tuple[list[dict], dict]:
    device = device_for_run()
    model, checkpoint = load_model(checkpoint_path, device)
    data = checkpoint["config"]["data"]
    dataset = EmbeddingWindowDataset(data["manifest"], data["embedding_root"], split, subject_limit=2 if checkpoint["config"].get("smoke") else None, **context_options(data))
    if dataset.provenance != checkpoint["config"].get("embedding_provenance"):
        raise ValueError("B1 evaluation embeddings differ from training provenance.")
    rows = []
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    for x, y, input_valid, target_valid, subjects, indexes in loader:
        hours = window_hours(indexes, input_valid, model.left_context).to(device) if model.time_proj is not None else None
        probabilities = torch.softmax(model(x.to(device), input_valid.to(device), hours), dim=-1).cpu()
        predictions = probabilities.argmax(-1)
        for batch in range(x.shape[0]):
            for position in range(20):
                if bool(target_valid[batch, position]):
                    rows.append(_row(subjects[batch], int(indexes[batch, position]), int(y[batch, position]), int(predictions[batch, position]), probabilities[batch, position], "b1"))
    return stitch_predictions(rows, dataset.expected_keys()), checkpoint


def _row(subject_id: str, epoch_index: int, target: int, prediction: int, probability: torch.Tensor, model_version: str) -> dict:
    return {
        "subject_id": str(subject_id),
        "epoch_index": epoch_index,
        "target": target,
        "prediction": prediction,
        "prob_wake": float(probability[0]),
        "prob_rem": float(probability[1]),
        "prob_light": float(probability[2]),
        "prob_deep": float(probability[3]),
        "valid": True,
        "model_version": model_version,
        "class_order_version": CLASS_ORDER_VERSION,
    }


def write_evaluation(rows: list[dict], checkpoint: dict, output_dir: str | Path) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    pd.DataFrame(rows).to_csv(output / "predictions.csv", index=False)
    metrics = evaluate_predictions(rows)
    source_files = sorted(Path(__file__).parent.glob("*.py"))
    code_hash = hashlib.sha256(b"".join(item.read_bytes() for item in source_files)).hexdigest()
    data = checkpoint["config"]["data"]
    logical_data_name = Path(data.get("cache_root", data.get("embedding_root", "unknown"))).name
    payload = {
        "schema_version": "1.0",
        "status": metrics.pop("status"),
        "experiment_id": output.name,
        "seed": checkpoint["config"]["seed"],
        "model_type": checkpoint["model_type"],
        "checkpoint_validation_macro_f1": checkpoint["validation_macro_f1"],
        "checkpoint_sha256": checkpoint.get("checkpoint_sha256"),
        "config_hash": checkpoint["config_hash"],
        "code_hash": code_hash,
        "logical_data_name": logical_data_name,
        "class_order": list(CANONICAL_CLASSES),
        "class_order_version": CLASS_ORDER_VERSION,
        "subjects": len({row["subject_id"] for row in rows}),
        "epochs": len(rows),
        "metrics": metrics,
        "artifact_location": str(output),
        "evaluation_scope": "validation-only",
        "smoke": checkpoint["config"].get("smoke", False),
        "input_epochs": checkpoint["config"]["data"].get("input_epochs") if checkpoint["model_type"] in {"b1", "temporal_transformer"} else 1,
        "embedding_provenance": checkpoint["config"].get("embedding_provenance"),
    }
    (output / "results.json").write_text(json.dumps(payload, indent=2))
    return payload
