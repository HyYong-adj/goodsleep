"""Training/evaluation for the T40/T80 controlled frozen-embedding experiments."""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader

from .data import EmbeddingWindowDataset
from .provenance import file_hash
from .train import WeightedLossMeter, _checkpoint, class_weights, device_for_run, seed_everything
from .transformer_model import transformer_from_config
from .windows import context_options, stitch_predictions


class PositionedEmbeddingWindowDataset(EmbeddingWindowDataset):
    """Keep the existing target grid and gaps; append real input epoch positions."""
    def __getitem__(self, index):
        item = super().__getitem__(index)
        ref = self.windows[index]
        positions = torch.arange(self.input_epochs, dtype=torch.long) + ref.target_start - self.left_context
        positions[~item[2]] = -1
        return (*item, positions)

    def canonical_labels(self):
        return np.concatenate([
            self._load(sid)[1][self._load(sid)[2]]
            for sid in self.groups
        ]).astype(np.int64)


def datasets(config, smoke=False):
    data = config["data"]
    result = [
        PositionedEmbeddingWindowDataset(
            data["manifest"], data["embedding_root"], split,
            subject_limit=limit if smoke else None, **context_options(data),
        ) for split, limit in (("train", 8), ("val", 2))
    ]
    if result[0].provenance != result[1].provenance:
        raise ValueError("Train/val embedding provenance differs.")
    return result


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def configure_precision():
    # Default cuDNN TF32 changed one borderline BiLSTM prediction in this environment.
    # IEEE FP32 reproduces both saved references; AMP training remains explicitly enabled.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


@torch.no_grad()
def validation(model, loader, criterion, device, collect_rows=False):
    from .evaluate import _row
    model.eval()
    truth, pred, rows = [], [], []
    meter = WeightedLossMeter(criterion.weight)
    for x, y, iv, tv, subjects, indexes, positions in loader:
        if not tv.any():
            continue
        logits = model(x.to(device), iv.to(device), positions.to(device))
        if not torch.isfinite(logits).all():
            raise RuntimeError("Non-finite Transformer validation logits.")
        logits = logits.float().cpu()
        targets = y.masked_fill(~tv, -100)
        meter.add_logits(logits, targets)
        predicted = logits.argmax(-1)
        truth.extend(y[tv].tolist())
        pred.extend(predicted[tv].tolist())
        if collect_rows:
            probabilities = logits.softmax(-1)
            for batch, position in tv.nonzero().tolist():
                rows.append(_row(
                    subjects[batch], int(indexes[batch, position]), int(y[batch, position]),
                    int(predicted[batch, position]), probabilities[batch, position],
                    "temporal_transformer",
                ))
    if not truth:
        raise ValueError("No valid validation targets.")
    score = float(f1_score(truth, pred, labels=np.arange(4), average="macro", zero_division=0))
    return score, meter.mean(), rows


def train_transformer(config, output, smoke=False, prepared_datasets=None):
    configure_precision()
    config = copy.deepcopy(config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    config["smoke"] = smoke
    config["manifest_hash"] = file_hash(config["data"]["manifest"])
    seed_everything(int(config["seed"]))
    train_set, val_set = prepared_datasets or datasets(config, smoke)
    if train_set.manifest_hash != config["manifest_hash"]:
        raise ValueError("Manifest changed after preflight.")
    config["embedding_provenance"] = train_set.provenance
    labels = train_set.canonical_labels()
    weights = class_weights(labels)
    config["class_weights"] = weights.tolist()
    config["train_class_counts"] = np.bincount(labels, minlength=4).tolist()
    config["attention_backend"] = "math_sdpa"
    config["float32_precision"] = "ieee_tf32_disabled"
    cfg = config["train"]
    generator = torch.Generator().manual_seed(int(config["seed"]))
    train_loader = DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=0, generator=generator)
    val_loader = DataLoader(val_set, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)
    device = device_for_run()
    model = transformer_from_config(config).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device), ignore_index=-100)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"],
                                  weight_decay=cfg["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and cfg["amp"])
    (output / "resolved_config.json").write_text(json.dumps(config, indent=2))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    started = time.perf_counter()
    best, stale, step, samples_seen, targets_seen = -1., 0, 0, 0, 0
    history = []
    for epoch in range(1 if smoke else cfg["epochs"]):
        model.train()
        meter = WeightedLossMeter(weights)
        skipped = 0
        epoch_started = time.perf_counter()
        for x, y, iv, tv, _, _, positions in train_loader:
            if not tv.any():
                skipped += 1
                continue
            targets = y.masked_fill(~tv, -100).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                logits = model(x.to(device), iv.to(device), positions.to(device))
                loss = criterion(logits.reshape(-1, 4), targets.reshape(-1))
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite Transformer training loss.")
            meter.add_mean(loss, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip_norm"], error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            step += 1
            samples_seen += len(x)
            targets_seen += int(tv.sum())
            if smoke and step >= 2:
                break
        synchronize(device)
        train_seconds = time.perf_counter() - epoch_started
        val_started = time.perf_counter()
        score, val_loss, _ = validation(model, val_loader, criterion, device)
        synchronize(device)
        record = dict(
            epoch=epoch, optimizer_steps=step, validation_macro_f1=score,
            train_loss=meter.mean(), validation_loss=val_loss,
            loss_aggregation="weighted CE sum / valid target weight sum",
            train_seconds=train_seconds, validation_seconds=time.perf_counter() - val_started,
            skipped_all_invalid_batches=skipped, windows_seen=samples_seen, valid_targets_seen=targets_seen,
        )
        if meter.mean() is None:
            raise ValueError("No valid training targets.")
        history.append(record)
        with (output / "history.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(dict(alias=config["experiment"]["alias"], **record)), flush=True)
        if score > best:
            best, stale = score, 0
            _checkpoint(output / "best.pt", model, config, "temporal_transformer", epoch, score)
        else:
            stale += 1
        if smoke or stale >= cfg["early_stopping_patience"]:
            break
    synchronize(device)
    metadata = dict(
        model_type="temporal_transformer", seed=config["seed"], smoke=smoke,
        best_validation_macro_f1=best, epochs_completed=len(history), optimizer_steps=step,
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        total_parameters=sum(p.numel() for p in model.parameters()),
        training_and_validation_seconds=time.perf_counter() - started,
        peak_cuda_memory_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        device=str(device), device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        windows_seen=samples_seen, valid_targets_seen=targets_seen,
    )
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    return output / "best.pt"


def evaluate_transformer(checkpoint_path, split="val"):
    from .evaluate import load_model
    configure_precision()
    if split != "val":
        raise ValueError("Transformer experiment evaluation is validation-only.")
    device = device_for_run()
    model, checkpoint = load_model(checkpoint_path, device)
    config = checkpoint["config"]
    if file_hash(config["data"]["manifest"]) != config["manifest_hash"]:
        raise ValueError("Transformer manifest changed since training.")
    data = config["data"]
    dataset = PositionedEmbeddingWindowDataset(
        data["manifest"], data["embedding_root"], split,
        subject_limit=2 if config.get("smoke") else None, **context_options(data),
    )
    if dataset.provenance != config["embedding_provenance"]:
        raise ValueError("Transformer evaluation cache differs from training.")
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(config["class_weights"], device=device), ignore_index=-100,
    )
    loader = DataLoader(dataset, batch_size=config["train"]["batch_size"], shuffle=False, num_workers=0)
    synchronize(device)
    started = time.perf_counter()
    score, loss, rows = validation(model, loader, criterion, device, collect_rows=True)
    synchronize(device)
    if abs(score - checkpoint["validation_macro_f1"]) > 1e-8:
        raise ValueError("Reloaded checkpoint evaluation does not match model selection.")
    checkpoint["evaluation_timing"] = {
        "wall_seconds_including_loader_metrics": time.perf_counter() - started,
        "subjects": len(dataset.groups), "windows": len(dataset),
        "note": "Window inference + loader + metric/row assembly, not pure full-night forward latency.",
        "validation_loss": loss,
    }
    return stitch_predictions(rows, dataset.expected_keys()), checkpoint
