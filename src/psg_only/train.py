from __future__ import annotations

import hashlib
import json
import math
import random
import os
import copy
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from .constants import CANONICAL_CLASSES, CLASS_ORDER_VERSION
from .data import CachedMelEpochDataset, EmbeddingWindowDataset, validate_inputs
from .provenance import file_hash
from .models import B0Model, B1Model, window_hours
from .windows import context_options


def load_config(path: str | Path) -> dict:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    context_options(config["data"])
    for key, expected in {"input_dim": 192, "embedding_dim": 192, "bidirectional": True}.items():
        if key in config["model"] and config["model"][key] != expected:
            raise ValueError(f"Unsupported {key}; expected {expected}.")
    return config


def config_hash(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def class_weights(labels: np.ndarray, multipliers: dict | None = None) -> torch.Tensor:
    """Inverse-sqrt class weights, optionally scaled per class before renormalising.

    ``multipliers`` is keyed by canonical class name (A4 uses ``{"REM": 2.0}``).
    Renormalising after the multiplier keeps the mean weight at 1 so the loss
    scale stays comparable to the parent run.
    """
    counts = np.bincount(labels, minlength=4).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError(f"Missing training class: {counts.tolist()}")
    weights = counts ** -0.5
    for name, factor in (multipliers or {}).items():
        if name not in CANONICAL_CLASSES:
            raise ValueError(f"Unknown class in class_weight_multipliers: {name!r}")
        weights[CANONICAL_CLASSES.index(name)] *= float(factor)
    weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def build_scheduler(optimizer, train_cfg: dict, max_epochs: int):
    """Per-epoch LR schedule. ``none`` (default) preserves the parent behaviour."""
    name = str(train_cfg.get("scheduler", "none")).lower()
    if name in {"none", ""}:
        return None
    if name != "cosine":
        raise ValueError(f"Unsupported scheduler: {name!r}")
    warmup = int(train_cfg.get("warmup_epochs", 0))
    if warmup >= max_epochs:
        raise ValueError("warmup_epochs must be smaller than the epoch budget.")

    def factor(epoch: int) -> float:
        if epoch < warmup:
            return float(epoch + 1) / float(max(1, warmup))
        progress = (epoch - warmup) / max(1, max_epochs - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def spec_augment(x: torch.Tensor, freq_mask: int = 8, time_mask: int = 150, gain_std: float = 0.25) -> torch.Tensor:
    """정규화된 log-Mel [B,1,F,T]에 주파수/시간 마스킹과 gain jitter 를 적용한다.

    정규화 이후이므로 0 으로 마스킹하는 것은 평균값으로 대체하는 것과 같다.
    학습 split 에만 적용하고 validation/embedding 생성에는 적용하지 않는다.
    """
    if x.ndim != 4 or x.shape[1] != 1:
        raise ValueError("spec_augment expects [batch, 1, mel, frames].")
    batch, _, n_mels, n_frames = x.shape
    if freq_mask >= n_mels or time_mask >= n_frames:
        raise ValueError("Mask width must be smaller than the spectrogram.")
    x = x.clone()
    for index in range(batch):
        width = random.randint(0, freq_mask)
        if width:
            start = random.randint(0, n_mels - width)
            x[index, :, start:start + width, :] = 0.0
        width = random.randint(0, time_mask)
        if width:
            start = random.randint(0, n_frames - width)
            x[index, :, :, start:start + width] = 0.0
    if gain_std > 0:
        x = x + torch.randn(batch, 1, 1, 1, device=x.device, dtype=x.dtype) * gain_std
    return x


def device_for_run() -> torch.device:
    if torch.cuda.is_available() and "CUDA_VISIBLE_DEVICES" not in os.environ:
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES to your assigned GPU, or an empty string for CPU.")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _checkpoint(path: Path, model: nn.Module, config: dict, model_type: str, epoch: int, score: float) -> None:
    torch.save({
        "state_dict": model.state_dict(),
        "config": config,
        "config_hash": config_hash(config),
        "model_type": model_type,
        "class_order": list(CANONICAL_CLASSES),
        "class_order_version": CLASS_ORDER_VERSION,
        "epoch": epoch,
        "validation_macro_f1": score,
    }, path)


def train_b0(config: dict, output: Path, smoke: bool = False) -> Path:
    config = copy.deepcopy(config)
    model_type = config["model"].get("type", "b0")
    if model_type not in {"b0", "conformer_epoch"}:
        raise ValueError("Unsupported single-epoch model type.")
    if model_type == "conformer_epoch":
        from .transformer import configure_precision
        configure_precision()
        config["float32_precision"] = "ieee_tf32_disabled"
        config["attention_backend"] = "math_sdpa"
    config["smoke"] = smoke
    config["manifest_hash"] = file_hash(config["data"]["manifest"])
    seed_everything(int(config["seed"]))
    data = config["data"]
    validate_inputs(data["manifest"], data["cache_root"], data["stats"])
    config["stats_hash"] = file_hash(data["stats"])
    limit = 8 if smoke else None
    val_limit = 2 if smoke else None
    night_norm = bool(data.get("night_norm", False))
    train_set = CachedMelEpochDataset(data["manifest"], data["cache_root"], data["stats"], "train", limit, night_norm=night_norm)
    val_set = CachedMelEpochDataset(data["manifest"], data["cache_root"], data["stats"], "val", val_limit, night_norm=night_norm)
    train_cfg = config["train"]
    if isinstance(train_set, EmbeddingWindowDataset):
        if train_set.provenance != val_set.provenance:
            raise ValueError("Train/val embedding provenance differs.")
        config["embedding_provenance"] = train_set.provenance
    train_loader = DataLoader(train_set, batch_size=int(train_cfg["batch_size"]), shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=int(train_cfg["batch_size"]), shuffle=False, num_workers=0)
    device = device_for_run()
    if model_type == "conformer_epoch":
        from .conformer import conformer_from_config
        model = conformer_from_config(config).to(device)
    else:
        model = B0Model(float(config["model"]["dropout"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["learning_rate"]), weight_decay=float(train_cfg["weight_decay"]))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and train_cfg.get("amp", False))
    criterion = nn.CrossEntropyLoss(weight=class_weights(train_set.labels()).to(device), ignore_index=-100)
    max_epochs = 1 if smoke else int(train_cfg["epochs"])
    max_steps = 2 if smoke else None
    augment = train_cfg.get("spec_augment") or None
    if augment is not None and not isinstance(augment, dict):
        raise ValueError("spec_augment must be a mapping of SpecAugment parameters.")
    best, stale, step = -1.0, 0, 0
    output.mkdir(parents=True, exist_ok=True)
    if (output / "best.pt").exists():
        raise FileExistsError("Run output already contains a checkpoint; choose a new output.")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for epoch in range(max_epochs):
        model.train()
        train_losses = WeightedLossMeter(criterion.weight)
        for x, y, _, _, _ in train_loader:
            if not (y != -100).any():
                continue
            x, y = x.to(device), y.to(device)
            if augment is not None:
                x = spec_augment(x, **augment)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                logits, _ = model(x)
                loss = criterion(logits, y)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite B0 loss.")
            train_losses.add_mean(loss, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["grad_clip_norm"]))
            scaler.step(optimizer)
            scaler.update()
            step += 1
            if max_steps and step >= max_steps:
                break
        score, validation_loss = _b0_macro_f1(model, val_loader, device, criterion)
        _log_epoch(output, epoch, step, score, train_losses.mean(), validation_loss)
        if score > best:
            best, stale = score, 0
            _checkpoint(output / "best.pt", model, config, model_type, epoch, score)
        else:
            stale += 1
        if max_steps or stale >= int(train_cfg["early_stopping_patience"]):
            break
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    (output / "run_metadata.json").write_text(json.dumps({
        "model_type": model_type, "seed": config["seed"], "smoke": smoke,
        "night_norm": night_norm, "spec_augment": augment,
        "class_order_version": CLASS_ORDER_VERSION, "best_validation_macro_f1": best,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "epochs_completed": epoch + 1, "optimizer_steps": step,
        "training_and_validation_seconds": time.perf_counter() - started,
        "peak_cuda_memory_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
    }, indent=2))
    return output / "best.pt"


@torch.no_grad()
def _b0_macro_f1(model: B0Model, loader: DataLoader, device: torch.device, criterion=None):
    from sklearn.metrics import f1_score
    model.eval(); truth, pred = [], []
    losses = WeightedLossMeter(criterion.weight) if criterion is not None else None
    for x, y, _, _, valid in loader:
        logits, _ = model(x.to(device)); mask = torch.as_tensor(valid, dtype=torch.bool)
        truth.extend(y[mask].tolist()); pred.extend(logits.cpu().argmax(1)[mask].tolist())
        if losses is not None:
            losses.add_logits(logits, y)
    model.train()
    score = float(f1_score(truth, pred, labels=np.arange(4), average="macro", zero_division=0))
    return score if losses is None else (score, losses.mean())


def train_b1(config: dict, output: Path, smoke: bool = False) -> Path:
    config = copy.deepcopy(config)
    config["smoke"] = smoke
    config["manifest_hash"] = file_hash(config["data"]["manifest"])
    seed_everything(int(config["seed"]))
    data, train_cfg = config["data"], config["train"]
    train_set = EmbeddingWindowDataset(data["manifest"], data["embedding_root"], "train", subject_limit=8 if smoke else None, **context_options(data))
    val_set = EmbeddingWindowDataset(data["manifest"], data["embedding_root"], "val", subject_limit=2 if smoke else None, **context_options(data))
    if isinstance(train_set, EmbeddingWindowDataset):
        if train_set.provenance != val_set.provenance:
            raise ValueError("Train/val embedding provenance differs.")
        config["embedding_provenance"] = train_set.provenance
    train_loader = DataLoader(train_set, batch_size=int(train_cfg["batch_size"]), shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=int(train_cfg["batch_size"]), shuffle=False, num_workers=0)
    device = device_for_run()
    model = B1Model(int(config["model"]["hidden_dim"]), int(config["model"]["layers"]), float(config["model"]["dropout"]),
                    input_epochs=context_options(data)["input_epochs"],
                    time_feature=bool(config["model"].get("time_feature", False))).to(device)
    labels = []
    for _, y, _, valid, _, _ in train_set:
        labels.extend(y[valid].tolist())
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and train_cfg.get("amp", False))
    criterion = nn.CrossEntropyLoss(
        weight=class_weights(np.asarray(labels), train_cfg.get("class_weight_multipliers")).to(device),
        ignore_index=-100,
        label_smoothing=float(train_cfg.get("label_smoothing", 0.0)),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["learning_rate"]), weight_decay=float(train_cfg["weight_decay"]))
    max_epochs = 1 if smoke else int(train_cfg["epochs"])
    scheduler = build_scheduler(optimizer, train_cfg, max_epochs)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "best.pt").exists():
        raise FileExistsError("Run output already contains a checkpoint; choose a new output.")
    best, stale, step = -1.0, 0, 0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    b1_started = time.perf_counter()
    for epoch in range(max_epochs):
        model.train()
        train_losses = WeightedLossMeter(criterion.weight)
        for x, y, valid_in, valid_out, _, target_indexes in train_loader:
            if not valid_out.any():
                continue
            optimizer.zero_grad(set_to_none=True)
            targets = y.to(device)
            targets[~valid_out.to(device)] = -100
            hours = window_hours(target_indexes, valid_in, model.left_context).to(device) if model.time_proj is not None else None
            with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                logits = model(x.to(device), valid_in.to(device), hours)
                loss = criterion(logits.reshape(-1, 4), targets.reshape(-1))
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite B1 loss.")
            train_losses.add_mean(loss, y if logits.ndim == 2 else targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["grad_clip_norm"]))
            scaler.step(optimizer)
            scaler.update()
            step += 1
            if smoke and step >= 2:
                break
        score, validation_loss = _b1_macro_f1(model, val_loader, device, criterion)
        _log_epoch(output, epoch, step, score, train_losses.mean(), validation_loss,
                   learning_rate=optimizer.param_groups[0]["lr"])
        if scheduler is not None:
            scheduler.step()
        if score > best:
            best, stale = score, 0
            _checkpoint(output / "best.pt", model, config, "b1", epoch, score)
        else:
            stale += 1
        if smoke or stale >= int(train_cfg["early_stopping_patience"]):
            break
    (output / "run_metadata.json").write_text(json.dumps({
        "model_type": "b1", "seed": config["seed"], "class_order_version": CLASS_ORDER_VERSION,
        "best_validation_macro_f1": best,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "epochs_completed": epoch + 1, "optimizer_steps": step,
        "training_and_validation_seconds": time.perf_counter() - b1_started,
        "peak_cuda_memory_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
    }, indent=2))
    return output / "best.pt"


@torch.no_grad()
def _b1_macro_f1(model: B1Model, loader: DataLoader, device: torch.device, criterion=None):
    from sklearn.metrics import f1_score
    model.eval(); truth, pred = [], []
    losses = WeightedLossMeter(criterion.weight) if criterion is not None else None
    for x, y, valid_in, valid_out, _, target_indexes in loader:
        hours = window_hours(target_indexes, valid_in, model.left_context).to(device) if model.time_proj is not None else None
        logits = model(x.to(device), valid_in.to(device), hours).cpu()
        truth.extend(y[valid_out].tolist()); pred.extend(logits.argmax(-1)[valid_out].tolist())
        if losses is not None:
            losses.add_logits(logits, y.masked_fill(~valid_out, -100))
    model.train()
    score = float(f1_score(truth, pred, labels=np.arange(4), average="macro", zero_division=0))
    return score if losses is None else (score, losses.mean())


def _log_epoch(output, epoch, steps, score, train_loss=None, validation_loss=None, learning_rate=None):
    record = {"epoch": epoch, "optimizer_steps": steps, "validation_macro_f1": score,
              "train_loss": train_loss, "validation_loss": validation_loss,
              "learning_rate": learning_rate,
              "loss_aggregation": "weighted CE sum / valid target weight sum"}
    with (output / "history.jsonl").open("a") as stream:
        stream.write(json.dumps(record) + "\n")
    print(json.dumps(record), flush=True)


class WeightedLossMeter:
    """Aggregate weighted CE without giving short/padded batches extra weight."""
    def __init__(self, weight):
        self.weight = weight.detach().float().cpu()
        self.numerator = self.denominator = 0.0

    def add_mean(self, loss, targets):
        targets = targets.detach().cpu().reshape(-1)
        valid = targets != -100
        denominator = float(self.weight[targets[valid]].sum())
        if denominator:
            self.numerator += float(loss.detach()) * denominator
            self.denominator += denominator

    def add_logits(self, logits, targets):
        logits = logits.detach().float().cpu().reshape(-1, 4)
        targets = targets.detach().cpu().reshape(-1)
        if not (targets != -100).any():
            return
        loss = nn.functional.cross_entropy(logits, targets, weight=self.weight, ignore_index=-100)
        self.add_mean(loss, targets)

    def mean(self):
        return self.numerator / self.denominator if self.denominator else None
