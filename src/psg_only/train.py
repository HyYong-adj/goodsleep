from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from .constants import CANONICAL_CLASSES, CLASS_ORDER_VERSION
from .data import CachedMelEpochDataset, EmbeddingWindowDataset
from .models import B0Model, B1Model


def load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def config_hash(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def class_weights(labels: np.ndarray) -> torch.Tensor:
    counts = np.bincount(labels, minlength=4).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError(f"Missing training class: {counts.tolist()}")
    weights = counts ** -0.5
    weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def device_for_run() -> torch.device:
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
    seed_everything(int(config["seed"]))
    data = config["data"]
    limit = 8 if smoke else None
    val_limit = 2 if smoke else None
    train_set = CachedMelEpochDataset(data["manifest"], data["cache_root"], data["stats"], "train", limit)
    val_set = CachedMelEpochDataset(data["manifest"], data["cache_root"], data["stats"], "val", val_limit)
    train_cfg = config["train"]
    train_loader = DataLoader(train_set, batch_size=int(train_cfg["batch_size"]), shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=int(train_cfg["batch_size"]), shuffle=False, num_workers=0)
    device = device_for_run()
    model = B0Model(float(config["model"]["dropout"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["learning_rate"]), weight_decay=float(train_cfg["weight_decay"]))
    criterion = nn.CrossEntropyLoss(weight=class_weights(train_set.labels()).to(device), ignore_index=-100)
    max_epochs = 1 if smoke else int(train_cfg["epochs"])
    max_steps = 2 if smoke else None
    best, stale, step = -1.0, 0, 0
    output.mkdir(parents=True, exist_ok=True)
    for epoch in range(max_epochs):
        model.train()
        for x, y, _, _, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(x)
            loss = criterion(logits, y)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite B0 loss.")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["grad_clip_norm"]))
            optimizer.step()
            step += 1
            if max_steps and step >= max_steps:
                break
        score = _b0_macro_f1(model, val_loader, device)
        if score > best:
            best, stale = score, 0
            _checkpoint(output / "best.pt", model, config, "b0", epoch, score)
        else:
            stale += 1
        if max_steps or stale >= int(train_cfg["early_stopping_patience"]):
            break
    (output / "run_metadata.json").write_text(json.dumps({"model_type": "b0", "seed": config["seed"], "class_order_version": CLASS_ORDER_VERSION, "best_validation_macro_f1": best}, indent=2))
    return output / "best.pt"


@torch.no_grad()
def _b0_macro_f1(model: B0Model, loader: DataLoader, device: torch.device) -> float:
    from sklearn.metrics import f1_score
    model.eval(); truth, pred = [], []
    for x, y, _, _, valid in loader:
        logits, _ = model(x.to(device)); mask = torch.as_tensor(valid, dtype=torch.bool)
        truth.extend(y[mask].tolist()); pred.extend(logits.cpu().argmax(1)[mask].tolist())
    model.train()
    return float(f1_score(truth, pred, labels=np.arange(4), average="macro", zero_division=0))


def train_b1(config: dict, output: Path, smoke: bool = False) -> Path:
    seed_everything(int(config["seed"]))
    data, train_cfg = config["data"], config["train"]
    train_set = EmbeddingWindowDataset(data["manifest"], data["embedding_root"], "train", subject_limit=8 if smoke else None)
    val_set = EmbeddingWindowDataset(data["manifest"], data["embedding_root"], "val", subject_limit=2 if smoke else None)
    train_loader = DataLoader(train_set, batch_size=int(train_cfg["batch_size"]), shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=int(train_cfg["batch_size"]), shuffle=False, num_workers=0)
    device = device_for_run()
    model = B1Model(int(config["model"]["hidden_dim"]), int(config["model"]["layers"]), float(config["model"]["dropout"])).to(device)
    labels = []
    for _, y, _, valid, _, _ in train_set:
        labels.extend(y[valid].tolist())
    criterion = nn.CrossEntropyLoss(weight=class_weights(np.asarray(labels)).to(device), ignore_index=-100)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["learning_rate"]), weight_decay=float(train_cfg["weight_decay"]))
    output.mkdir(parents=True, exist_ok=True)
    best, stale, step = -1.0, 0, 0
    for epoch in range(1 if smoke else int(train_cfg["epochs"])):
        model.train()
        for x, y, valid_in, valid_out, _, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(x.to(device), valid_in.to(device))
            targets = y.to(device)
            targets[~valid_out.to(device)] = -100
            loss = criterion(logits.reshape(-1, 4), targets.reshape(-1))
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite B1 loss.")
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["grad_clip_norm"])); optimizer.step()
            step += 1
            if smoke and step >= 2:
                break
        score = _b1_macro_f1(model, val_loader, device)
        if score > best:
            best, stale = score, 0
            _checkpoint(output / "best.pt", model, config, "b1", epoch, score)
        else:
            stale += 1
        if smoke or stale >= int(train_cfg["early_stopping_patience"]):
            break
    (output / "run_metadata.json").write_text(json.dumps({"model_type": "b1", "seed": config["seed"], "class_order_version": CLASS_ORDER_VERSION, "best_validation_macro_f1": best}, indent=2))
    return output / "best.pt"


@torch.no_grad()
def _b1_macro_f1(model: B1Model, loader: DataLoader, device: torch.device) -> float:
    from sklearn.metrics import f1_score
    model.eval(); truth, pred = [], []
    for x, y, valid_in, valid_out, _, _ in loader:
        logits = model(x.to(device), valid_in.to(device)).cpu()
        truth.extend(y[valid_out].tolist()); pred.extend(logits.argmax(-1)[valid_out].tolist())
    model.train()
    return float(f1_score(truth, pred, labels=np.arange(4), average="macro", zero_division=0))
