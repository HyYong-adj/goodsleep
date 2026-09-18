"""Phase 3: 동결 2단계를 버리고 Conformer + BiLSTM 을 공동 학습한다.

구조는 parent 와 같고 **학습 방식만** 바꾼다. 비교 대상은 같은 구성요소를 동결 2단계로
학습한 Phase 2 의 C0 대조군(없으면 parent B1 0.483261)이다.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from psg_only.constants import CANONICAL_CLASSES, CLASS_ORDER_VERSION  # noqa: E402
from psg_only.data import MelWindowDataset  # noqa: E402
from run_phase4 import gated_kd_loss  # noqa: E402
from psg_only.endtoend import endtoend_from_config  # noqa: E402
from psg_only.metrics import evaluate_predictions  # noqa: E402
from psg_only.provenance import file_hash  # noqa: E402
from psg_only.train import (build_scheduler, class_weights, config_hash, device_for_run,
                            seed_everything, spec_augment)  # noqa: E402
from psg_only.transformer import configure_precision  # noqa: E402
from psg_only.windows import context_options, stitch_predictions  # noqa: E402

PARENT = ROOT / "artifacts/experiments/conformer_epoch_20260915T054210274368058"
PARENT_B1_MACRO_F1 = 0.48326135281572924
TEACHER = ROOT / "artifacts/teacher_canonical"


class KDMelWindowDataset(MelWindowDataset):
    """Mel 윈도우에 canonical/epoch 축 teacher logits 를 함께 돌려준다."""

    def __init__(self, *args, teacher_root, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_root = Path(teacher_root)
        self._teacher = {}

    def _teacher_arrays(self, subject_id):
        if subject_id not in self._teacher:
            base = self.teacher_root / subject_id
            logits = np.load(base / "logits.npy")
            valid = np.load(base / "valid.npy")
            span = len(self.index[subject_id][0])
            if len(logits) < span:
                raise ValueError(f"{subject_id}: teacher axis {len(logits)} shorter than epoch axis {span}")
            self._teacher[subject_id] = (logits, valid)
        return self._teacher[subject_id]

    def __getitem__(self, position):
        x, y, input_valid, target_valid, subject_id, target_indexes = super().__getitem__(position)
        logits, t_valid = self._teacher_arrays(subject_id)
        teacher = np.zeros((self.target_epochs, 4), np.float32)
        teacher_ok = np.zeros(self.target_epochs, bool)
        for slot, epoch in enumerate(target_indexes.tolist()):
            if epoch >= 0 and epoch < len(t_valid) and t_valid[epoch]:
                teacher[slot] = logits[epoch]
                teacher_ok[slot] = True
        return (x, y, input_valid, target_valid, subject_id, target_indexes,
                torch.from_numpy(teacher), torch.from_numpy(teacher_ok))


def default_config(seed: int) -> dict:
    c1 = yaml.safe_load((PARENT / "c1.yaml").read_text())
    b1 = yaml.safe_load((PARENT / "b1.yaml").read_text())
    return {
        "seed": int(seed),
        "data": {
            "manifest": c1["data"]["manifest"],
            "cache_root": c1["data"]["cache_root"],
            "stats": c1["data"]["stats"],
            "night_norm": False,
            "input_epochs": b1["data"]["input_epochs"],
            "target_epochs": b1["data"]["target_epochs"],
            "stride": b1["data"]["stride"],
        },
        "model": {
            "type": "endtoend_conformer_bilstm",
            "embedding_dim": 192,
            "layers": c1["model"]["layers"],
            "num_heads": c1["model"]["num_heads"],
            "ff_dim": c1["model"]["ff_dim"],
            "conv_kernel_size": c1["model"]["conv_kernel_size"],
            "dropout": c1["model"]["dropout"],
            "hidden_dim": b1["model"]["hidden_dim"],
            "lstm_layers": b1["model"]["layers"],
        },
        "train": {
            "epochs": 30,
            "batch_size": 4,
            "learning_rate": 0.0003,
            "weight_decay": 0.0001,
            "early_stopping_patience": 12,
            "grad_clip_norm": 1.0,
            "amp": True,
            # Phase 3(1차)는 16 epoch 조기종료 + 스케줄 없음으로 미학습이었고 best 가
            # 부풀림 +0.0868 인 단일 epoch 스파이크였다. Phase 2 에서 유일하게 확인된
            # SpecAugment 와 Phase 1 에서 유일하게 부호가 긍정적이던 cosine 을 붙인다.
            "scheduler": "cosine",
            "warmup_epochs": 3,
            "spec_augment": {"freq_mask": 8, "time_mask": 150, "gain_std": 0.25},
            "kd": None,
        },
    }


@torch.no_grad()
def _score(model, loader, device, criterion):
    from sklearn.metrics import f1_score
    model.eval()
    truth, pred, losses, weights = [], [], 0.0, 0.0
    for batch in loader:
        x, y, valid_in, valid_out = batch[0], batch[1], batch[2], batch[3]
        logits = model(x.to(device), valid_in.to(device)).float().cpu()
        targets = y.masked_fill(~valid_out, -100)
        if (targets != -100).any():
            loss = nn.functional.cross_entropy(logits.reshape(-1, 4), targets.reshape(-1),
                                               weight=criterion.weight.cpu(), ignore_index=-100)
            weight = float(criterion.weight.cpu()[targets[targets != -100]].sum())
            losses += float(loss) * weight
            weights += weight
        truth.extend(y[valid_out].tolist())
        pred.extend(logits.argmax(-1)[valid_out].tolist())
    model.train()
    return (float(f1_score(truth, pred, labels=np.arange(4), average="macro", zero_division=0)),
            losses / weights if weights else None)


@torch.no_grad()
def _predict(model, loader, device):
    model.eval()
    rows = []
    for batch in loader:
        x, y, valid_in, valid_out, subjects, indexes = batch[:6]
        probabilities = torch.softmax(model(x.to(device), valid_in.to(device)).float(), dim=-1).cpu()
        predictions = probabilities.argmax(-1)
        for item in range(x.shape[0]):
            for position in range(probabilities.shape[1]):
                if not bool(valid_out[item, position]):
                    continue
                probability = probabilities[item, position]
                rows.append({
                    "subject_id": str(subjects[item]), "epoch_index": int(indexes[item, position]),
                    "target": int(y[item, position]), "prediction": int(predictions[item, position]),
                    "prob_wake": float(probability[0]), "prob_rem": float(probability[1]),
                    "prob_light": float(probability[2]), "prob_deep": float(probability[3]),
                    "valid": True, "model_version": "endtoend_conformer_bilstm",
                    "class_order_version": CLASS_ORDER_VERSION,
                })
    return rows


def run(config: dict, output: Path, smoke: bool = False) -> dict:
    configure_precision()
    seed_everything(int(config["seed"]))
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
    data, train_cfg = config["data"], config["train"]
    window = context_options(data)
    limits = dict(subject_limit=6) if smoke else {}
    common = dict(manifest_path=data["manifest"], cache_root=data["cache_root"], stats_path=data["stats"],
                  night_norm=bool(data.get("night_norm", False)), **window)
    kd = train_cfg.get("kd")
    maker = (lambda **kw: KDMelWindowDataset(teacher_root=TEACHER, **common, **kw)) if kd else \
            (lambda **kw: MelWindowDataset(**common, **kw))
    train_set = maker(split="train", **limits)
    val_set = maker(split="val", **(dict(subject_limit=3) if smoke else {}))
    workers = 2 if not smoke else 0
    train_loader = DataLoader(train_set, batch_size=int(train_cfg["batch_size"]), shuffle=True, num_workers=workers)
    val_loader = DataLoader(val_set, batch_size=int(train_cfg["batch_size"]), shuffle=False, num_workers=workers)

    device = device_for_run()
    # CUDA 초기화 실패 시 device_for_run 은 조용히 CPU 를 돌려준다. 무인 장시간 실행이
    # CPU 로 떨어져 헛도는 것을 막기 위해 정규 실행에서는 즉시 실패시킨다(2026-09-18 실제 발생).
    if not smoke and device.type != "cuda":
        raise RuntimeError(
            "CUDA unavailable - refusing to start a full end-to-end run on CPU. "
            "노드의 CUDA 상태를 확인하라(GPU 2 하드웨어 오류가 전체를 오염시킨 전례 있음)."
        )
    model = endtoend_from_config(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["learning_rate"]),
                                  weight_decay=float(train_cfg["weight_decay"]))
    max_epochs = 1 if smoke else int(train_cfg["epochs"])
    # smoke 는 1 epoch 이라 warmup 이 예산보다 길어진다. 스케줄은 정규 실행에서만 건다.
    scheduler = None if smoke else build_scheduler(optimizer, train_cfg, max_epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and train_cfg.get("amp", False))
    criterion = nn.CrossEntropyLoss(weight=class_weights(train_set.labels_flat()).to(device), ignore_index=-100,
                                    label_smoothing=float(train_cfg.get("label_smoothing", 0.0)))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    augment = train_cfg.get("spec_augment") or None
    started, best, stale, step = time.perf_counter(), -1.0, 0, 0
    for epoch in range(max_epochs):
        model.train()
        for batch in train_loader:
            x, y, valid_in, valid_out = batch[0], batch[1], batch[2], batch[3]
            if not valid_out.any():
                continue
            optimizer.zero_grad(set_to_none=True)
            targets = y.to(device)
            targets[~valid_out.to(device)] = -100
            mel = x.to(device)
            if augment is not None:
                # 지역 이름이 dataloader batch 튜플을 가리지 않게 한다(이전 버그).
                n_items, n_epochs = mel.shape[:2]
                mel = spec_augment(mel.reshape(n_items * n_epochs, 1, 48, 1499), **augment).reshape(mel.shape)
            with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                logits = model(mel, valid_in.to(device))
                loss = criterion(logits.reshape(-1, 4), targets.reshape(-1))
                if kd:
                    kd_targets = targets.clone()
                    kd_targets[~batch[7].to(device)] = -100
                    kd_loss = gated_kd_loss(logits.reshape(-1, 4).float(),
                                            batch[6].to(device).reshape(-1, 4).float(),
                                            kd_targets.reshape(-1), criterion.weight,
                                            float(kd["temperature"]))
                    loss = (1.0 - float(kd["alpha"])) * loss + float(kd["alpha"]) * kd_loss
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite end-to-end loss.")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["grad_clip_norm"]))
            scaler.step(optimizer)
            scaler.update()
            step += 1
            if smoke and step >= 2:
                break
        score, validation_loss = _score(model, val_loader, device, criterion)
        record = {"epoch": epoch, "optimizer_steps": step, "validation_macro_f1": score,
                  "validation_loss": validation_loss, "learning_rate": optimizer.param_groups[0]["lr"],
                  "elapsed_seconds": time.perf_counter() - started}
        with (output / "history.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        if scheduler is not None:
            scheduler.step()
        if score > best:
            best, stale = score, 0
            torch.save({"state_dict": model.state_dict(), "config": config, "config_hash": config_hash(config),
                        "model_type": "endtoend_conformer_bilstm", "class_order": list(CANONICAL_CLASSES),
                        "class_order_version": CLASS_ORDER_VERSION, "epoch": epoch,
                        "validation_macro_f1": score}, output / "best.pt")
        else:
            stale += 1
        if smoke or stale >= int(train_cfg["early_stopping_patience"]):
            break

    model.load_state_dict(torch.load(output / "best.pt", map_location=device, weights_only=False)["state_dict"])
    rows = stitch_predictions(_predict(model, val_loader, device), val_set.expected_keys())
    import pandas as pd
    (output / "evaluation").mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "evaluation/predictions.csv", index=False)
    metrics = evaluate_predictions(rows)
    status = metrics.pop("status")
    payload = {"status": status, "seed": config["seed"], "model_type": "endtoend_conformer_bilstm",
               "smoke": smoke, "epochs": len(rows), "subjects": len({r["subject_id"] for r in rows}),
               "metrics": metrics, "best_validation_macro_f1": best,
               "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
               "epochs_completed": epoch + 1, "optimizer_steps": step,
               "training_seconds": time.perf_counter() - started,
               "peak_cuda_memory_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
               "manifest_hash": file_hash(data["manifest"]), "evaluation_scope": "validation-only",
               "parent_b1_macro_f1": PARENT_B1_MACRO_F1,
               "delta_vs_parent_b1": metrics["macro_f1"] - PARENT_B1_MACRO_F1}
    (output / "evaluation/results.json").write_text(json.dumps(payload, indent=2))
    if not smoke and status != "complete":
        raise ValueError("End-to-end evaluation missing classes.")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--night-norm", action="store_true")
    parser.add_argument("--kd", action="store_true", help="PSG teacher gated KD (a0.3, T2)")
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()
    config = default_config(args.seed)
    if args.night_norm:
        config["data"]["night_norm"] = True
    if args.kd:
        config["train"]["kd"] = {"alpha": 0.3, "temperature": 2.0}
    if args.epochs:
        config["train"]["epochs"] = args.epochs
    result = run(config, args.output.resolve(), smoke=args.smoke)
    print(json.dumps({"stage": "complete", "macro_f1": result["metrics"]["macro_f1"],
                      "delta_vs_parent_b1": result["delta_vs_parent_b1"]}), flush=True)


if __name__ == "__main__":
    main()
