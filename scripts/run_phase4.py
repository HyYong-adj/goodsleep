"""Phase 4: C3(SpecAugment) embedding 위에서 아직 안 쓴 레버를 얹는다.

C3 가 Phase 2 의 유일한 생존 후보다(부풀림 보정 후 +0.0247). 그 embedding 캐시를
재사용하므로 런당 약 2분이다. 새 레버는 둘:
  - cosine LR + warmup : Phase 1 에서 유일하게 부호가 긍정적이었다(A2)
  - gated KD           : PSG teacher(macro F1 0.672) — 본 트랙이 한 번도 쓰지 않은 신호

게이트 없는 기본 KD 는 쓰지 않는다. 참조 트랙에서 REM -0.10 으로 폐기됐고,
원인은 teacher 의 REM 약점(F1 0.58, precision 0.51)과 KD 항의 Light prior 다.
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

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from psg_only.constants import CANONICAL_CLASSES, CLASS_ORDER_VERSION  # noqa: E402
from psg_only.data import EmbeddingWindowDataset  # noqa: E402
from psg_only.evaluate import evaluate_b1, write_evaluation  # noqa: E402
from psg_only.train import (build_scheduler, class_weights, config_hash, device_for_run,  # noqa: E402
                            load_config, seed_everything)
from psg_only.windows import context_options  # noqa: E402

PARENT = ROOT / "artifacts/experiments/phase2_20260916/C3_s20260910"
TEACHER = ROOT / "artifacts/teacher_canonical"
PARENT_MACRO_F1 = 0.5052

VARIANTS = {
    "D0": {"label": "C3 기준 (변경 없음)", "train": {}},
    "D1": {"label": "+ cosine LR + warmup 3", "train": {"scheduler": "cosine", "warmup_epochs": 3}},
    "D2": {"label": "+ gated KD (a0.3 T2)", "train": {"kd": {"alpha": 0.3, "temperature": 2.0}}},
    "D3": {"label": "+ cosine + gated KD", "train": {"scheduler": "cosine", "warmup_epochs": 3,
                                                      "kd": {"alpha": 0.3, "temperature": 2.0}}},
}


class KDWindowDataset(EmbeddingWindowDataset):
    """기존 윈도우에 teacher logits(canonical 순서, epoch 축)를 함께 돌려준다."""

    def __init__(self, *args, teacher_root, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_root = Path(teacher_root)
        self._teacher: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def _teacher_arrays(self, subject_id):
        if subject_id not in self._teacher:
            base = self.teacher_root / subject_id
            logits = np.load(base / "logits.npy")
            valid = np.load(base / "valid.npy")
            embeddings = self._load(subject_id)[0]
            if len(logits) != len(embeddings):
                raise ValueError(f"{subject_id}: teacher axis {len(logits)} != embedding axis {len(embeddings)}")
            self._teacher[subject_id] = (logits, valid)
        return self._teacher[subject_id]

    def __getitem__(self, index):
        x, y, input_valid, target_valid, subject_id, target_indexes = super().__getitem__(index)
        logits, t_valid = self._teacher_arrays(subject_id)
        teacher = np.zeros((self.target_epochs, 4), np.float32)
        teacher_ok = np.zeros(self.target_epochs, bool)
        for position, epoch in enumerate(target_indexes.tolist()):
            if epoch >= 0 and t_valid[epoch]:
                teacher[position] = logits[epoch]
                teacher_ok[position] = True
        return (x, y, input_valid, target_valid, subject_id, target_indexes,
                torch.from_numpy(teacher), torch.from_numpy(teacher_ok))


def gated_kd_loss(student, teacher, targets, weight, temperature):
    """teacher 신뢰도로 게이트한 KD. 유효 target 이 없으면 0 을 돌려준다.

    게이트 = teacher 최대 확률. teacher 가 헷갈리는 epoch 은 적게 배운다.
    클래스 가중치를 함께 곱해 다수 클래스(Light) prior 가 KD 항을 지배하지 않게 한다.
    """
    mask = targets != -100
    if not mask.any():
        return student.sum() * 0.0
    student, teacher, targets = student[mask], teacher[mask], targets[mask]
    with torch.no_grad():
        probabilities = torch.softmax(teacher, dim=-1)
        gate = probabilities.max(dim=-1).values
        soft = torch.softmax(teacher / temperature, dim=-1)
        scale = gate * weight[targets]
    log_student = torch.log_softmax(student / temperature, dim=-1)
    per_epoch = torch.nn.functional.kl_div(log_student, soft, reduction="none").sum(-1) * (temperature ** 2)
    denominator = scale.sum()
    return (per_epoch * scale).sum() / denominator if denominator > 0 else student.sum() * 0.0


def train_one(config, output, teacher_root):
    from psg_only.models import B1Model
    seed_everything(int(config["seed"]))
    data, train_cfg = config["data"], config["train"]
    kd = train_cfg.get("kd")
    window = context_options(data)
    maker = (lambda split: KDWindowDataset(data["manifest"], data["embedding_root"], split,
                                           teacher_root=teacher_root, **window)) if kd else \
            (lambda split: EmbeddingWindowDataset(data["manifest"], data["embedding_root"], split, **window))
    train_set, val_set = maker("train"), maker("val")
    if train_set.provenance != val_set.provenance:
        raise ValueError("Train/val embedding provenance differs.")
    config["embedding_provenance"] = train_set.provenance
    train_loader = DataLoader(train_set, batch_size=int(train_cfg["batch_size"]), shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=int(train_cfg["batch_size"]), shuffle=False, num_workers=0)

    device = device_for_run()
    model = B1Model(int(config["model"]["hidden_dim"]), int(config["model"]["layers"]),
                    float(config["model"]["dropout"]), input_epochs=window["input_epochs"]).to(device)
    labels = []
    for item in train_set:
        labels.extend(item[1][item[3]].tolist())
    weight = class_weights(np.asarray(labels), train_cfg.get("class_weight_multipliers")).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=-100,
                                    label_smoothing=float(train_cfg.get("label_smoothing", 0.0)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["learning_rate"]),
                                  weight_decay=float(train_cfg["weight_decay"]))
    max_epochs = int(train_cfg["epochs"])
    scheduler = build_scheduler(optimizer, train_cfg, max_epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and train_cfg.get("amp", False))
    output.mkdir(parents=True, exist_ok=True)
    best, stale, step = -1.0, 0, 0
    from sklearn.metrics import f1_score
    for epoch in range(max_epochs):
        model.train()
        for batch in train_loader:
            x, y, valid_in, valid_out = batch[0], batch[1], batch[2], batch[3]
            if not valid_out.any():
                continue
            optimizer.zero_grad(set_to_none=True)
            targets = y.to(device)
            targets[~valid_out.to(device)] = -100
            with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                logits = model(x.to(device), valid_in.to(device))
                loss = criterion(logits.reshape(-1, 4), targets.reshape(-1))
                if kd:
                    teacher = batch[6].to(device)
                    teacher_ok = batch[7].to(device)
                    kd_targets = targets.clone()
                    kd_targets[~teacher_ok] = -100
                    kd_loss = gated_kd_loss(logits.reshape(-1, 4).float(), teacher.reshape(-1, 4).float(),
                                            kd_targets.reshape(-1), weight, float(kd["temperature"]))
                    loss = (1.0 - float(kd["alpha"])) * loss + float(kd["alpha"]) * kd_loss
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss.")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["grad_clip_norm"]))
            scaler.step(optimizer)
            scaler.update()
            step += 1
        model.eval()
        truth, pred = [], []
        with torch.no_grad():
            for batch in val_loader:
                x, y, valid_in, valid_out = batch[0], batch[1], batch[2], batch[3]
                out = model(x.to(device), valid_in.to(device)).cpu()
                truth.extend(y[valid_out].tolist())
                pred.extend(out.argmax(-1)[valid_out].tolist())
        score = float(f1_score(truth, pred, labels=np.arange(4), average="macro", zero_division=0))
        with (output / "history.jsonl").open("a") as stream:
            stream.write(json.dumps({"epoch": epoch, "optimizer_steps": step, "validation_macro_f1": score,
                                     "learning_rate": optimizer.param_groups[0]["lr"]}) + "\n")
        if scheduler is not None:
            scheduler.step()
        if score > best:
            best, stale = score, 0
            torch.save({"state_dict": model.state_dict(), "config": config, "config_hash": config_hash(config),
                        "model_type": "b1", "class_order": list(CANONICAL_CLASSES),
                        "class_order_version": CLASS_ORDER_VERSION, "epoch": epoch,
                        "validation_macro_f1": score}, output / "best.pt")
        else:
            stale += 1
        if stale >= int(train_cfg["early_stopping_patience"]):
            break
    return output / "best.pt"


def inflation(path: Path):
    scores = [json.loads(l)["validation_macro_f1"] for l in path.read_text().splitlines() if l.strip()]
    best = max(range(len(scores)), key=lambda i: scores[i])
    neighbours = [scores[i] for i in (best - 1, best + 1) if 0 <= i < len(scores)]
    mean = statistics.fmean(neighbours) if neighbours else scores[best]
    return {"best": scores[best], "best_epoch": best, "epochs": len(scores),
            "neighbour_mean": mean, "inflation": scores[best] - mean}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", default="D0,D1,D2,D3")
    parser.add_argument("--seeds", default="20260910,20260911,20260912")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    base = load_config(PARENT / "b1.yaml")
    base["data"]["embedding_root"] = str(PARENT / "embeddings")
    state = {"phase": "phase4", "status": "RUNNING", "started_at": datetime.now(timezone.utc).isoformat(),
             "device": str(device_for_run()), "parent_run": str(PARENT), "parent_macro_f1": PARENT_MACRO_F1,
             "teacher_cache": str(TEACHER), "evaluation_scope": "validation-only",
             "variants": {n: VARIANTS[n]["label"] for n in variants}, "seeds": seeds, "records": []}

    def save():
        (out / "phase4.json").write_text(json.dumps(state, indent=2, ensure_ascii=False))

    save()
    try:
        for variant in variants:
            for seed in seeds:
                tag = f"{variant}_s{seed}"
                config = copy.deepcopy(base)
                config["seed"] = seed
                config["train"].update(copy.deepcopy(VARIANTS[variant]["train"]))
                run_dir = out / tag
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "b1.yaml").write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
                started = time.perf_counter()
                checkpoint_path = train_one(config, run_dir / "b1", TEACHER)
                rows, checkpoint = evaluate_b1(checkpoint_path)
                result = write_evaluation(rows, checkpoint, run_dir / "b1/evaluation")
                if result["status"] != "complete":
                    raise ValueError(f"{tag}: evaluation missing classes.")
                metrics = result["metrics"]
                record = {"variant": variant, "variant_label": VARIANTS[variant]["label"], "seed": seed,
                          "macro_f1": metrics["macro_f1"], "accuracy": metrics["accuracy"],
                          "cohen_kappa": metrics["cohen_kappa"],
                          "per_class_f1": {k: v["f1"] for k, v in metrics["per_class"].items()},
                          "per_class_recall": {k: v["recall"] for k, v in metrics["per_class"].items()},
                          "selection_inflation": inflation(run_dir / "b1/history.jsonl"),
                          "wall_seconds": time.perf_counter() - started}
                (run_dir / "record.json").write_text(json.dumps(record, indent=2, ensure_ascii=False))
                state["records"].append(record)
                save()
                print(json.dumps({"stage": "done", "run": tag, "macro_f1": record["macro_f1"]}), flush=True)
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
