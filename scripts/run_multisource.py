"""PSG-Audio + SNUBH 다중 데이터셋 end-to-end 학습 (choihy-e027 / e028).

모델·학습 설정은 e014/e018 parent 와 같고(`run_phase3.default_config`), 바뀌는 변수는 **학습 데이터 구성** 하나다.

  --train psg,snubh   C1 동시 학습 (e027)
  --train snubh       B1 SNUBH 단독 (e028)

정규화는 데이터셋별 전역 통계: PSG-Audio 는 e018 과 같은 train_stats.json, SNUBH 는 fold train 녹화로 계산한
`artifacts/manifests/snubh_cv5/fold{k}_train_stats.json`. 각 데이터셋 입력은 자기 통계로만 정규화된다.

매 epoch 두 데이터셋의 fold val 을 모두 채점해 history 에 남기고, 선택 기준별로 checkpoint 를 따로 저장한다.
  - 사전 등록 주 기준: 학습에 쓴 데이터셋 val macro-F1 의 평균 (C1 = psg/snubh 평균, B1 = snubh)
  - 민감도: psg 단독, snubh 단독 기준
학습에 쓰지 않은 데이터셋의 val 로는 선택하지 않는다(B1 의 PSG-Audio val 은 순수 외부 평가).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from psg_only.data import MelWindowDataset  # noqa: E402
from psg_only.endtoend import endtoend_from_config  # noqa: E402
from psg_only.metrics import evaluate_predictions  # noqa: E402
from psg_only.provenance import file_hash  # noqa: E402
from psg_only.train import build_scheduler, class_weights, config_hash, device_for_run, seed_everything, spec_augment  # noqa: E402
from psg_only.transformer import configure_precision  # noqa: E402
from psg_only.windows import context_options, stitch_predictions  # noqa: E402
from run_phase3 import _predict, _score, default_config  # noqa: E402

SOURCES = {
    "psg": dict(manifest="/home/sleep/researchers/byoungjun/manifests/cv5/fold{k}.csv",
                cache_root="/home/sleep/researchers/byoungjun/cache/audio_compact_full",
                stats="/home/sleep/researchers/byoungjun/cache/audio_compact_full/train_stats.json"),
    "snubh": dict(manifest=str(ROOT / "artifacts/manifests/snubh_cv5/fold{k}.csv"),
                  cache_root="/home/sleep/data/marg_psg_audio/_derived/snubh_psg/cache/audio_compact_v1",
                  stats=str(ROOT / "artifacts/manifests/snubh_cv5/fold{k}_train_stats.json")),
}


def build_config(seed: int, fold: int, train_sources: list[str], epochs: int | None, snubh_fraction: float) -> dict:
    config = default_config(seed)
    config["data"] = {key: value for key, value in config["data"].items() if key not in {"manifest", "cache_root", "stats"}}
    config["data"]["fold"] = fold
    config["data"]["sources"] = {name: {key: value.format(k=fold) for key, value in spec.items()} for name, spec in SOURCES.items()}
    config["data"]["train_sources"] = list(train_sources)
    config["data"]["snubh_train_fraction"] = float(snubh_fraction)
    config["data"]["selection_primary"] = "mean:" + ",".join(train_sources)
    if epochs:
        config["train"]["epochs"] = int(epochs)
    return config


def subset_subjects(dataset: MelWindowDataset, fraction: float, seed: int) -> None:
    """학습 곡선용: 녹화 단위로 일부만 남긴다(결정적)."""
    if fraction >= 1.0:
        return
    subjects = sorted(dataset.index)
    rng = np.random.default_rng(seed)
    keep = set(rng.permutation(subjects)[: max(1, int(round(len(subjects) * fraction)))].tolist())
    dataset.index = {sid: value for sid, value in dataset.index.items() if sid in keep}
    dataset.windows = [w for w in dataset.windows if w.subject_id in keep]


def run(config: dict, output: Path, smoke: bool = False) -> dict:
    configure_precision()
    seed_everything(int(config["seed"]))
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
    data, train_cfg = config["data"], config["train"]
    window = context_options(data)
    workers = 0 if smoke else 2

    def make(name, split, **kw):
        spec = data["sources"][name]
        return MelWindowDataset(spec["manifest"], spec["cache_root"], spec["stats"], split=split, **window, **kw)

    train_parts = []
    for name in data["train_sources"]:
        part = make(name, "train", **(dict(subject_limit=3) if smoke else {}))
        if name == "snubh":
            subset_subjects(part, float(data["snubh_train_fraction"]), int(config["seed"]))
        train_parts.append(part)
    val_sets = {name: make(name, "val", **(dict(subject_limit=2) if smoke else {})) for name in data["sources"]}
    train_set = ConcatDataset(train_parts)
    labels = np.concatenate([part.labels_flat() for part in train_parts])
    composition = {name: {"subjects": len(part.index), "windows": len(part.windows), "epochs": int(len(part.labels_flat())),
                          "class_counts": np.bincount(part.labels_flat(), minlength=4).tolist()}
                   for name, part in zip(data["train_sources"], train_parts)}
    (output / "composition.json").write_text(json.dumps(composition, indent=2))
    print(json.dumps({"composition": composition}), flush=True)
    train_loader = DataLoader(train_set, batch_size=int(train_cfg["batch_size"]), shuffle=True, num_workers=workers)
    val_loaders = {name: DataLoader(ds, batch_size=int(train_cfg["batch_size"]), shuffle=False, num_workers=workers)
                   for name, ds in val_sets.items()}

    device = device_for_run()
    if not smoke and device.type != "cuda":
        raise RuntimeError("CUDA unavailable - refusing a full run on CPU.")
    if not smoke:
        free, _ = torch.cuda.mem_get_info(device)
        if free < 4 * 2 ** 30:
            raise RuntimeError(f"GPU free memory {free / 2**30:.1f} GiB < 4 GiB — 대기 후 재실행.")
    model = endtoend_from_config(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["learning_rate"]),
                                  weight_decay=float(train_cfg["weight_decay"]))
    max_epochs = 1 if smoke else int(train_cfg["epochs"])
    scheduler = None if smoke else build_scheduler(optimizer, train_cfg, max_epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and train_cfg.get("amp", False))
    criterion = nn.CrossEntropyLoss(weight=class_weights(labels).to(device), ignore_index=-100)
    augment = train_cfg.get("spec_augment") or None

    criteria = {"primary": list(data["train_sources"]), "psg": ["psg"], "snubh": ["snubh"]}
    # 학습에 안 쓴 데이터셋으로는 선택하지 않는다.
    criteria = {key: value for key, value in criteria.items() if set(value) <= set(data["train_sources"])}
    best = {key: -1.0 for key in criteria}
    stale = 0
    started, step = time.perf_counter(), 0
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
                n_items, n_epochs, n_channels = mel.shape[:3]
                mel = spec_augment(mel.reshape(n_items * n_epochs * n_channels, 1, 48, 1499), **augment).reshape(mel.shape)
            with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                loss = criterion(model(mel, valid_in.to(device)).reshape(-1, 4), targets.reshape(-1))
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss.")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["grad_clip_norm"]))
            scaler.step(optimizer)
            scaler.update()
            step += 1
            if smoke and step >= 2:
                break
        scores = {name: _score(model, loader, device, criterion)[0] for name, loader in val_loaders.items()}
        record = {"epoch": epoch, "optimizer_steps": step, **{f"val_macro_f1_{k}": v for k, v in scores.items()},
                  "learning_rate": optimizer.param_groups[0]["lr"], "elapsed_seconds": time.perf_counter() - started}
        with (output / "history.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        if scheduler is not None:
            scheduler.step()
        improved_primary = False
        for key, members in criteria.items():
            value = float(np.mean([scores[m] for m in members]))
            if value > best[key]:
                best[key] = value
                improved_primary |= key == "primary"
                torch.save({"state_dict": model.state_dict(), "config": config, "config_hash": config_hash(config),
                            "model_type": "endtoend_conformer_bilstm", "epoch": epoch, "criterion": key,
                            "selection_score": value, "val_scores": scores}, output / f"best_{key}.pt")
        stale = 0 if improved_primary else stale + 1
        if smoke or stale >= int(train_cfg["early_stopping_patience"]):
            break

    results = {"experiment": config.get("experiment"), "fold": data["fold"], "train_sources": data["train_sources"],
               "snubh_train_fraction": data["snubh_train_fraction"], "composition": composition,
               "epochs_completed": epoch + 1, "optimizer_steps": step, "training_seconds": time.perf_counter() - started,
               "manifest_sha256": {n: file_hash(s["manifest"]) for n, s in data["sources"].items()},
               "evaluation_scope": "cv fold val (best-epoch selection on training-domain val only)", "selections": {}}
    for key in criteria:
        payload = torch.load(output / f"best_{key}.pt", map_location=device, weights_only=False)
        model.load_state_dict(payload["state_dict"])
        block = {"epoch": payload["epoch"], "selection_score": payload["selection_score"], "eval": {}}
        for name, ds in val_sets.items():
            rows = stitch_predictions(_predict(model, val_loaders[name], device), ds.expected_keys())
            (output / "evaluation").mkdir(exist_ok=True)
            pd.DataFrame(rows).to_csv(output / f"evaluation/pred_{key}_{name}.csv", index=False)
            metrics = evaluate_predictions(rows)
            block["eval"][name] = metrics
        results["selections"][key] = block
    (output / "evaluation/results.json").write_text(json.dumps(results, indent=2, default=float))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--train", required=True, help="psg,snubh | snubh | psg")
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--snubh-fraction", type=float, default=1.0)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    sources = [s for s in args.train.split(",") if s]
    if not sources or set(sources) - set(SOURCES):
        raise SystemExit(f"--train must be a subset of {sorted(SOURCES)}")
    config = build_config(args.seed, args.fold, sources, args.epochs, args.snubh_fraction)
    config["experiment"] = args.experiment
    result = run(config, args.output.resolve(), smoke=args.smoke)
    print(json.dumps({"stage": "complete", **{k: {n: round(m["macro_f1"], 4) if m.get("status") == "complete" else m.get("status")
                                                    for n, m in v["eval"].items()} for k, v in result["selections"].items()}}), flush=True)


if __name__ == "__main__":
    main()
