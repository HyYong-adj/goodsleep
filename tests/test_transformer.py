import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from psg_only.constants import CANONICAL_CLASSES, CLASS_ORDER_VERSION
from psg_only.evaluate import load_model
from psg_only.provenance import file_hash
from psg_only.transformer import PositionedEmbeddingWindowDataset, evaluate_transformer, train_transformer
from psg_only.transformer_model import TemporalTransformer
from psg_only.train import seed_everything


def fixture_config(tmp_path):
    rows = []
    root = tmp_path / "embeddings"
    rng = np.random.default_rng(42)
    for split, sid in (("train", "train_a"), ("val", "val_b")):
        directory = root / sid
        directory.mkdir(parents=True)
        valid = np.ones(65, dtype=bool)
        valid[:20] = False
        valid[32] = False
        labels = np.asarray((0, 2, 3, 1))[np.arange(65) % 4]
        labels[~valid] = -100
        np.save(directory / "embeddings.npy", rng.normal(size=(65, 192)).astype(np.float16))
        np.save(directory / "labels.npy", labels)
        np.save(directory / "valid.npy", valid)
        rows.extend(dict(subject_id=sid, split=split, subject_epoch_index=i,
                         recording_start_seconds=i * 30, label_index=i % 4)
                    for i in range(65) if i != 32)
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    for directory in root.iterdir():
        metadata = dict(
            checkpoint_sha256="fixture", config_hash="fixture", stats_hash="fixture", source_cache="fixture",
            class_order=list(CANONICAL_CLASSES), class_order_version=CLASS_ORDER_VERSION,
            manifest_hash=file_hash(manifest), subject_id=directory.name,
            files={name: file_hash(directory / name) for name in ("embeddings.npy", "labels.npy", "valid.npy")},
        )
        (directory / "metadata.json").write_text(json.dumps(metadata))
    cfg = yaml.safe_load(Path("configs/t40_transformer.yaml").read_text())
    cfg["data"].update(manifest=str(manifest), embedding_root=str(root))
    cfg["model"].update(layers=1, ff_dim=32)
    cfg["train"].update(batch_size=2, epochs=1)
    return cfg


def test_positions_retain_real_timeline_and_target_grid(tmp_path):
    cfg = fixture_config(tmp_path)
    ds = PositionedEmbeddingWindowDataset(cfg["data"]["manifest"], cfg["data"]["embedding_root"], "train")
    _, y, iv, tv, _, indexes, positions = ds[1]
    assert positions[0] == -1  # unscored source position 10
    assert positions[10] == 20
    assert positions[22] == -1  # real gap at 32, not compressed
    assert positions[23] == 33
    assert not tv[12] and y[12] == -100
    assert indexes[13] == 33
    all_keys = [(ds[i][4], int(t)) for i in range(len(ds)) for t in ds[i][5][ds[i][3]]]
    assert len(all_keys) == len(set(all_keys)) == len(ds.expected_keys()) == 44
    assert len(ds.canonical_labels()) == 44  # not overlapping context-token counts
    last = ds[len(ds) - 1]
    assert last[5].tolist() == [60, 61, 62, 63, 64] + [-1] * 15


@pytest.mark.parametrize("length", [40, 80])
def test_shape_central_slice_and_masked_backward(length):
    model = TemporalTransformer(layers=1, ff_dim=32, dropout=0., input_epochs=length)
    x = torch.randn(2, length, 192, requires_grad=True)
    valid = torch.ones(2, length, dtype=torch.bool)
    valid[0] = False
    valid[1, 15] = False
    positions = torch.arange(length).repeat(2, 1)
    logits = model(x, valid, positions)
    assert logits.shape == (2, 20, 4)
    assert torch.isfinite(logits).all() and (logits[0] == 0).all()
    left = (length - 20) // 2
    torch.testing.assert_close(logits, model.forward_all(x, valid, positions)[:, left:left + 20])
    logits[1].square().sum().backward()
    assert torch.isfinite(x.grad).all()
    assert (x.grad[~valid] == 0).all()


def test_padding_values_length_and_batch_do_not_affect_valid_queries():
    model = TemporalTransformer(layers=2, ff_dim=32, dropout=0.).eval()
    x = torch.randn(1, 7, 192)
    valid = torch.ones(1, 7, dtype=torch.bool)
    positions = torch.arange(100, 107).unsqueeze(0)
    with torch.no_grad():
        expected = model.forward_all(x, valid, positions)
        padded = torch.full((2, 13, 192), float("nan"))
        padded[0, :7] = x
        mask = torch.zeros(2, 13, dtype=torch.bool)
        mask[0, :7] = True
        indices = torch.full((2, 13), -1, dtype=torch.long)
        indices[0, :7] = positions
        result = model.forward_all(padded, mask, indices)
    torch.testing.assert_close(result[0, :7], expected[0], atol=2e-6, rtol=2e-6)
    assert torch.isfinite(result).all()
    assert (result[1] == 0).all()


def test_reject_bad_mask_positions_and_context():
    model = TemporalTransformer(layers=1, ff_dim=32)
    x = torch.zeros(1, 40, 192)
    valid = torch.ones(1, 40, dtype=torch.bool)
    positions = torch.arange(40).unsqueeze(0)
    with pytest.raises(ValueError, match="boolean"):
        model(x, valid.float(), positions)
    with pytest.raises(ValueError, match="nonnegative"):
        model(x, valid, positions - 1)
    with pytest.raises(ValueError, match="context"):
        model(x[:, :20], valid[:, :20], positions[:, :20])


def test_training_seed_reload_coverage_and_provenance(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    cfg = fixture_config(tmp_path)
    first = train_transformer(cfg, tmp_path / "run_a", smoke=True)
    second = train_transformer(cfg, tmp_path / "run_b", smoke=True)
    a = torch.load(first, weights_only=False)
    b = torch.load(second, weights_only=False)
    assert all(torch.equal(a["state_dict"][key], b["state_dict"][key]) for key in a["state_dict"])
    assert a["config"]["train_class_counts"] == [11, 11, 11, 11]
    model, _ = load_model(first, torch.device("cpu"))
    assert isinstance(model, TemporalTransformer)
    rows, payload = evaluate_transformer(first)
    assert len(rows) == 44
    assert all(r["model_version"] == "temporal_transformer" for r in rows)
    assert payload["evaluation_timing"]["subjects"] == 1
    with pytest.raises(ValueError, match="validation-only"):
        evaluate_transformer(first, "test")
    (Path(cfg["data"]["embedding_root"]) / "val_b/metadata.json").write_text("{}")
    with pytest.raises(ValueError, match="class order"):
        evaluate_transformer(first)


def test_context_pair_initialization_matches():
    seed_everything(20260910)
    a = TemporalTransformer(input_epochs=40)
    seed_everything(20260910)
    b = TemporalTransformer(input_epochs=80)
    assert all(torch.equal(a.state_dict()[k], b.state_dict()[k]) for k in a.state_dict())
    assert not torch.equal(a.layers[0].linear1.weight, a.layers[1].linear1.weight)


def test_pair_rejects_ignored_options_and_unmatched_training(tmp_path):
    spec = importlib.util.spec_from_file_location("run_transformer_pair", "scripts/run_transformer_pair.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    paths = []
    for length in (40, 80):
        cfg = yaml.safe_load(Path(f"configs/t{length}_transformer.yaml").read_text())
        path = tmp_path / f"t{length}.yaml"
        path.write_text(yaml.safe_dump(cfg))
        paths.append(str(path))
    assert len(runner.read_configs(paths)) == 2
    cfg["model"]["causal"] = True
    Path(paths[1]).write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="Unsupported"):
        runner.read_configs(paths)
    cfg["model"]["causal"] = False
    cfg["train"]["learning_rate"] = 0.0001
    Path(paths[1]).write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="share model, train and seed"):
        runner.read_configs(paths)
