import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from test_data import make_fixture
from psg_only.conformer import ConformerEpochModel, conformer_from_config
from psg_only.data import EmbeddingWindowDataset
from psg_only.evaluate import evaluate_b0, evaluate_b1, load_model
from psg_only.train import train_b0, train_b1


def cache_module():
    spec = importlib.util.spec_from_file_location("conformer_test_cache", "scripts/cache_embeddings.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configuration(tmp_path):
    manifest, root, stats = make_fixture(tmp_path)
    frame = pd.read_csv(manifest)
    frame["recording_start_seconds"] = frame.subject_epoch_index * 30
    frame.loc[frame.subject_epoch_index >= 2, "recording_start_seconds"] += 30
    frame.to_csv(manifest, index=False)
    cfg = yaml.safe_load(Path("configs/c1_conformer_epoch.yaml").read_text())
    cfg["data"].update(manifest=str(manifest), cache_root=str(root), stats=str(stats))
    cfg["model"].update(layers=1, ff_dim=32, conv_kernel_size=5)
    return cfg


def test_geometry_gradients_and_fixed_input_contract():
    model = ConformerEpochModel(layers=1, ff_dim=32, conv_kernel_size=5)
    mel = torch.randn(2, 1, 48, 1499, requires_grad=True)
    assert model.subsampling(mel).shape == (2, 96, 6, 188)
    logits, embedding = model(mel)
    assert logits.shape == (2, 4) and embedding.shape == (2, 192)
    assert torch.isfinite(logits).all()
    torch.nn.functional.cross_entropy(logits, torch.tensor([1, 3])).backward()
    assert torch.isfinite(mel.grad).all()
    assert mel.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="fixed Mel shape"):
        model(mel[:, :, :, :-1])


def test_conformer_config_does_not_ignore_unknown_options():
    cfg = yaml.safe_load(Path("configs/c1_conformer_epoch.yaml").read_text())
    conformer_from_config(cfg)
    cfg["model"]["relative_position"] = True
    with pytest.raises(ValueError, match="Unknown/missing"):
        conformer_from_config(cfg)
    with pytest.raises(ValueError, match="positive and odd"):
        ConformerEpochModel(conv_kernel_size=4)


def test_checkpoint_cache_bilstm_roundtrip_and_gap(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    cfg = configuration(tmp_path)
    cp = train_b0(cfg, tmp_path / "c1", smoke=True)
    model, payload = load_model(cp, torch.device("cpu"))
    assert payload["model_type"] == "conformer_epoch"
    assert isinstance(model, ConformerEpochModel)
    c1_rows, _ = evaluate_b0(cp)
    assert len(c1_rows) == 4
    assert all(row["model_version"] == "conformer_epoch" for row in c1_rows)
    cache = tmp_path / "new_embeddings"
    cache_module().cache_embeddings(cp, cache, batch_size=2)
    valid = np.load(cache / "s1/valid.npy")
    assert valid.tolist() == [True, True, False, True, True]
    assert np.load(cache / "s1/labels.npy").tolist() == [0, 2, -100, 3, 1]
    assert np.load(cache / "s1/embeddings.npy").shape == (5, 192)
    metadata = json.loads((cache / "s1/metadata.json").read_text())
    assert metadata["encoder_type"] == "conformer_epoch"
    before = (cache / "s1/embeddings.npy").read_bytes()
    cache_module().cache_embeddings(cp, cache, batch_size=2)
    assert (cache / "s1/embeddings.npy").read_bytes() == before

    b1 = copy.deepcopy(cfg["b1"])
    b1["seed"] = cfg["seed"]
    b1["model"].update(hidden_dim=4, layers=1)
    b1["data"].update(manifest=cfg["data"]["manifest"], embedding_root=str(cache))
    ds = EmbeddingWindowDataset(cfg["data"]["manifest"], cache, "train")
    assert ds.expected_keys() == {("s1", 0), ("s1", 1), ("s1", 3), ("s1", 4)}
    b1_cp = train_b1(b1, tmp_path / "b1", smoke=True)
    b1_rows, _ = evaluate_b1(b1_cp)
    keys = lambda rows: {(r["subject_id"], r["epoch_index"], r["target"]) for r in rows}
    assert keys(c1_rows) == keys(b1_rows)

    other_cp = tmp_path / "changed_encoder.pt"
    changed = torch.load(cp, weights_only=False)
    changed["epoch"] = 100
    torch.save(changed, other_cp)
    with pytest.raises(FileExistsError, match="different provenance"):
        cache_module().cache_embeddings(other_cp, cache, batch_size=2)


def test_legacy_cnn_cache_still_works(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    cfg = configuration(tmp_path)
    cfg["model"] = {"embedding_dim": 192, "dropout": 0.2}
    cp = train_b0(cfg, tmp_path / "cnn", smoke=True)
    cache_module().cache_embeddings(cp, tmp_path / "cnn_cache", batch_size=2)
    dataset = EmbeddingWindowDataset(cfg["data"]["manifest"], tmp_path / "cnn_cache", "val")
    assert len(dataset.expected_keys()) == 4
    assert torch.load(cp, weights_only=False)["model_type"] == "b0"


def test_gpu_wait_and_source_change_guard(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from psg_only.provenance import file_hash
    spec = importlib.util.spec_from_file_location("conformer_test_runner", "scripts/run_conformer_experiment.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    replies = iter(["12345", ""])
    monkeypatch.setattr(runner.subprocess, "run",
                        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=next(replies), stderr=""))
    slept, waited = [], []
    monkeypatch.setattr(runner.time, "sleep", slept.append)
    runner.wait_for_gpu(3, True, lambda: waited.append(True))
    assert slept == [30] and waited == [True]
    monkeypatch.setattr(runner.subprocess, "run",
                        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="12345", stderr=""))
    with pytest.raises(RuntimeError, match="GPU is busy"):
        runner.wait_for_gpu(3, False, lambda: None)
    source = tmp_path / "source.py"
    source.write_text("original")
    (tmp_path / "environment.json").write_text(json.dumps({"source_hashes": {"source.py": file_hash(source)}}))
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    runner.verify_source_snapshot(tmp_path)
    source.write_text("changed")
    with pytest.raises(RuntimeError, match="Source changed"):
        runner.verify_source_snapshot(tmp_path)
