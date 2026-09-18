#!/usr/bin/env python3
"""Local raw cache builder adapted from byoungjun/scripts/cache_audio_compact.py.
Preserves 8 kHz / 48 Mel / 1499 frame transform and legacy disk labels.
Adds complete raw audit and resume provenance checks; never modifies source data.
"""
from __future__ import annotations

import argparse
import json
import time
import sys
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pyedflib
import torch
import torchaudio


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from psg_only.raw import audit_raw
from psg_only.data import read_manifest
from psg_only.provenance import file_hash
TARGET_RATE = 8000
N_MELS = 48
N_FRAMES = 1499


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu", choices=["cpu"])
    parser.add_argument("--raw-root", default="/home/sleep/data/psg_audio")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--limit-subjects", type=int, default=0)
    return parser.parse_args()


def process_batch(waveforms, mel_transform, device):
    batch = torch.from_numpy(np.stack(waveforms)).to(device)
    batch = torchaudio.functional.resample(batch, 48000, TARGET_RATE)
    with torch.no_grad():
        mels = mel_transform(batch).clamp_min_(1e-10).log10_().mul_(10.0)
    if tuple(mels.shape[1:]) != (N_MELS, N_FRAMES):
        raise RuntimeError(f"Unexpected compact mel shape: {tuple(mels.shape)}")
    return mels.cpu().numpy()


def main() -> None:
    args = parse_args()
    frame = read_manifest(args.manifest)
    report = audit_raw(frame, args.raw_root)
    if report["status"] != "ready":
        raise RuntimeError(f"Raw copy/alignment is not ready: {report}")
    manifest_digest = file_hash(args.manifest)
    subjects = list(frame.groupby("subject_id", sort=True))
    if args.limit_subjects:
        subjects = subjects[: args.limit_subjects]
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=TARGET_RATE, n_fft=256, win_length=200, hop_length=160,
        f_min=30.0, f_max=3900.0, n_mels=N_MELS, power=2.0, center=False,
    ).to(device)
    total_sum = total_sq = 0.0
    total_count = rejected_total = 0
    started_all = time.perf_counter()

    for number, (subject_id, rows) in enumerate(subjects, 1):
        rows = rows.sort_values("subject_epoch_index").reset_index(drop=True)
        source_state = [(str(Path(name).name), Path(name).stat().st_size, Path(name).stat().st_mtime_ns)
                        for name in sorted(rows.edf_path.unique())]
        provenance = {"manifest_hash": manifest_digest, "source_state_hash": hashlib.sha256(json.dumps(source_state).encode()).hexdigest(),
                      "rml_hash": file_hash(Path(args.raw_root) / "V3" / "APNEA_RML" / f"{subject_id}.rml")}
        subject_root = args.output / subject_id
        subject_root.mkdir(exist_ok=True)
        mels_path, stats_path = subject_root / "mels.npy", subject_root / "stats.json"
        if mels_path.is_file() and stats_path.is_file() and (subject_root / "valid.npy").is_file():
            stats = json.loads(stats_path.read_text())
            if stats.get("provenance") != provenance:
                raise ValueError("Raw cache provenance changed; choose a new output directory.")
            for name in ("mels.npy", "labels.npy", "valid.npy"):
                if stats.get("files", {}).get(name) != file_hash(subject_root / name):
                    raise ValueError("Incomplete or corrupt raw cache; choose a new output.")
            total_sum += stats["sum"]
            total_sq += stats["sq_sum"]
            total_count += stats["count"]
            rejected_total += stats["rejected"]
            print(json.dumps({"subject": number, "total": len(subjects), "status": "exists"}), flush=True)
            continue
        started = time.perf_counter()
        output = np.lib.format.open_memmap(mels_path, mode="w+", dtype=np.float16, shape=(len(rows), N_MELS, N_FRAMES))
        valid = np.ones(len(rows), np.bool_)
        labels = rows.label_index.to_numpy(np.int64)
        batch_waveforms, batch_indices = [], []
        subject_sum = subject_sq = 0.0
        subject_count = subject_rejected = 0
        reader = None
        current_path = None

        def flush():
            nonlocal batch_waveforms, batch_indices, subject_sum, subject_sq, subject_count
            if not batch_waveforms:
                return
            values = process_batch(batch_waveforms, mel_transform, device)
            for value, output_index in zip(values, batch_indices):
                output[output_index] = value.astype(np.float16)
                if rows.at[output_index, "split"] == "train" and valid[output_index]:
                    value64 = value.astype(np.float64)
                    subject_sum += float(value64.sum())
                    subject_sq += float(np.square(value64).sum())
                    subject_count += value.size
            batch_waveforms, batch_indices = [], []

        try:
            for output_index, row in enumerate(rows.itertuples(index=False)):
                if row.edf_path != current_path:
                    if reader is not None:
                        reader.close()
                    reader = pyedflib.EdfReader(row.edf_path)
                    current_path = row.edf_path
                source_rate = int(round(float(row.sample_rate)))
                if source_rate != 48000:
                    raise ValueError(f"Expected 48 kHz Mic, got {source_rate}")
                start = int(round(float(row.local_start_seconds) * source_rate))
                waveform = reader.readSignal(int(row.channel_index), start=start, n=30 * source_rate).astype(np.float32)
                if len(waveform) != 30 * source_rate or not np.isfinite(waveform).all() or float(waveform.std()) < 1e-8:
                    valid[output_index] = False
                    subject_rejected += 1
                    waveform = np.zeros(30 * source_rate, np.float32)
                else:
                    waveform -= waveform.mean()
                batch_waveforms.append(waveform)
                batch_indices.append(output_index)
                if len(batch_waveforms) >= args.batch_size:
                    flush()
            flush()
        finally:
            if reader is not None:
                reader.close()
            output.flush()
        np.save(subject_root / "valid.npy", valid)
        np.save(subject_root / "labels.npy", labels)
        stats = {"sum": subject_sum, "sq_sum": subject_sq, "count": subject_count, "rejected": subject_rejected}
        after_state = [(str(Path(name).name), Path(name).stat().st_size, Path(name).stat().st_mtime_ns)
                       for name in sorted(rows.edf_path.unique())]
        if source_state != after_state:
            raise RuntimeError("Source EDF changed during feature extraction.")
        stats["provenance"] = provenance
        stats["files"] = {name: file_hash(subject_root / name) for name in ("mels.npy", "labels.npy", "valid.npy")}
        stats_path.write_text(json.dumps(stats) + "\n")
        total_sum += subject_sum
        total_sq += subject_sq
        total_count += subject_count
        rejected_total += subject_rejected
        print(json.dumps({"subject": number, "total": len(subjects), "epochs": len(rows), "rejected": subject_rejected, "seconds": round(time.perf_counter() - started, 2)}), flush=True)

    mean = total_sum / total_count
    variance = max(total_sq / total_count - mean * mean, 1e-12)
    summary = {"mean": mean, "std": variance ** 0.5, "train_elements": total_count, "rejected_epochs": rejected_total, "sample_rate_hz": TARGET_RATE, "n_mels": N_MELS, "n_frames": N_FRAMES, "elapsed_seconds": time.perf_counter() - started_all}
    (args.output / "train_stats.json").write_text(json.dumps(summary, indent=2) + "\n")
    if not args.limit_subjects:
        (args.output / "READY").write_text("ready\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
