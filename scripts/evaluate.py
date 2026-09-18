from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from psg_only.evaluate import evaluate_b0, evaluate_b1, write_evaluation

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--split", default="val", choices=["val"])
args = parser.parse_args()

import torch
payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
if payload["model_type"] == "temporal_transformer":
    from psg_only.transformer import evaluate_transformer
    rows, checkpoint = evaluate_transformer(args.checkpoint, args.split)
elif payload["model_type"] in {"b0", "b1", "conformer_epoch"}:
    evaluator = evaluate_b1 if payload["model_type"] == "b1" else evaluate_b0
    rows, checkpoint = evaluator(args.checkpoint, args.split)
else:
    raise ValueError("Unknown checkpoint model type.")
print(write_evaluation(rows, checkpoint, args.output))
