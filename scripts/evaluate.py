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
rows, checkpoint = (evaluate_b0(args.checkpoint, args.split) if payload["model_type"] == "b0" else evaluate_b1(args.checkpoint, args.split))
print(write_evaluation(rows, checkpoint, args.output))
