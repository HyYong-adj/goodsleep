from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from psg_only.train import load_config, train_b0

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/b0_epoch.yaml")
parser.add_argument("--output", default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--smoke", action="store_true")
args = parser.parse_args()
cfg = load_config(args.config)
if args.seed is not None:
    cfg["seed"] = args.seed
stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
out = Path(args.output) if args.output else Path(cfg["output"]["root"]) / f"b0_{stamp}"
print(train_b0(cfg, out, args.smoke))
