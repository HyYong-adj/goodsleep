from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from psg_only.data import validate_inputs
from psg_only.train import load_config

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
args = parser.parse_args()
cfg = load_config(args.config)
report = validate_inputs(cfg["data"]["manifest"], cfg["data"]["cache_root"], cfg["data"]["stats"])
print(json.dumps(report, indent=2))
