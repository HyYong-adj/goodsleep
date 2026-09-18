#!/usr/bin/env bash
set -euo pipefail
PSG_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PSG_PYTHON="${PSG_PYTHON:-/venv/main/bin/python}"
"$PSG_PYTHON" -m venv --system-site-packages "$PSG_PROJECT_ROOT/.venv-raw"
"$PSG_PROJECT_ROOT/.venv-raw/bin/python" -m pip install -r "$PSG_PROJECT_ROOT/requirements-raw.txt"
"$PSG_PROJECT_ROOT/.venv-raw/bin/python" -c 'import torch, torchaudio, pyedflib, numpy, pandas; print(torch.__version__, torchaudio.__version__)'
