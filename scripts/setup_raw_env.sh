#!/usr/bin/env bash
set -euo pipefail
# 인터프리터 선택은 _env.sh 와 같은 규칙을 쓴다. 호스트 이미지의 /venv/main 이
# 없는 기계에서도 설치가 진행되도록 한다.
source "$(dirname -- "${BASH_SOURCE[0]}")/_env.sh"
PSG_PROJECT_ROOT="$PSG_ROOT"
"$PY" -m venv --system-site-packages "$PSG_PROJECT_ROOT/.venv-raw"
"$PSG_PROJECT_ROOT/.venv-raw/bin/python" -m pip install -r "$PSG_PROJECT_ROOT/requirements-raw.txt"
"$PSG_PROJECT_ROOT/.venv-raw/bin/python" -c 'import torch, torchaudio, pyedflib, numpy, pandas; print(torch.__version__, torchaudio.__version__)'
