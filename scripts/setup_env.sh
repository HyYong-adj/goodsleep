#!/usr/bin/env bash
# 저장소 안에 학습용 venv(.venv)를 만든다.
#
# 지금까지는 호스트 이미지의 `/venv/main` 을 썼다. 그것은 이 계정 소유가
# 아니고 다른 연구자의 editable 설치가 얹혀 있으며, 계정을 옮기면 사라진다.
# `_env.sh` 는 `.venv` 가 있으면 그쪽을 먼저 쓴다.
#
# `--system-site-packages` 를 쓰지 않는다. 그걸 쓰면 기반 venv 의 .pth 가
# 그대로 처리돼 남의 src 가 다시 sys.path 에 올라오고, 독립이라는 목적이
# 무너진다.
#
# 주의: 이 스크립트는 `requirements-lock.txt` 의 고정 버전을 설치한다.
# CUDA 세대가 다른 기계에서는 그 휠이 없다. 그 경우 버전을 바꿔야 하고,
# 바꾼 사실을 실험 기록에 남긴 뒤 동일 환경 대조군과 함께 비교해야 한다.
set -euo pipefail

PSG_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
BOOTSTRAP="${PSG_PYTHON:-$(command -v python3)}"

if [ -z "$BOOTSTRAP" ] || [ ! -x "$BOOTSTRAP" ]; then
  echo "venv 를 만들 python 을 찾지 못했다. PSG_PYTHON 으로 지정하라." >&2
  exit 1
fi

"$BOOTSTRAP" -m venv "$PSG_ROOT/.venv"
"$PSG_ROOT/.venv/bin/python" -m pip install --upgrade pip
"$PSG_ROOT/.venv/bin/python" -m pip install -r "$PSG_ROOT/requirements-lock.txt"

# 검증도 환경의 일부다. 고정본에 pytest 가 없으면 dev 목록으로 채운다.
"$PSG_ROOT/.venv/bin/python" -c 'import pytest' 2>/dev/null \
  || "$PSG_ROOT/.venv/bin/python" -m pip install -r "$PSG_ROOT/requirements-dev.txt"

# 남의 src 가 sys.path 에 섞여 들어오지 않는지 확인한다.
"$PSG_ROOT/.venv/bin/python" - <<'PY'
import sys
foreign = [p for p in sys.path if "/researchers/" in p and "/choihy/" not in p]
if foreign:
    raise SystemExit(f"다른 연구자 경로가 sys.path 에 있다: {foreign}")
import torch, numpy, pandas, sklearn, yaml, pyedflib
print("OK", torch.__version__, "| cuda", torch.cuda.is_available())
PY
