#!/usr/bin/env bash
# chain 스크립트 공통 부트스트랩.
#
# 이전에는 각 스크립트가 `cd /home/sleep/researchers/choihy` 와
# `PY=/venv/main/bin/python` 을 박아 두고 있었다. 둘 다 이 호스트에만 있는
# 경로다. 계정을 다른 곳으로 복원하면 앞의 것은 존재하지 않는 디렉터리를,
# 뒤의 것은 호스트 이미지가 제공하던 venv 를 가리킨다.
#
# 저장소 뿌리는 스크립트 위치에서 유도하고, 인터프리터는 고를 수 있게 한다.
# `.venv-raw` 는 CPU 전용 torch 이므로 학습 기본값으로 쓰지 않는다.

PSG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PSG_ROOT"

if [ -n "${PSG_PYTHON:-}" ]; then
  PY="$PSG_PYTHON"
elif [ -x "$PSG_ROOT/.venv/bin/python" ]; then
  PY="$PSG_ROOT/.venv/bin/python"
elif [ -x /venv/main/bin/python ]; then
  PY=/venv/main/bin/python
else
  PY="$(command -v python3)"
fi

if [ -z "$PY" ] || [ ! -x "$PY" ]; then
  echo "python 인터프리터를 찾지 못했다. PSG_PYTHON 으로 지정하라." >&2
  exit 1
fi

export PSG_ROOT PY
