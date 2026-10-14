#!/usr/bin/env bash
# choihy-e015 : Conformer depth 2 -> 4, 1 seed 스크리닝 (GPU 2)
set -u; cd /home/sleep/researchers/choihy
export OMP_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=2
echo "[e015 $(date -u +%FT%TZ)] layers=4 seed 20260910"
/venv/main/bin/python -u scripts/run_phase3.py --seed 20260910 --epochs 50 --layers 4 \
  --output artifacts/experiments/e015_L4_s20260910 > artifacts/logs/e015_L4.log 2>&1
code=$?   # $(date) 가 서브셸로 실행되며 $? 를 덮으므로 먼저 잡아둔다
echo "[e015 $(date -u +%FT%TZ)] exit=$code"
