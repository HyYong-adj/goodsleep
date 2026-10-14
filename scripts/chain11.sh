#!/usr/bin/env bash
set -u; cd /home/sleep/researchers/choihy
export OMP_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=2
echo "[e019 $(date -u +%FT%TZ)] focal gamma=2"
/venv/main/bin/python -u scripts/run_phase3.py --seed 20260910 --epochs 50 --focal 2.0 \
  --output artifacts/experiments/e019_focal_s20260910 > artifacts/logs/e019.log 2>&1
code=$?   # $(date) 가 서브셸로 실행되며 $? 를 덮으므로 먼저 잡아둔다
echo "[e019 $(date -u +%FT%TZ)] exit=$code"
