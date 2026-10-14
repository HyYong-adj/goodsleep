#!/usr/bin/env bash
set -u; cd /home/sleep/researchers/choihy
export OMP_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=3
echo "[e020 $(date -u +%FT%TZ)] supcon w=0.1"
/venv/main/bin/python -u scripts/run_phase3.py --seed 20260910 --epochs 50 --supcon 0.1 \
  --output artifacts/experiments/e020_supcon_s20260910 > artifacts/logs/e020.log 2>&1
code=$?   # $(date) 가 서브셸로 실행되며 $? 를 덮으므로 먼저 잡아둔다
echo "[e020 $(date -u +%FT%TZ)] exit=$code"
