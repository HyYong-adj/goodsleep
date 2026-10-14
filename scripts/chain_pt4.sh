#!/usr/bin/env bash
set -u; cd /home/sleep/researchers/choihy
export OMP_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=1
echo "[e021-L4 $(date -u +%FT%TZ)] post_transformer=4"
/venv/main/bin/python -u scripts/run_phase3.py --seed 20260910 --epochs 50 --post-transformer 4 \
  --output artifacts/experiments/e021_posttf4_s20260910 > artifacts/logs/e021_L4.log 2>&1
code=$?   # $(date) 가 서브셸로 실행되며 $? 를 덮으므로 먼저 잡아둔다
echo "[e021-L4 $(date -u +%FT%TZ)] exit=$code"
