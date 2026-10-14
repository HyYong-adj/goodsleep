#!/usr/bin/env bash
# choihy-e013 : end-to-end + gated KD, 3 seed (GPU 2)
set -u; cd /home/sleep/researchers/choihy
export OMP_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=2
PY=/venv/main/bin/python
for s in 20260910 20260911 20260912; do
  echo "[e013 $(date -u +%FT%TZ)] seed $s"
  $PY -u scripts/run_phase3.py --seed $s --epochs 50 --kd \
    --output artifacts/experiments/e013_kd_s$s > artifacts/logs/e013_s$s.log 2>&1
  code=$?   # $(date) 가 서브셸로 실행되며 $? 를 덮으므로 먼저 잡아둔다
  echo "[e013 $(date -u +%FT%TZ)] seed $s exit=$code"
done
echo "[e013 $(date -u +%FT%TZ)] 완료"
