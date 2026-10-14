#!/usr/bin/env bash
# choihy-e016 : gated KD alpha 0.3 -> 0.15, 3 seed (GPU 3)
set -u; cd /home/sleep/researchers/choihy
export OMP_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=3
for s in 20260910 20260911 20260912; do
  echo "[e016 $(date -u +%FT%TZ)] alpha=0.15 seed $s"
  /venv/main/bin/python -u scripts/run_phase3.py --seed $s --epochs 50 --kd --kd-alpha 0.15 \
    --output artifacts/experiments/e016_kd015_s$s > artifacts/logs/e016_s$s.log 2>&1
  code=$?   # $(date) 가 서브셸로 실행되며 $? 를 덮으므로 먼저 잡아둔다
  echo "[e016 $(date -u +%FT%TZ)] seed $s exit=$code"
done
echo "[e016 $(date -u +%FT%TZ)] 완료"
