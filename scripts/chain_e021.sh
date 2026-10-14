#!/usr/bin/env bash
# choihy-e021 재실행: GPU 2 에서 L2 -> L4 순차. (GPU 0/1 은 타인 소유 — 절대 사용 금지)
set -u; cd /home/sleep/researchers/choihy
export OMP_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=2
for n in 2 4; do
  echo "[e021-L$n $(date -u +%FT%TZ)] post_transformer=$n start"
  /venv/main/bin/python -u scripts/run_phase3.py --seed 20260910 --epochs 50 --post-transformer $n \
    --output artifacts/experiments/e021_posttf${n}_s20260910 > artifacts/logs/e021_L$n.log 2>&1
  code=$?
  echo "[e021-L$n $(date -u +%FT%TZ)] exit=$code"
done
echo "[e021 $(date -u +%FT%TZ)] 완료"
