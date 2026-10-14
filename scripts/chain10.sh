#!/usr/bin/env bash
# choihy-e018 : 5-fold CV, fold 3/4 (GPU 3)
set -u; cd /home/sleep/researchers/choihy
export OMP_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=3
for f in 3 4; do
  echo "[e018-B $(date -u +%FT%TZ)] fold $f"
  /venv/main/bin/python -u scripts/run_phase3.py --seed 20260910 --epochs 50 \
    --manifest /home/sleep/researchers/byoungjun/manifests/cv5/fold$f.csv \
    --output artifacts/experiments/e018_cv_fold$f > artifacts/logs/e018_fold$f.log 2>&1
  code=$?   # $(date) 가 서브셸로 실행되며 $? 를 덮으므로 먼저 잡아둔다
  echo "[e018-B $(date -u +%FT%TZ)] fold $f exit=$code"
done
echo "[e018-B $(date -u +%FT%TZ)] 완료"
