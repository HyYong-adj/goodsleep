#!/usr/bin/env bash
# choihy-e025: e018 5-fold 체크포인트의 SNUBH 외부 평가 (평가 전용, GPU 3 만 사용).
set -u; cd /home/sleep/researchers/choihy
export OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=3
free=$(nvidia-smi -i 3 --query-gpu=memory.free --format=csv,noheader,nounits)
if [ "$free" -lt 4096 ]; then echo "GPU 3 free ${free} MiB < 4 GiB — 대기 후 재실행"; exit 1; fi
CK=$(ls artifacts/experiments/e018_cv_fold{0,1,2,3,4}/best.pt)
for norm in train night domain; do
  echo "[e025 $(date -u +%FT%TZ)] norm=$norm"
  /venv/main/bin/python -u scripts/eval_external_snubh.py --checkpoint $CK --norm $norm \
    --output artifacts/experiments/e025_snubh_external >> artifacts/logs/e025.log 2>&1
  code=$?
  echo "[e025 $(date -u +%FT%TZ)] norm=$norm exit=$code"
done
