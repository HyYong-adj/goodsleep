#!/usr/bin/env bash
# choihy-e014 : e012(end-to-end, KD 없음) 재현 seed 2개 (GPU 1)
set -u; cd /home/sleep/researchers/choihy
export OMP_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=1
PY=/venv/main/bin/python
for s in 20260911 20260912; do
  echo "[e014 $(date -u +%FT%TZ)] seed $s"
  $PY -u scripts/run_phase3.py --seed $s --epochs 50 \
    --output artifacts/experiments/e014_repro_s$s > artifacts/logs/e014_s$s.log 2>&1
  code=$?   # $(date) 가 서브셸로 실행되며 $? 를 덮으므로 먼저 잡아둔다
  echo "[e014 $(date -u +%FT%TZ)] seed $s exit=$code"
done
echo "[e014 $(date -u +%FT%TZ)] 완료"
