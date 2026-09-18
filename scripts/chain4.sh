#!/usr/bin/env bash
# GPU1: Phase 3b(end-to-end, KD 없음) 재현 seed 2개 — 0.5310 이 단일 seed 였다
set -u; cd /home/sleep/researchers/choihy
export OMP_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=1
PY=/venv/main/bin/python
for s in 20260911 20260912; do
  echo "[chain4 $(date -u +%FT%TZ)] Phase 3b seed $s"
  $PY -u scripts/run_phase3.py --seed $s --epochs 50 \
    --output artifacts/experiments/phase3b_s$s > artifacts/logs/phase3b_s$s.log 2>&1
  echo "[chain4 $(date -u +%FT%TZ)] seed $s exit=$?"
done
echo "[chain4 $(date -u +%FT%TZ)] 완료"
