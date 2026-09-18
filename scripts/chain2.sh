#!/usr/bin/env bash
# Phase 4 (C3 위 cosine/KD, 12런) -> Phase 3b (end-to-end 제대로: cosine + SpecAugment + 긴 예산)
set -u
cd /home/sleep/researchers/choihy
export OMP_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=3
PY=/venv/main/bin/python
L=artifacts/logs

echo "[chain2 $(date -u +%FT%TZ)] Phase 4 시작"
$PY -u scripts/run_phase4.py --output artifacts/experiments/phase4_20260917 > "$L/phase4_20260917.log" 2>&1
echo "[chain2 $(date -u +%FT%TZ)] Phase 4 exit=$?"

echo "[chain2 $(date -u +%FT%TZ)] Phase 3b 시작 (end-to-end, SpecAugment + cosine, 50 epochs)"
$PY -u scripts/run_phase3.py --seed 20260910 --epochs 50 \
  --output artifacts/experiments/phase3b_20260917 > "$L/phase3b_20260917.log" 2>&1
echo "[chain2 $(date -u +%FT%TZ)] Phase 3b exit=$?"
echo "[chain2 $(date -u +%FT%TZ)] 완료"
