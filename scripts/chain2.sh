#!/usr/bin/env bash
# Phase 4 (C3 위 cosine/KD, 12런) -> Phase 3b (end-to-end 제대로: cosine + SpecAugment + 긴 예산)
set -u
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
export OMP_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=3
L=artifacts/logs

echo "[chain2 $(date -u +%FT%TZ)] Phase 4 시작"
$PY -u scripts/run_phase4.py --output artifacts/experiments/phase4_20260917 > "$L/phase4_20260917.log" 2>&1
code=$?   # $(date) 가 서브셸로 실행되며 $? 를 덮으므로 먼저 잡아둔다
echo "[chain2 $(date -u +%FT%TZ)] Phase 4 exit=$code"

echo "[chain2 $(date -u +%FT%TZ)] Phase 3b 시작 (end-to-end, SpecAugment + cosine, 50 epochs)"
$PY -u scripts/run_phase3.py --seed 20260910 --epochs 50 \
  --output artifacts/experiments/phase3b_20260917 > "$L/phase3b_20260917.log" 2>&1
code=$?   # $(date) 가 서브셸로 실행되며 $? 를 덮으므로 먼저 잡아둔다
echo "[chain2 $(date -u +%FT%TZ)] Phase 3b exit=$code"
echo "[chain2 $(date -u +%FT%TZ)] 완료"
