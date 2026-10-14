#!/usr/bin/env bash
# GPU3: Phase 5 (end-to-end + gated KD) 3 seed
set -u; source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
export OMP_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=3
for s in 20260910 20260911 20260912; do
  echo "[chain3 $(date -u +%FT%TZ)] Phase 5 seed $s"
  $PY -u scripts/run_phase3.py --seed $s --epochs 50 --kd \
    --output artifacts/experiments/phase5_kd_s$s > artifacts/logs/phase5_s$s.log 2>&1
  code=$?   # $(date) 가 서브셸로 실행되며 $? 를 덮으므로 먼저 잡아둔다
  echo "[chain3 $(date -u +%FT%TZ)] seed $s exit=$code"
done
echo "[chain3 $(date -u +%FT%TZ)] 완료"
