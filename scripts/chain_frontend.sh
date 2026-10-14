#!/usr/bin/env bash
# choihy-e023 (PCEN) -> e024 (log-Mel + delta), GPU 2 순차. GPU 0/1 은 타인 소유 — 사용 금지.
set -u; cd /home/sleep/researchers/choihy
export OMP_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=2
PY=/venv/main/bin/python
run() {  # id frontend stats
  echo "[$1 $(date -u +%FT%TZ)] frontend=$2 start"
  $PY -u scripts/run_phase3.py --seed 20260910 --epochs 50 --frontend "$2" --frontend-stats "$3" \
    --output "artifacts/experiments/$1_$2_s20260910" > "artifacts/logs/$1.log" 2>&1
  code=$?   # $(date) 가 서브셸로 $? 를 덮으므로 먼저 잡아둔다
  echo "[$1 $(date -u +%FT%TZ)] exit=$code"
}
run e023 pcen         artifacts/frontend_stats/pcen_train.json
run e024 logmel_delta artifacts/frontend_stats/logmel_delta_train.json
echo "[frontend $(date -u +%FT%TZ)] 완료"
