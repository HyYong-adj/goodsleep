#!/usr/bin/env bash
# choihy-e027 (C1 동시학습) / e028 (B1 SNUBH 단독) 작업 큐. GPU 2/3 만 사용.
# 사용: bash scripts/chain_e027_e028.sh <gpu>  — 두 워커가 같은 큐에서 mkdir 잠금으로 작업을 나눠 가진다.
set -u; cd /home/sleep/researchers/choihy
GPU=$1
case "$GPU" in 2|3) ;; *) echo "GPU 0/1 은 타인 배정 — 사용 금지"; exit 1;; esac
export OMP_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=$GPU
LOCKS=artifacts/experiments/.locks_e027_e028; mkdir -p $LOCKS
JOBS="e027:psg,snubh:0 e027:psg,snubh:1 e028:snubh:0 e028:snubh:1 e027:psg,snubh:2 e027:psg,snubh:3 e028:snubh:2 e028:snubh:3 e027:psg,snubh:4 e028:snubh:4"
for job in $JOBS; do
  IFS=: read exp train fold <<< "$job"
  out=artifacts/experiments/${exp}_fold${fold}
  mkdir $LOCKS/${exp}_fold${fold} 2>/dev/null || continue
  until [ "$(nvidia-smi -i $GPU --query-gpu=memory.free --format=csv,noheader,nounits)" -ge 4096 ]; do sleep 60; done
  echo "[$exp f$fold gpu$GPU $(date -u +%FT%TZ)] start"
  /venv/main/bin/python -u scripts/run_multisource.py --experiment choihy-$exp --fold $fold --train $train \
    --epochs 50 --output $out > artifacts/logs/${exp}_fold${fold}.log 2>&1
  code=$?
  echo "[$exp f$fold gpu$GPU $(date -u +%FT%TZ)] exit=$code"
done
