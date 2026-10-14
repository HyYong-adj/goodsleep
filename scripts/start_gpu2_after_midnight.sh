#!/usr/bin/env bash
# 2026-10-11: GPU 2 는 다른 사람이 사용 중. 사용자 지시:
#   - 한국시간 자정(2026-10-12 00:00 KST = 2026-10-11 15:00 UTC) 이후에만 쓴다.
#   - 자정에 바로 시작하지 말고, **다른 사용자가 GPU 2 작업을 끈 뒤에** 시작한다.
# 판정: 5 초 간격으로 보고, GPU 2 에 compute 프로세스가 없고 메모리 사용 < 1 GiB 인 상태가 20 분 연속이면 "끔" 으로 본다.
# 주의: 상대 작업은 과열/고부하 후 90 초씩 멈추도록 설정돼 있어 그 사이 프로세스가 안 보일 수 있다.
#  - 90 초 정지가 연달아 와도 사이에 잠깐 올라오는 프로세스를 놓치지 않게 간격을 5 초로 촘촘히 본다.
#  - 창(20 분)은 90 초 정지보다 훨씬 길게 잡아 정지 구간을 종료로 오인하지 않게 한다.
set -u; cd /home/sleep/researchers/choihy
GPU2_UUID=GPU-e3d7e4ce-fb55-2370-c54e-093ecf6fc635
START=$(date -u -d "2026-10-11 15:00:00" +%s)
INTERVAL=5
NEEDED=240   # 240 × 5 s = 20 min
while [ "$(date -u +%s)" -lt "$START" ]; do sleep 60; done
echo "[gpu2 $(date -u +%FT%TZ)] midnight KST reached — waiting for the other user's GPU 2 process to end"
idle=0
while [ "$idle" -lt "$NEEDED" ]; do
  apps=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c "$GPU2_UUID")
  used=$(nvidia-smi -i 2 --query-gpu=memory.used --format=csv,noheader,nounits)
  if [ "$apps" -eq 0 ] && [ "$used" -lt 1024 ]; then idle=$((idle + 1)); else idle=0; fi
  sleep $INTERVAL
done
echo "[gpu2 $(date -u +%FT%TZ)] GPU 2 idle for 20 min — starting worker"
exec bash scripts/chain_e027_e028.sh 2
