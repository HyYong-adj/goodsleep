#!/usr/bin/env bash
# 새 서버에서 계정을 검증하고 실험을 이어갈 수 있는 상태로 만든다.
#
#   scripts/migrate_in.sh          # 검증만 (빠름)
#   scripts/migrate_in.sh --venv   # 검증 + .venv 구성 + 테스트 + 입력 검증
#
# 검증을 먼저 하는 이유: 전송 중 깨진 자산으로 학습을 돌리면 결과가 조용히
# 달라진다. 특히 동결 split 이 바뀌면 이전 실험과의 모든 비교가 무효가 되고
# test_locked 오염은 되돌릴 수 없다.
set -euo pipefail

source "$(dirname -- "${BASH_SOURCE[0]}")/_env.sh"
WITH_VENV=0
[ "${1:-}" = "--venv" ] && WITH_VENV=1

fail() { echo "  실패: $*" >&2; exit 1; }

echo "== 1 동결 split"
( cd "$PSG_ROOT/vendor/splits" && sha256sum -c psg_audio_subject_split_v1.sha256 ) \
  || fail "동결 split 이 승인본과 다르다. 이 상태로는 어떤 비교도 유효하지 않다."

echo "== 2 vendor 작은 자산 (manifests, teacher_targets, splits, train_stats)"
( cd "$PSG_ROOT/vendor" && sha256sum -c --quiet MANIFEST.sha256 ) \
  || fail "vendor 사본이 손상됐다."
echo "  OK $(wc -l < "$PSG_ROOT/vendor/MANIFEST.sha256") 개 파일"

echo "== 3 mel 캐시"
# 19GB 라 다른 사람이 옮기는 경우가 있다. vendor 밖에 있으면 PSG_MEL_CACHE 로
# 위치를 알려준다. 디렉터리 이름은 audio_compact_full 이어야 한다.
MEL="${PSG_MEL_CACHE:-$PSG_ROOT/vendor/cache/audio_compact_full}"
EXPECTED="5921a164be122032dd5d0603932b3f6ece778c91665477e57273f3e57fea9377"
if [ ! -d "$MEL" ] || [ -z "$(ls -A "$MEL" 2>/dev/null | grep -m1 '^[0-9]')" ]; then
  echo "  없음: $MEL"
  echo "  Phase2/3(encoder 재학습·end-to-end)은 이것 없이 돌지 않는다."
  echo "  다른 위치에 있다면:  export PSG_MEL_CACHE=/실제/경로/audio_compact_full"
  echo "  없다면 scripts/cache_mels_raw.py 로 원본에서 다시 만든다(약 2.7h)."
  MEL_OK=0
else
  [ "$(basename "$MEL")" = "audio_compact_full" ] \
    || fail "mel 캐시 디렉터리 이름이 audio_compact_full 이 아니다. embedding provenance 가 어긋난다."
  ACTUAL="$(cd "$MEL/.." && find "$(basename "$MEL")" -type f -printf '%P %s\n' | sort | sha256sum | cut -d' ' -f1)"
  if [ "$ACTUAL" = "$EXPECTED" ]; then
    echo "  OK 235 피험자 / 1177 파일  ($MEL)"
  else
    echo "  경고: 구성이 기준과 다르다 (기대 ${EXPECTED:0:12}…, 실제 ${ACTUAL:0:12}…)"
    echo "  다른 사람이 옮긴 사본이면 내용이 같은지 stats_hash 게이트로 확인된다."
  fi
  MEL_OK=1
fi

echo "== 4 이어갈 실험의 parent 산출물"
for p in artifacts/teacher_canonical \
         artifacts/experiments/20260914T062144930455Z/embeddings \
         artifacts/experiments/phase2_20260916/C3_s20260910 \
         artifacts/experiments/phase3b_20260917/best.pt; do
  [ -e "$PSG_ROOT/$p" ] || fail "$p 가 없다. e013~e015 를 이어갈 수 없다."
done
echo "  OK teacher_canonical / parent embedding / C3 / e012 체크포인트"

echo "== 5 저장소 밖 자산 (_transfer)"
if [ -d "$PSG_ROOT/_transfer" ]; then
  ( cd "$PSG_ROOT/_transfer" && sha256sum -c --quiet MANIFEST.sha256 ) || fail "_transfer 아카이브가 손상됐다."
  echo "  OK. 공용 기록물 복원은 아래 안내를 따른다(자동 전개하지 않는다 — 공용 영역이다)."
else
  echo "  _transfer 없음. 공용 registry 와 계정 이력은 따로 챙겼는지 확인하라."
fi

if [ "$WITH_VENV" = "1" ]; then
  echo "== 6 .venv 구성"
  [ -d "$PSG_ROOT/.venv" ] || "$PSG_ROOT/scripts/setup_env.sh"
  source "$(dirname -- "${BASH_SOURCE[0]}")/_env.sh"   # .venv 를 집도록 다시 고른다

  echo "== 7 테스트"
  "$PY" -m pytest "$PSG_ROOT/tests" -q

  if [ "${MEL_OK:-0}" = "1" ]; then
    echo "== 8 입력 검증 (실데이터 경로 전체)"
    "$PY" "$PSG_ROOT/scripts/validate_inputs.py" --config "$PSG_ROOT/configs/c1_conformer_epoch.yaml" > /dev/null
    echo "  OK"
  else
    echo "== 8 입력 검증 건너뜀 (mel 캐시 없음)"
  fi

  echo "== 9 GPU"
  "$PY" -c "import torch; print('  cuda:', torch.cuda.is_available(), '| devices:', torch.cuda.device_count())"
fi

cat <<'EOF'

검증 완료. 이어갈 작업:

  실험      scripts/chain3.sh      # choihy-e013 end-to-end + gated KD, 3 seed
            scripts/chain4.sh
  기록      STATUS.md 의 "Next experiments"

공용 기록물 복원(새 서버에 /home/sleep 구조를 세운다면):
  tar -I zstd -xf _transfer/shared_*.tar.zst -C <공용 루트>
  이미 다른 사람이 세운 공용 영역이라면 덮어쓰지 말고 내용만 대조하라.

계정 이력:
  tar -I zstd -xf _transfer/account_*.tar.zst -C .
  자격증명은 포함돼 있지 않다. Codex/Claude 는 재로그인한다.
EOF
