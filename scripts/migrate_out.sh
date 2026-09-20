#!/usr/bin/env bash
# 이 계정을 다른 서버로 통째로 옮긴다.
#
#   scripts/migrate_out.sh root@194.14.47.19:/home/sleep/researchers/choihy
#   scripts/migrate_out.sh /mnt/volume/choihy
#
# 기본값이 아닌 포트나 키는 RSYNC_RSH 로 준다(rsync 와 사전점검이 함께 쓴다):
#   RSYNC_RSH="ssh -p 22372 -i /root/.ssh/id_ed25519_migrate" \
#     scripts/migrate_out.sh root@194.14.47.19:/home/sleep/researchers/choihy
#
# 옵션:
#   --dry-run   실제로 보내지 않고 목록만 센다
#   --force     대상에 이미 artifacts/ 나 vendor/ 가 있어도 진행한다
#
# 저장소 디렉터리가 이미 vendor/(19GB) 와 artifacts/(850MB) 를 품고 있으므로
# staging 사본을 만들지 않는다. 중복 없이 한 번에 보낸다. 저장소 밖에 있어서
# 잊기 쉬운 것들(공용 registry, 계정 이력, git 이력)은 `_transfer/` 에 모은다.
#
# 자격증명(.codex/auth.json)은 보내지 않는다. 새 서버에서 재로그인한다.
# --delete 는 쓰지 않는다. 대상에 있는 것을 지우지 않는다.
set -euo pipefail

source "$(dirname -- "${BASH_SOURCE[0]}")/_env.sh"

DRY=0; FORCE=0; TARGET=""
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY=1 ;;
    --force)   FORCE=1 ;;
    -*)        echo "모르는 옵션: $arg" >&2; exit 1 ;;
    *)         TARGET="$arg" ;;
  esac
done
if [ -z "$TARGET" ]; then
  echo "사용법: $0 [--dry-run] [--force] <user@host:/path/choihy | /local/path/choihy>" >&2
  exit 1
fi

SSH_CMD="${RSYNC_RSH:-ssh}"
REMOTE=""; REMOTE_PATH="$TARGET"
case "$TARGET" in
  *:*) REMOTE="${TARGET%%:*}"; REMOTE_PATH="${TARGET#*:}" ;;
esac

run_there() {  # 대상에서 명령을 돌린다. 로컬 대상이면 그냥 돌린다.
  if [ -n "$REMOTE" ]; then $SSH_CMD "$REMOTE" "$@"; else bash -c "$*"; fi
}

echo "== 0/3 대상 사전점검: $TARGET"
if ! run_there "true" 2>/dev/null; then
  echo "  대상에 접속할 수 없다. 키가 등록됐는지, RSYNC_RSH 의 포트가 맞는지 확인하라." >&2
  exit 1
fi
run_there "mkdir -p '$REMOTE_PATH'"
echo "  여유 공간:"
run_there "df -h '$REMOTE_PATH' | tail -1" | sed 's/^/    /'

NEED=$(du -sb --exclude=.venv --exclude=.venv-raw --exclude=__pycache__ \
           --exclude=.pytest_cache "$PSG_ROOT" | cut -f1)
AVAIL=$(run_there "df -B1 --output=avail '$REMOTE_PATH' | tail -1" | tr -d ' ')
echo "  필요 $(numfmt --to=iec "$NEED") / 가용 $(numfmt --to=iec "$AVAIL")"
if [ "$AVAIL" -lt "$NEED" ]; then
  echo "  대상 여유 공간이 부족하다." >&2
  exit 1
fi

# 대상에 이미 실험 자산이 있으면 멈춘다. rsync 는 지우지 않지만, 섞이면
# 어느 산출물이 어느 실행의 것인지 알 수 없게 된다.
EXISTING=$(run_there "ls -A '$REMOTE_PATH' 2>/dev/null | head -20" || true)
if [ -n "$EXISTING" ]; then
  echo "  대상이 비어 있지 않다:"
  echo "$EXISTING" | sed 's/^/    /'
  OCCUPIED=$(run_there "ls -d '$REMOTE_PATH'/artifacts '$REMOTE_PATH'/vendor 2>/dev/null" || true)
  if [ -n "$OCCUPIED" ] && [ "$FORCE" != "1" ]; then
    echo "  artifacts/ 또는 vendor/ 가 이미 있다. 내용을 확인하고 --force 로 진행하라." >&2
    exit 1
  fi
fi

echo "== 1/3 저장소 밖 자산을 _transfer/ 에 모은다"
STAMP="$(date -u +%Y%m%d)"
TRANSFER="$PSG_ROOT/_transfer"
SHARED_ROOT="$(cd "$PSG_ROOT/../.." && pwd)"   # /home/sleep
rm -rf "$TRANSFER"; mkdir -p "$TRANSFER"

# 공용 기록물. /home/sleep 은 git 저장소가 아니라 어디에도 백업돼 있지 않다.
tar -C "$SHARED_ROOT" -cf - \
    registry RECOVERY.md TEAM.md AGENTS.md README.md data/README.md data/splits \
  | zstd -q -19 -T0 -o "$TRANSFER/shared_${STAMP}.tar.zst"

# 계정 이력. 재로그인으로 복구되지 않는다. 자격증명만 제외한다.
tar -C "$PSG_ROOT" --exclude=".codex/auth.json" -cf - .codex .claude \
  | zstd -q -10 -T0 -o "$TRANSFER/account_${STAMP}.tar.zst"

# git 이력 전체. GitHub 에 닿지 못하는 상황에서도 저장소를 되살릴 수 있다.
git -C "$PSG_ROOT" bundle create "$TRANSFER/repo_${STAMP}.bundle" --all 2>/dev/null

( cd "$TRANSFER" && sha256sum ./*.tar.zst ./*.bundle > "MANIFEST.sha256" )
du -sh "$TRANSFER"/* | sed 's/^/    /'

echo "== 2/3 전송: $TARGET"
RSYNC_OPTS=(-a --human-readable --partial
            --exclude='.venv/' --exclude='.venv-raw/'
            --exclude='__pycache__/' --exclude='.pytest_cache/'
            --exclude='.codex/auth.json')
if [ "$DRY" = "1" ]; then
  RSYNC_OPTS+=(--dry-run --stats)          # dry-run 에 진행률은 의미가 없다
else
  RSYNC_OPTS+=(--info=progress2)
fi
[ -n "$REMOTE" ] && RSYNC_OPTS+=(-e "$SSH_CMD")
rsync "${RSYNC_OPTS[@]}" "$PSG_ROOT/" "$TARGET/"

if [ "$DRY" = "1" ]; then
  echo "== dry-run 이라 실제로 보내지 않았다."
  exit 0
fi

echo "== 3/3 전송 완료"
cat <<EOF

대상 서버에서 이어서 실행하라:

    cd $REMOTE_PATH
    scripts/migrate_in.sh          # 검증만
    scripts/migrate_in.sh --venv   # 검증 + .venv 구성 + 테스트 + 입력 검증

자격증명은 보내지 않았다. Codex/Claude 는 새 서버에서 재로그인한다.
EOF
