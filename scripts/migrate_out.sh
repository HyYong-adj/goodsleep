#!/usr/bin/env bash
# 이 계정을 다른 서버로 통째로 옮긴다.
#
#   scripts/migrate_out.sh user@host:/path/to/choihy
#   scripts/migrate_out.sh /mnt/volume/choihy
#
# 저장소 디렉터리가 이미 vendor/(19GB) 와 artifacts/(850MB) 를 품고 있으므로
# 따로 staging 사본을 만들지 않는다. 중복 없이 한 번에 보낸다.
#
# 저장소 밖에 있어서 잊기 쉬운 것들(공용 registry, 계정 이력, git 이력)은
# `_transfer/` 에 모아 함께 보낸다. 대상 서버에서는 `scripts/migrate_in.sh` 를
# 돌려 검증하고 복원한다.
#
# 자격증명(.codex/auth.json)은 보내지 않는다. 새 서버에서 재로그인한다.
set -euo pipefail

source "$(dirname -- "${BASH_SOURCE[0]}")/_env.sh"
TARGET="${1:-}"
if [ -z "$TARGET" ]; then
  echo "사용법: $0 <user@host:/path/choihy | /local/path/choihy>" >&2
  exit 1
fi

STAMP="$(date -u +%Y%m%d)"
TRANSFER="$PSG_ROOT/_transfer"
SHARED_ROOT="$(cd "$PSG_ROOT/../.." && pwd)"   # /home/sleep

echo "== 1/3 저장소 밖 자산을 _transfer/ 에 모은다"
rm -rf "$TRANSFER"
mkdir -p "$TRANSFER"

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
du -sh "$TRANSFER"/* | sed 's/^/   /'

echo "== 2/3 대상으로 전송한다: $TARGET"
# .venv 는 기계에 묶여 있어 옮기지 않는다. 대상에서 setup_env.sh 로 다시 만든다.
rsync -a --info=progress2 --human-readable \
  --exclude='.venv/' --exclude='.venv-raw/' \
  --exclude='__pycache__/' --exclude='.pytest_cache/' \
  --exclude='.codex/auth.json' \
  "$PSG_ROOT/" "$TARGET/"

echo "== 3/3 전송 완료"
cat <<EOF

대상 서버에서 이어서 실행하라:

    cd <복원된 choihy 디렉터리>
    scripts/migrate_in.sh          # 검증만
    scripts/migrate_in.sh --venv   # 검증 + .venv 구성 + smoke 까지

자격증명은 보내지 않았다. Codex/Claude 는 새 서버에서 재로그인한다.
EOF
