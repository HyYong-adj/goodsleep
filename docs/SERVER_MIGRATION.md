# 서버 이전 절차 (choihy 트랙)

이 인스턴스는 볼륨이 없어 종료되면 디스크가 사라진다. 이 문서는 계정을 다른
서버로 옮겨 **실험을 그대로 이어가기** 위한 절차다. 2026-09-20 에 실제 이전
리허설로 검증했다.

## 무엇이 어디에 있나

| 자산 | 크기 | 위치 | 이전 수단 |
|---|---|---|---|
| 코드·설정·문서·보고서 | 2 MB | git (`origin/main`) | git, 그리고 `_transfer/repo_*.bundle` |
| 동결 split v1 | 16 KB | git (`vendor/splits/`) | git |
| mel 캐시 (235명) | 19 GB | `vendor/cache/audio_compact_full/` | rsync, 또는 **다른 사람이 옮긴 사본을 `PSG_MEL_CACHE` 로 가리킴** |
| PSG teacher logits | 68 MB | `vendor/cache/teacher_targets/` | rsync |
| manifests (`teacher_full`, `cv5`) | 108 MB | `vendor/manifests/` | rsync |
| 실험 산출물·체크포인트 | 850 MB | `artifacts/` | rsync |
| 공용 registry·RECOVERY·TEAM | 11 KB | `_transfer/shared_*.tar.zst` | rsync |
| Codex/Claude 대화 이력 | 76 MB | `_transfer/account_*.tar.zst` | rsync |

`artifacts/` 와 `vendor/` 는 git 에서 제외돼 있다(AGENTS.md: 캐시된
logits/feature 와 대용량 산출물은 ordinary Git 금지). **git clone 만으로는
실험을 이어갈 수 없다.** rsync 가 반드시 필요하다.

원본 PSG-Audio 919 GB 는 옮기지 않는다. 일상 학습은 mel 캐시만 쓴다. 새 서버에
원본이 있다면 `PSG_RAW_ROOT` 로 위치만 알려주면 되고, 없으면 캐시 재생성만
불가능하다.

## 옮기기

옛 서버에서 한 줄이면 된다.

```bash
scripts/migrate_out.sh user@newhost:/path/to/choihy
```

기본 포트가 아니거나 키를 지정해야 하면 `RSYNC_RSH` 로 준다.

```bash
RSYNC_RSH="ssh -p 22372 -i /root/.ssh/id_ed25519_migrate" \
  scripts/migrate_out.sh root@194.14.47.19:/home/sleep/researchers/choihy
```

**mel 캐시를 다른 사람이 옮기는 경우** `--skip-mel-cache` 를 붙인다.
전송량이 20.7 GB 에서 1.3 GB 로 줄어든다. 223 바이트짜리 `train_stats.json` 은
`stats_hash` 게이트가 쓰므로 이 경우에도 함께 보낸다.

```bash
RSYNC_RSH="ssh -p 22372 -i ~/.ssh/id_ed25519_migrate" \
  scripts/migrate_out.sh --skip-mel-cache root@194.14.47.19:/home/sleep/researchers/choihy
```

그 캐시가 `vendor/` 밖에 놓이면 새 서버에서 위치를 알려준다. 다시 복사할
필요는 없다.

```bash
export PSG_MEL_CACHE=/home/sleep/researchers/byoungjun/cache/audio_compact_full
```

`_transfer/` 를 새로 만들고(공용 기록물·계정 이력·git 번들) 계정 전체를
rsync 한다. `.venv*`, `__pycache__`, `.codex/auth.json` 은 제외한다.
**자격증명은 옮기지 않는다** — 새 서버에서 재로그인한다.

로컬 볼륨에 먼저 받아도 된다.

```bash
scripts/migrate_out.sh /mnt/volume/choihy
```

mel 캐시를 함께 보내면 20.7 GB, 빼면 1.3 GB 다. rsync 라 중단되면 이어받는다.

## 새 서버에서

```bash
cd /path/to/choihy
scripts/migrate_in.sh --venv
```

순서대로 확인한다.

1. **동결 split 해시** — 다르면 즉시 중단한다. split membership 이 바뀌면
   이전 실험과의 모든 비교가 무효가 되고 `test_locked` 오염은 되돌릴 수 없다.
2. vendor 작은 자산 484 개 해시
3. mel 캐시 — 있으면 집계 지문(235 피험자 / 1177 파일)을 대조하고, 없거나
   다른 곳에 있으면 `PSG_MEL_CACHE` 안내를 띄우고 넘어간다(중단하지 않는다).
   이것 없이는 Phase2/3 만 못 돌고 나머지 검증은 끝까지 간다.
4. 이어갈 실험의 parent 산출물 — `teacher_canonical`, parent embedding,
   C3 체크포인트, e012 체크포인트
5. `_transfer` 아카이브 해시
6. `.venv` 구성 → 테스트 85 건 → 입력 검증(mel 캐시 있을 때) → GPU 확인

`--venv` 없이 돌리면 1~5 만 한다(빠름).

### 공용 영역

`_transfer/shared_*.tar.zst` 는 **자동으로 풀지 않는다.** `/home/sleep` 은
여러 연구자가 함께 쓰는 영역이고, 다른 사람이 이미 세워 둔 곳에 덮어쓰면
안 된다. 새로 세우는 경우에만 푼다.

```bash
tar -I zstd -xf _transfer/shared_*.tar.zst -C <공용 루트>
```

이미 있다면 풀지 말고 `registry/EXPERIMENTS.md` 등을 내용만 대조한다.

인프라(계정, 런처, Slack 봇) 복구는 별도 문서 `/home/sleep/RECOVERY.md` 가
다룬다. 이 문서는 연구 자산만 다룬다.

## 환경 재현 주의

`scripts/setup_env.sh` 는 `requirements-lock.txt` 의 고정 버전을 설치한다
(torch 2.14.0+cu130, CUDA 13.0 / cuDNN 9.24 기준).

**버전을 바꾸면 기존 값과 비교할 수 없다.** STATUS.md 방법론 발견 2 번대로
GPU 가 다르면 embedding 요소의 약 0.05% 가 달라지고 그 차이가 downstream 을
발산시켜 같은 encoder 인데 B1 이 .0117 차이 났다. 라이브러리 버전이 바뀌면
커널 선택이 또 달라진다. CUDA 세대가 달라 고정본을 쓸 수 없으면,
① 변경 사실을 실험 기록에 남기고 ② C0 같은 동일 환경 대조군을 함께 돌린다.
과거 체크포인트의 해시를 재현 기준으로 쓰지 않는다.

`/venv/main` 은 호스트 이미지의 것이라 새 서버에는 없다. 그 venv 에는 다른
연구자가 editable 설치한 `__editable__.sleep_audio_kd-0.1.0.pth` 가 있어
`byoungjun/src` 가 sys.path 에 올라왔다(모듈명이 겹치지 않아 실해는 없었다).
`setup_env.sh` 는 `--system-site-packages` 를 쓰지 않아 그 경로가 따라오지 않고,
설치 후 다른 연구자 경로가 sys.path 에 없는지 확인한다.

## 이어갈 실험

`STATUS.md` 의 "Next experiments" 가 기준이다. 바로 돌릴 수 있는 것:

```bash
scripts/chain3.sh   # choihy-e013 end-to-end + gated KD, 3 seed (구현·smoke 완료)
scripts/chain4.sh
```

e013 은 2026-09-18 시점에 CUDA 노드 장애로 BLOCKED 였다. 새 서버에서는
`migrate_in.sh --venv` 의 GPU 확인을 통과하면 바로 시작할 수 있다.

e015(5-fold CV)에 필요한 `vendor/manifests/cv5/` 도 함께 옮겨진다.

## 검증된 사실 (2026-09-20 리허설)

계정을 다른 경로로 복사해 실행한 결과:

- 테스트 85 건 통과
- `REPO_ROOT` 와 vendor 해석이 새 위치를 따라감
- GPU smoke 학습 완주, `validation_macro_f1` 이 원래 경로 실행과 **동일**
  (같은 GPU 이므로 비트 재현이 성립하는 구간)
- 새 체크포인트의 `manifest_hash`·`stats_hash` 가 기존 실험과 동일
- strace 기준 원래 계정 경로 접근 **0 건**, 다른 연구자 데이터 접근 **0 건**,
  자기 `vendor/` 접근 2914 건
