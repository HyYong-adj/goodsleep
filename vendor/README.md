# vendor/ — 복사해 온 외부 의존 자산

이 트랙(choihy)은 원래 다른 연구자 소유 디렉터리와 공용 `data/` 를 절대경로로
직접 읽었다. 계정을 옮기면 그 경로가 전부 끊기므로 사본을 여기에 둔다.
**여기 있는 것은 사본이지 원본이 아니다.** 승인·소유권을 주장하지 않는다.

공용 `data/` 와 혼동을 피하려고 이름을 `vendor/` 로 했다. AGENTS.md 는
`data/`·`eval/`·`common/`·`integration/` 을 검토된 공용 영역으로 규정한다.
이 디렉터리는 그 영역이 아니며, 원본을 고치지 않는다.

## 내용

| 경로 | 출처 | 크기 | git |
|---|---|---|---|
| `splits/` | `/home/sleep/data/splits/` (공용, 승인된 동결 split) | 16 KB | **추적** |
| `manifests/` | `researchers/byoungjun/manifests/` | 108 MB | 제외 |
| `cache/audio_compact_full/` | `researchers/byoungjun/cache/audio_compact_full/` | 19 GB (235명, 1177 파일) | 제외 |
| `cache/teacher_targets/` | `researchers/byoungjun/cache/teacher_targets/` | 68 MB (235명) | 제외 |

`manifests/teacher_full.csv.bak_disk4path` 는 복사하지 않았다(중복 백업본).

git 추적 정책은 AGENTS.md 를 따른다 — 캐시된 logits/feature 와 대용량 산출물은
ordinary Git 에 넣지 않고 **위치와 해시만** 기록한다. 그 기록이
`MANIFEST.sha256`(작은 자산 484개의 sha256)이다.

## 이름을 바꾸면 안 되는 것

`cache/audio_compact_full` 의 **basename**. embedding provenance 가
`source_cache=Path(cache_root).name` 으로 이 문자열을 박아 두고 있어서,
디렉터리 이름을 바꾸면 기존 embedding 캐시와 provenance 가 어긋나 학습이
`Mixed embedding provenance` 로 거부된다.

## 검증

```bash
# 동결 split 무결성 (승인본과 같은지)
cd vendor/splits && sha256sum -c psg_audio_subject_split_v1.sha256

# 작은 자산 전체
cd vendor && sha256sum -c MANIFEST.sha256

# 19GB mel 캐시 집계 지문 — 아래 값과 같아야 한다
find cache/audio_compact_full -type f -printf '%P %s\n' | sort | sha256sum
# 5921a164be122032dd5d0603932b3f6ece778c91665477e57273f3e57fea9377
# 총 19,429,822,387 bytes / 1177 파일 / 235 피험자
```

2026-09-20 복사 시점에 `rsync -rcn`(전체 체크섬 대조)로 원본과 **바이트 동일**함을
확인했다. 차이 0건.

`night_stats.json` 235개는 원본 캐시 안에 이미 있던 값이며(2026-09-10 생성)
C2/C4 의 `night_norm` 재현에 필요하므로 함께 복사했다.

## 경로 해석

코드는 이 디렉터리를 직접 가리키지 않고 `psg_only.paths` 를 거친다.

- `PSG_VENDOR_ROOT` — 이 디렉터리 위치 (기본: 저장소 뿌리의 `vendor/`)
- `PSG_RAW_ROOT` — 원본 PSG-Audio 919 GB (기본: `/home/sleep/data/psg_audio`).
  **복사 대상이 아니다.** 이 트랙은 mel 캐시만 쓰므로 일상 학습에는 필요 없고,
  캐시를 처음부터 다시 만들 때만 있으면 된다.

새 config 는 절대경로 대신 `${VENDOR}` / `${REPO}` / `${RAW}` 를 쓴다. 그래야
`config_hash` 가 기계마다 달라지지 않는다. 보관된 실험 디렉터리의 `c1.yaml`·
`b1.yaml` 은 기록이므로 고치지 않았고, 옛 절대경로는 로드 시점에 해석된다.

## 재생성

사본을 잃었을 때 되살리는 방법은 자산마다 다르다.

- `splits/`, `manifests/` — 원본 보유자에게서 다시 받는다. 재생성 불가.
- `cache/teacher_targets/` — PSG teacher 모델(bj-e001) 없이는 재생성 불가.
- `cache/audio_compact_full/` — `scripts/cache_mels_raw.py` 로 원본 919 GB 에서
  다시 만들 수 있다. 원본 접근권이 있어야 하고 약 2.7 시간이 걸린다
  (`train_stats.json` 의 `elapsed_seconds` 9792).
