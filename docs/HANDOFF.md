# choihy 트랙 인수인계 (에이전트·연구자용)

**작성:** 2026-10-14 · 서버 이전 직전 스냅샷
**대상:** 이 트랙을 이어받는 사람이나 에이전트(Claude/Codex). 이 문서만 읽어도 지금 상태, 파일 위치, 다음에 할 일, 지켜야 할 규칙을 알 수 있게 썼다.

먼저 읽을 문서는 이 순서다. 이 문서 → `STATUS.md`(트랙 결론) → `docs/SNUBH_CROSS_DATASET_PLAN.md`(최근 작업 전체) → `docs/RESEARCH_WALKTHROUGH.md`(e001~e022 해설) → `docs/SERVER_MIGRATION.md`(이전 절차).
실험 원장은 `/home/sleep/registry/EXPERIMENTS.md` 의 choihy-e001~e028 이다. 공용 파일이라 git 에는 없고 `migrate_out.sh` 의 `_transfer/shared_*.tar.zst` 로 옮겨진다.

---

## 1. 이 트랙이 무엇인가

- 수면 단계 4분류(Wake/REM/Light/Deep)를 **실내 마이크 소리만으로** 하는 모델을 만든다. 이 레인의 모델은 Conformer epoch encoder + BiLSTM 을 end-to-end 로 학습한다.
- 데이터: PSG-Audio(그리스, 48 kHz 마이크, 학습 주력)와 SNUBH PSG(분당서울대병원 2014, 연구실 내부 자료, 영상 음성 트랙 WMA 16 kbps).
- 판단은 전부 validation(5-fold CV 또는 고정 val-44)으로 했다. `test_locked` 49명은 한 번도 열지 않았다. 공용 evaluator 와도 연동하지 않았다.

## 2. 현재 결론 (2026-10-14)

| 축 | 결론 | 근거 |
|---|---|---|
| PSG-Audio 성능 | parent = e014 (Conformer+BiLSTM e2e, SpecAugment, cosine). **5-fold CV .4954 ± .0217**. 팀 CNN 기준선과 같은 수준이다 | e014, e018 |
| 성능 개선 시도 | parent 이후 11개 실험이 모두 음성이다: KD, 용량, 앙상블, focal, SupCon, 후단 Transformer, front-end 특징, PCEN(e023 −.0025), logmel+delta(e024 −.0014) | e013–e024 |
| REM 병목 | REM 과 Light 는 소리(호흡 포락선)로 갈리지 않는다. 경증 코호트(SNUBH)에서도 같은 시간대끼리 비교하면 갈리지 않는다 → 원인은 "중증 OSA"가 아니라 **실내 마이크 소리 표현 자체**다 | e022, e026 |
| 외부 데이터 제로샷 | PSG-Audio 모델을 SNUBH 에 그대로 적용하면 **.260**(REM ≈ 0)이다. 정규화로 회복되는 몫은 +.02 뿐이다. 클래스 AUROC 가 무너졌다 → 코덱·장비 차이가 표현을 깨뜨린다 | e025 |
| 추가 데이터 효과 | PSG-Audio + SNUBH 동시 학습: SNUBH **.526**(+.257), PSG-Audio .484(−.011, fold 간 편차 안). SNUBH 단독: SNUBH .521, PSG-Audio .332. **두 데이터셋은 서로를 돕지 않는다.** 핵심은 평가 환경과 같은 녹음 조건의 데이터다 | e027, e028 |
| SNUBH 의 REM | 모델의 REM F1 은 .35–.39 다. 그런데 밤 중 위치만으로 예측해도 .308 이 나온다 → 소리가 보탠 몫은 +.04–.08 뿐이다 | e027 분석 |

## 3. 실험 목록 (최근 것, 전체는 레지스트리)

| ID | 무엇 | 결과 | 산출물 |
|---|---|---|---|
| e018 | 5-fold CV (parent 설정) | .4954 ± .0217 | `artifacts/experiments/e018_cv_fold{0-4}` |
| e023 / e024 | PCEN / logmel+delta front-end | 음성 | `e023_pcen_s20260910`, `e024_logmel_delta_s20260910` |
| e025 | e018 체크포인트로 SNUBH 외부 평가, 정규화 3조건 | .260 / .282 / .284 | `e025_snubh_external/` (+ `summary.json`) |
| e026 | SNUBH·PSG-Audio REM/Light/Deep 호흡특징 진단 (CPU) | REM-Light 비분리 (시간대 대조) | `e026_snubh_feature_probe/` |
| e027 | PSG-Audio + SNUBH 동시 학습, 5 fold | 위 표 | `e027_fold{0-4}/` |
| e028 | SNUBH 단독 학습, 5 fold | 위 표 | `e028_fold{0-4}/`; 집계 `e027_e028_summary.csv` |

중단돼 보관한 run 이 두 개 있다. 결과에는 쓰지 않는다.
- `e027_fold1_stopped_gpu2_owner_request`: GPU 2 소유자 요청으로 2 epoch 에서 멈춤.
- `e028_fold4_crashed_server_down`: 서버 장애로 14 epoch 에서 멈춤.

두 run 모두 GPU 3 에서 처음부터 다시 돌려 완료했다.

## 4. 코드 지도

| 파일 | 역할 |
|---|---|
| `src/psg_only/` | 데이터 로더(`data.py` MelWindowDataset), 모델(`endtoend.py`, `conformer.py`), 지표(`metrics.py`), 경로 해석(`paths.py`) |
| `scripts/run_phase3.py` | PSG-Audio 단일 데이터 e2e 러너(e012~e024). `--manifest` 로 cv5 fold 를 지정한다 |
| `scripts/run_multisource.py` | **다중 데이터셋 러너**(e027/e028). `--train psg,snubh` / `snubh`, `--snubh-fraction`(학습 곡선용). 매 epoch 양쪽 val 을 채점하고, 선택 기준별 checkpoint 를 `best_{primary,psg,snubh}.pt` 로 저장한다 |
| `scripts/build_snubh_cv5.py` | SNUBH 참가자 단위 5-fold(진단 AHI 층화)와 fold-train 정규화 통계 → `artifacts/manifests/snubh_cv5/` |
| `scripts/eval_external_snubh.py` | 동결 체크포인트를 SNUBH 에서 평가(e025). `--norm train/night/domain` |
| `scripts/snubh_feature_probe.py`, `snubh_feature_timematched.py` | e026 |
| `scripts/summarize_e025.py`, `summarize_e027_e028.py` | 집계. 같은 fold 기준선과 paired 로 비교한다 |
| `scripts/chain_e025.sh`, `chain_e027_e028.sh <gpu>` | 실행 큐. `chain_e027_e028.sh` 는 mkdir 잠금으로 여러 GPU 워커가 같은 작업 큐를 나눠 가진다 |
| `scripts/migrate_out.sh` / `migrate_in.sh` | 서버 이전(아래 §6) |

테스트: `pytest tests/` (09-23 기준 101건 통과).

## 5. 데이터 의존성 — 새 서버에 있어야 하는 것

| 자산 | 위치(현 서버) | 소유 | 이전 |
|---|---|---|---|
| PSG-Audio mel 캐시 19 GB | `vendor/cache/audio_compact_full` 또는 `PSG_MEL_CACHE` | byoungjun 원본, 사본은 이 트랙 | `migrate_out.sh` (또는 `--skip-mel-cache` 후 경로 지정) |
| PSG-Audio cv5 manifest | `/home/sleep/researchers/byoungjun/manifests/cv5/` (+ `vendor/manifests/cv5`) | byoungjun | vendor 사본은 migrate 로 옮겨진다. `run_multisource.py` 의 `SOURCES` 는 byoungjun 경로를 직접 가리키므로 새 서버에서 경로를 확인할 것 |
| **SNUBH 파생 데이터** | `/home/sleep/data/marg_psg_audio/_derived/snubh_psg/` (mel 캐시 `audio_compact_v1` 7.9 GB, 번들, FLAC 83 GB) | **pbj** (`pbj-e040`) | **이 트랙의 migrate 대상이 아니다.** pbj 가 옮기거나, 새 서버에서 `pbj_robust.multisrc` CLI 로 재생성한다. 이 트랙에 필요한 것은 mel 캐시 + `bundles/snubh_psg_external_v1` 뿐이다 |
| SNUBH 원본 | `/home/sleep/data/marg_psg_audio/SNUBH_PSG` (155 GB, **실명 포함**) | 연구실/pbj | 개인정보 원본이라 통제된 경로로만 옮긴다. 이 트랙은 원본을 직접 읽지 않는다 |
| SNUBH 참가자 메타 | `/home/sleep/researchers/pbj/external_manifests/snubh_psg/participants.csv` | pbj | `build_snubh_cv5.py` 가 읽는다 |
| 공용 레지스트리 | `/home/sleep/registry/EXPERIMENTS.md` | 공용 | migrate 의 `_transfer/shared_*` |

## 6. 서버 이전 / 백업

- **코드·문서·설정**: git(`origin` = GitHub `HyYong-adj/goodsleep`). `artifacts/`, `vendor/` 의 큰 파일, `*.pt` 는 `.gitignore` 로 제외돼 있다. AGENTS.md 가 체크포인트·캐시·raw audio 를 일반 Git 에 넣는 것을 금지하기 때문이다.
- **체크포인트·실험 산출물**: `artifacts/` 1.9 GB (`.pt` 114개, 777 MB). 해시 목록은 `docs/CHECKPOINTS.sha256`(git 추적)에 있고, 새 서버에서 `sha256sum -c docs/CHECKPOINTS.sha256` 로 검증한다.
- **SNUBH 로 학습한 체크포인트 30개**(`e027_*`, `e028_*`)는 내부 병원 데이터로 학습한 것이다. 연구 가설 확인용으로만 허용됐으므로 **공개 저장소·외부 공유 금지, 통제된 저장소에만 둔다.**
- **백업 아카이브 (2026-10-14, `_transfer/`, git 제외)**:
  - `artifacts_psgaudio_20261014T0914Z.tar.zst` (827 MB): PSG-Audio 로만 학습한 산출물과 체크포인트 84개.
  - `artifacts_snubh_INTERNAL_20261014T0914Z.tar.zst` (375 MB): e025–e028 산출물, SNUBH 학습 체크포인트 30개, SNUBH 분할. **내부 데이터 — 통제된 저장소 전용.**
  - 해시는 `artifacts_20261014T0914Z.sha256`. 복원: `tar -I zstd -xf <파일>` → `sha256sum -c docs/CHECKPOINTS.sha256`.
- 이전 명령: `scripts/migrate_out.sh [--skip-mel-cache] user@newhost:/path` → 새 서버에서 `scripts/migrate_in.sh --venv`. 상세는 `docs/SERVER_MIGRATION.md`.

## 7. 반드시 지킬 규칙

- **GPU**: choihy 작업은 **GPU 2/3 만** 쓴다. GPU 0/1 은 타인 배정이다. GPU 2 도 다른 사람이 쓰는 경우가 있으니 GPU 3 을 기본으로 쓴다. 꽉 차면 다른 GPU 로 옮기지 말고 기다린다.
- GPU 2 의 다른 사용자 작업은 과열·고부하 뒤 90초씩 멈춘다. 그동안 nvidia-smi 에서 프로세스가 사라져 보일 수 있으므로, "비었다"는 판정은 5초 간격으로 20분 이상 연속 관찰한 뒤에만 한다.
- 프로세스는 PID 로 종료한다. `pkill -f` 는 자기 셸과 매칭되므로 쓰지 않는다.
- **SNUBH 데이터 사용 범위**: 평가는 항상 허용된다. 학습은 2026-10-11 사용자 전달("연구가설 확인용 학습 가능") 이후 연구 목적으로만 한다. 승인 주체는 문서에 기록돼 있지 않으니 pbj 데이터 카드에 남겨야 한다. 상업 릴리스나 팀 release candidate 에 편입하는 것은 금지다.
- **개인정보**: `SNUBH_PSG/` 폴더명·헤더에 실명·생년월일·환자번호가 있다. 문서·로그·커밋에는 비식별 ID(`SNUBH-R-*`, `SNUBH-P-*`)만 쓴다. `_deid/` 매핑표는 열람하지 않는다.
- **분할**: 공식 split 과 cv5 만 쓴다. `test_locked` 는 열지 않는다. 실험은 레지스트리에 사전 등록하고, 실패·중단 run 도 기록한다.
- 같은 fold / 같은 seed 의 paired 차이로만 판단한다. fold 간 편차(~.02)가 seed 간 편차(~.004)보다 훨씬 크다.
- 선점 레인은 재실행하지 않는다. 코덱 증강·실측 소음 강건성은 pbj, Tracheal·AudioMAE 는 cloud9sm, RML 이벤트 보조는 sjh 다.

## 8. 다음 할 일 후보

1. **SNUBH 적응 곡선**: `run_multisource.py --train psg,snubh --snubh-fraction 0.1/0.25/0.5` 로 돌린다. "새 녹음 환경 데이터가 몇 밤 있으면 제로샷 성능 차이가 메워지는가"를 본다. 스마트폰 배포 결정과 바로 연결된다. 레지스트리 등록 후 진행한다.
2. **코덱 열화 증강**: pbj 레인이므로 제안만 한다. 근거는 e025 와 e027.
3. **팀 공유**: e025–e028 결과(데이터셋 간 비전이, REM 은 위치 효과)를 byoungjun(bj-e058 중증도 구성 실험)과 pbj(외부 평가)에 넘긴다.
4. **기록 보완**: STATUS.md 갱신(이 문서 기준). pbj 데이터 카드에 학습 허용 근거 기록을 요청한다.

## 9. 알려진 함정

- SNUBH 매니페스트의 `recording_start_seconds` 는 30초 격자에 있지 않다(ANALYSIS-START 기준). 내 로더는 30초 격자를 강제하므로 이 열을 뺀 파생 매니페스트를 쓴다(`build_snubh_cv5.py`, `eval_external_snubh.py` 가 처리).
- e018 등 옛 체크포인트의 config 에는 `post_transformer_*`, `in_channels` 키가 없다. 로드할 때 기본값을 채워야 한다(`eval_external_snubh.py` 의 `LEGACY_MODEL_DEFAULTS`).
- 번들의 `ahi_class` 는 녹화 기준이라 CPAP 밤은 치료 중 값이다. 환자 단위 층화에는 `participants.csv` 의 진단 AHI 를 쓴다.
- GPU 가 달라지면 비트 단위로 재현되지 않는다. 체크포인트 해시는 재현 기준이 아니라 파일 무결성 확인용이다.
- 체크포인트를 고르는 데 쓴 val 로 성능을 보고하면 낙관 편향이 생긴다(+.01~.11, regime 마다 다르다). 이 트랙의 수치는 모두 best-epoch val 이다.
