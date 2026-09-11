# HomeSleepNet-like Baseline 구현 인계서

**목적:** 다음 구현자가 공개 PSG-audio 기반 4-stage baseline을 같은 split·전처리·평가로 재현하게 한다.  
**현재 상태:** 구현 전 설계 단계. 현재 경로에는 문서와 `codex`만 확인되며 코드, manifest, split, checkpoint는 없다.  
**이번 범위:** Gate A와 Baseline v0까지. UDA, consistency, distillation, mobile 배포는 v0 이후 독립 실험으로 진행한다.  
**기준일:** 2026-09-10  
**클래스 순서:** `0=Wake, 1=REM, 2=Light, 3=Deep`

---

## 0. 문서 권위와 값의 구분

충돌 시 연구계획/계약 → [MODEL_DEVELOPMENT_STRATEGY.md](./MODEL_DEVELOPMENT_STRATEGY.md) → [PROJECT_HANDBOOK.md](./PROJECT_HANDBOOK.md) → 승인된 `registry/DECISIONS.md` → 이 문서 → resolved config 순서로 따른다.

| 표기 | 의미 | 변경 방법 |
|---|---|---|
| **[PAPER]** | 논문 본문에서 확인한 사실 | 논문 근거 없이 변경하지 않음 |
| **[PROJECT]** | 상위 문서의 필수 계약 | 승인된 decision과 버전 변경 필요 |
| **[V0 DEFAULT]** | 공개 구현을 위한 기본값 | parent 대비 한 축만 바꾸는 실험으로 변경 |
| **[OPEN]** | 데이터·환경 확인 후 정할 값 | 코드에 숨기지 말고 decision log에 기록 |

> 공식 HomeSleepNet 코드와 원 데이터는 공개되어 있지 않다. 이 구현은 공식 복제가 아닌 **HomeSleepNet-like baseline**이다. 논문에 없는 값을 “논문 설정”이라고 표현하지 않는다.

## 1. 목표와 완료 경계

### 1.1 구현할 baseline

1. **v0a — single-epoch sanity baseline:** 30초 log-Mel 한 개를 CNN으로 분류해 정렬, label, loss, metric을 검증한다.
2. **v0b — sequence baseline:** 40개 epoch를 입력하고 중앙 20개를 CNN epoch encoder와 BiLSTM으로 예측한다. 후속 UDA·consistency·teacher 실험의 공통 parent다.

두 모델은 같은 manifest, split, feature contract, evaluator를 사용한다.

### 1.2 이번 범위가 아닌 것

- 비공개 원 데이터의 성능 직접 재현
- 미공개 architecture를 추정해 공식 구현과 동일하다고 주장
- v0에서 UDA, consistency, pseudo-label, ensemble을 동시 적용
- v0에서 package 50 MB 미만과 RAM 250 MB 이하 달성
- OSA 또는 임상 진단 성능 주장

| 구분 | Baseline v0 | 최종 프로젝트 |
|---|---|---|
| 데이터 | 공개 데이터, 고정 split | 승인된 real-home smartphone held-out 포함 |
| 모델 | CNN, CNN+BiLSTM | teacher, distilled student, optional cloud model |
| 성능 | 신뢰성과 재현성 우선 | hybrid Macro-F1 ≥0.60, κ ≥0.50, Accuracy ≥0.60 |
| 배포 | 파라미터 수 참고 | package <50 MB, RAM ≤250 MB, 8시간 검증 |

## 2. 논문 근거와 재현 한계

기준 논문 저장본은 [homesleepnet.pdf](./homesleepnet.pdf)다.

### 2.1 확인된 사실 [PAPER]

- audio와 PSG label을 30초 epoch로 정렬했다.
- adaptive noise reduction, Mel 변환, pitch shifting을 사용했다.
- SoundSleepNet 사전학습 파라미터로 초기화했다.
- 40개 Mel epoch 입력에서 중앙 20개를 예측했다.
- supervised loss는 cross-entropy다.
- UDA는 labeled hospital/source와 unlabeled home/target, domain BCE, adversarial training을 사용했다.
- UDA auxiliary loss에는 conditional entropy와 virtual adversarial training이 포함된다.
- consistency는 clean과 서로 다른 두 noisy view 예측의 Jensen–Shannon divergence다.
- noise는 Mel 영역에서 합성했고 SNR은 -10~10 dB였다.
- 세 training component를 동시에 실행했다.
- Adam, 고정 learning rate 0.0002, 20 epochs를 사용했다.

| 논문 평가 | Macro-F1 | Cohen's κ | Accuracy |
|---|---:|---:|---:|
| 4-stage | 0.582 | 0.416 | 59.4% |
| 3-stage | 0.714 | 0.557 | 76.2% |

이는 비공개 데이터의 **외부 참고치**다. 공개 데이터 v0의 pass/fail 기준이나 직접 비교 leaderboard로 쓰지 않는다.

### 2.2 논문만으로 확정할 수 없는 항목

- layer별 architecture와 checkpoint
- adaptive noise reduction 알고리즘
- 완전한 STFT/Mel/normalization 설정
- batch size, weight decay, seed, scheduler, early stopping
- 40→20 야간 경계 padding
- loss 가중치와 batch composition

기존 문서의 `20 mel × 1201 frames`는 현재 PDF 본문에서 확인되지 않는다. 아래에서는 **[V0 DEFAULT]**로만 사용한다.

## 3. 구현 전 열린 결정과 중단 조건

| ID | 확정할 것 | 증거 | 미확정 시 |
|---|---|---|---|
| D-01 | PSG-Audio 위치·버전 | root, checksum | loader만 작성, 학습 금지 |
| D-02 | Multimodal OSA 위치·버전 | root, checksum | cross-domain 평가 보류 |
| D-03 | 연구/상업 이용 범위 | license/DUA | 연구 외 사용 금지 |
| D-04 | audio와 PSG clock 관계 | sync metadata | 임의 offset 보정 금지 |
| D-05 | 사용할 channel | channel/device 설명 | room/mobile mic만 기본 사용 |
| D-06 | stable subject/night key | 식별 규칙 | split 생성 중단 |
| D-07 | unknown/movement label | label dictionary | valid class로 강제 변환 금지 |
| D-08 | compute/artifact store | URI, quota, access | 대규모 run 금지 |

subject-disjoint split, alignment, 데이터 권한 중 하나라도 보장할 수 없으면 학습을 중단한다. partition 사이에 subject/night/channel derivative/content hash가 겹치거나 raw audio·식별자·secret이 Git/로그에 포함돼도 해당 run은 `INVALID`다.

## 4. 목표 저장소 구조

```text
project/
├── configs/
│   ├── data/public_v0.yaml
│   └── teacher/{v0a_epoch_cnn,v0b_cnn_bilstm_40to20}.yaml
├── data/
│   ├── README.md
│   ├── manifests/{dataset}_{records,epochs}_v1.parquet
│   └── splits/public_v0_seed20260910.json
├── eval/
│   ├── VERSION
│   ├── {labels,splits,metrics,slices,runner}.py
│   ├── schemas/results.schema.json
│   └── fixtures/
├── common/
│   ├── audio/{io,features,alignment}.py
│   ├── data/{datasets,manifest,windows}.py
│   ├── models/{epoch_cnn,sequence,heads}.py
│   └── training/{engine,losses,seed,checkpoint}.py
├── registry/{EXPERIMENTS,DECISIONS,TEST_ACCESS,RELEASES}.md
├── researchers/r01/experiments/
├── reports/stage1/
├── tests/test_{labels,alignment,splits,features,windows,model_shapes,metrics}.py
└── tools/{build_manifest,make_splits,cache_features,audit_split}.py
```

label mapping, split, metric은 `eval/`만 소유한다. raw/processed audio, checkpoint, cached feature/logit은 Git에 넣지 않는다.

## 5. 데이터 계약

### 5.1 recording manifest

| 필드 | 타입 | 설명 |
|---|---|---|
| `recording_id` | string | 비식별 stable ID |
| `subject_id`, `night_id` | string | split group key |
| `dataset` | category | psg_audio, multimodal_osa 등 |
| `audio_uri`, `audio_sha256` | string | 승인 위치와 무결성 |
| `channel`, `device_id` | string/null | mic/channel domain |
| `sample_rate_hz`, `audio_start_sec` | int/float | 원본과 timeline |
| `label_uri`, `label_version` | string | annotation provenance |
| `rights_status` | category | approved_research/commercial, blocked, unknown |
| `quality_flags` | list[string] | clipping, dropout, sync uncertainty |

같은 subject의 모든 night, 같은 night의 모든 channel과 파생물은 같은 split에 둔다.

### 5.2 epoch manifest

| 필드 | 타입 | 설명 |
|---|---|---|
| `epoch_id`, `recording_id` | string | PK와 recording FK |
| `subject_id`, `night_id` | string | audit용 group ID |
| `epoch_idx` | int | 0-based |
| `start_sec`, `duration_sec` | float | 정상값 30i, 30.0 |
| `source_label`, `target` | string/null, int/null | 원본/canonical label |
| `valid_label`, `exclude_reason` | bool, string/null | 포함 여부와 이유 |
| `feature_uri`, `feature_sha256` | string/null | cache provenance |
| `split`, `domain_id` | category/string | partition과 domain |

### 5.3 label mapping [PROJECT]

```text
0 = Wake
1 = REM
2 = Light  (N1 + N2)
3 = Deep   (N3; legacy N4/S3/S4 포함)
```

movement, unknown, unscored, artifact-only는 `target=null, valid_label=false`다. 30초보다 짧은 끝 조각은 exclude하며 zero-padding 학습하지 않는다. source label을 보존한다. mapping/class order 변경은 evaluator major version 변경이다.

### 5.4 audio–label alignment

1. audio와 annotation 시작 시각을 canonical relative timeline으로 바꾼다.
2. label interval `[30i, 30(i+1))`의 samples를 정확히 추출한다.
3. 16 kHz resampling 후 정상 epoch가 480,000 samples인지 검사한다.
4. drift/gap은 보간으로 숨기지 말고 flag와 exclude reason을 남긴다.
5. 각 night의 처음·중간·끝에서 최소 3 epoch를 수동 점검한다.
6. epoch, label, 누락, 제외 수의 per-night report를 만든다.

기본 sync tolerance는 0.5초다. 초과 night는 자동 보정하지 않고 review queue로 보낸다.

## 6. split과 leakage 계약

### 6.1 기본 split [V0 DEFAULT]

- seed `20260910`
- `subject_id` 기준; 같은 subject의 모든 night를 한 partition에 둔다.
- 각 dataset 내부 subject 수 기준 70/15/15 train/validation/locked-test
- dataset, device/channel, subject dominant stage를 근사 stratify하되 group integrity 우선
- official subject split이 있으면 우선하고 출처 기록
- locked test는 v0 freeze 때 한 번만 쓰고 `TEST_ACCESS.md`에 기록

두 dataset 확보 시 **within-public**과 **cross-domain** 결과를 분리한다. cross-domain은 PSG-Audio 학습, Multimodal OSA 전체 target-held-out 평가다. 두 protocol을 평균하거나 같은 leaderboard 행에 섞지 않는다.

### 6.2 audit 실패 조건

- subject/night/channel 또는 content hash가 partition을 넘음
- normalization이 train 외 데이터를 참조
- augmentation/teacher target이 원본과 다른 split에 존재
- split에 없는 epoch 또는 중복 epoch 존재

split manifest에는 생성 command, seed, generator version, source manifest hash와 자체 hash를 기록하고 수동 편집하지 않는다.

## 7. feature 계약 [V0 DEFAULT]

| 항목 | 값 |
|---|---:|
| resample / samples | 16,000 Hz / 480,000 |
| waveform | mono float32 [-1,1], epoch mean 제거 |
| adaptive noise reduction | v0에서 사용하지 않음 |
| STFT | n_fft=512, Hann, win=400, hop=400 |
| center / power | true with reflect / 2.0 |
| Mel | 20 bins, f_min=20, f_max=8000, Slaney |
| log | 10 × log10(max(mel, 1e-10)) |
| raw shape | [1,20,1201] |
| clipping | train의 0.1/99.9 percentile |
| normalization | train-only global per-Mel-bin mean/std |
| dtype | float32 |

`1201`은 v0 계약값이다. 다른 shape를 조용히 crop/pad하지 말고 실패시킨다. cache key는 audio SHA-256, sample range, canonical feature config, feature code version의 SHA-256이다.

필수 검사는 silence/impulse/sine/noise의 finite output, 동일 input/config hash, shape `[1,20,1201]`, CPU/device `atol=1e-5, rtol=1e-4`, train-only normalization이다.

## 8. 40→20 window 계약

- input `[B,40,1,20,1201]`
- 중앙 positions `[10:30]`의 output `[B,20,4]`
- stride 20, night 경계를 넘지 않음
- 앞은 첫 valid epoch 10회 repeat-pad
- 뒤는 마지막 valid epoch를 필요한 만큼 repeat-pad
- 가짜 output은 mask하고 loss/metric support에서 제외
- 실제 epoch는 evaluation에서 정확히 한 번만 등장

`test_windows.py`는 night 길이 1, 19, 20, 21, 39, 40, 41에서 누락·중복 0을 검증한다.

## 9. 모델 명세 [V0 DEFAULT]

### 9.1 v0a

```text
Input [B,1,20,1201]
Conv(1→32,3) + BN + ReLU + MaxPool(2,4)
Conv(32→64,3) + BN + ReLU + MaxPool(2,4)
Conv(64→128,3) + BN + ReLU + MaxPool(2,4)
Conv(128→128,3) + BN + ReLU
AdaptiveAvgPool(1,1)
Linear(128→128) + ReLU + Dropout(0.3)
Linear(128→4)
```

### 9.2 v0b

```text
[B,40,1,20,1201]
→ shared v0a CNN → [B,40,128]
→ BiLSTM(input=128, hidden=128, layers=2, bidirectional, dropout=0.3)
→ [B,40,256] → positions [10:30]
→ Dropout(0.3) + Linear(256→4) → [B,20,4]
```

v0a checkpoint로 초기화하고 전체를 fine-tune한다. BiLSTM은 future context를 쓰므로 **post-wake/offline**이며 실시간 edge model이라고 표현하지 않는다. 모델은 logits, embedding, output_mask를 반환하고 forward에서 softmax를 적용하지 않는다.

## 10. 학습 명세 [V0 DEFAULT]

| 항목 | 값 |
|---|---|
| seeds | 17, 23, 42 |
| optimizer | AdamW, lr=2e-4, weight_decay=1e-4 |
| max epochs | 30 |
| scheduler | ReduceLROnPlateau, factor=0.5, patience=3 |
| early stop | validation Macro-F1, patience=7 |
| gradient clip | global norm 5.0 |
| effective batch | v0a 128 epochs, v0b 8 windows |
| checkpoint | best validation Macro-F1 |

OOM이면 accumulation으로 effective batch를 유지한다. 논문 근접 비교는 별도 config에서 `Adam, lr=2e-4, 20 epochs, no scheduler`를 쓴다.

```text
L = weighted_cross_entropy(valid_logits, valid_targets)
w[c] = 1 / sqrt(train_count[c])
w = w / mean(w), clip to [0.5, 3.0]
```

weight는 train에서만 계산한다. padded target을 제외한다. weighted sampler와 weighted CE를 동시에 쓰지 않는다. formal claim은 세 seed 평균±표준편차로만 한다.

checkpoint에는 states, epoch/step/seed, resolved config/hash, Git SHA, data/split/feature/evaluator version/hash, class counts/weights와 validation metrics를 넣는다. Git이 없으면 `UNVERSIONED`이며 formal result로 쓰지 않는다.

## 11. 평가 계약

필수 출력 [PROJECT]:

- Accuracy, Macro-F1, Cohen's κ
- class별 precision, recall, F1, support
- raw-count와 row-normalized confusion matrix
- dataset/device/channel slice와 가능한 demographic/noise slice
- 세 seed 평균·표준편차
- freeze checkpoint의 subject-level bootstrap 95% CI

class 하나라도 support가 0이면 formal 4-stage 결과는 `INVALID`다. 유효 class만으로 Macro-F1을 재평균하지 않는다.

prediction row에는 recording/subject/night ID, epoch/timestamp, target/prediction, 네 logits/probabilities, valid, dataset/device, model/preprocessing version을 포함한다. evaluator는 중복 epoch와 시간 역전을 실패 처리한다.

모델 선택은 validation Macro-F1 → κ/REM F1/Deep F1/worst-domain → 단순성 순이다. Accuracy 상승만으로 승격하지 않는다. practical tie 초기값은 Macro-F1 0.01이며 팀 승인 후 고정한다.

## 12. 구현 순서와 CLI 계약

### Phase 0 — foundation

```bash
python -m pytest tests/test_labels.py tests/test_metrics.py -q
```

repository skeleton, dependency lock, TEAM/registry 파일을 만들고 label/metric fixture부터 구현한다.

### Phase 1 — manifest, split, audit

```bash
python tools/build_manifest.py --config configs/data/public_v0.yaml --output data/manifests
python tools/make_splits.py --manifest data/manifests/public_epochs_v1.parquet \
  --seed 20260910 --output data/splits/public_v0_seed20260910.json
python tools/audit_split.py --manifest data/manifests/public_epochs_v1.parquet \
  --split data/splits/public_v0_seed20260910.json
```

통과: dataset별 count, overlap 0, 모든 class가 validation/test에 존재, alignment 수동 점검 완료.

### Phase 2 — feature

```bash
python tools/cache_features.py --config configs/data/public_v0.yaml --split train
python -m pytest tests/test_features.py tests/test_alignment.py -q
```

통과: fixture, train-only normalization, random 100 epoch NaN/Inf 0, 재생성 hash 일치.

### Phase 3 — v0a

```bash
python -m train --config configs/teacher/v0a_epoch_cnn.yaml run.mode=smoke seed=17
python -m train --config configs/teacher/v0a_epoch_cnn.yaml seed=17
```

smoke 통과: forward/backward, logits `[B,4]`, finite loss, non-zero gradient, 200 step 내 loss 감소, tiny subset accuracy ≥0.90 또는 원인 기록. 이후 seed 23, 42를 실행한다.

### Phase 4 — v0b

```bash
python -m pytest tests/test_windows.py tests/test_model_shapes.py -q
python -m train --config configs/teacher/v0b_cnn_bilstm_40to20.yaml \
  init_from=<v0a-checkpoint> run.mode=smoke seed=17
python -m train --config configs/teacher/v0b_cnn_bilstm_40to20.yaml \
  init_from=<v0a-checkpoint> seed=17
```

이후 seed 23, 42를 실행한다.

### Phase 5 — evaluation

```bash
python -m eval.runner --checkpoint <checkpoint> --split validation \
  --output <experiment-dir>/results.json
python -m eval.runner --checkpoint <frozen-v0b-checkpoint> --split locked_test \
  --output reports/stage1/v0b_locked_test_results.json
```

locked test 전에 `TEST_ACCESS.md`에 목적, candidate, 승인자를 기록한다.

## 13. 필수 테스트

| 영역 | 테스트 | 합격 |
|---|---|---|
| labels | known/unknown/movement/partial | canonical ID 또는 mask |
| alignment | synthetic timestamps | 기대 sample range |
| split | duplicate subject/night/channel/hash | audit 실패 |
| feature | 30초 waveform/repeat | [1,20,1201], finite, hash 동일 |
| window | 다양한 night 길이 | epoch 누락·중복 0 |
| model | v0a/v0b | [B,4] / [B,20,4] |
| loss | padded output | gradient 기여 0 |
| metric | hand-computed fixture | Accuracy/F1/κ 일치 |
| metric | missing class | formal result INVALID |
| save/load | 동일 input | logits tolerance 내 일치 |
| determinism | 동일 seed smoke 2회 | 합의 tolerance 내 일치 |

최소 두 사람이 같은 config와 artifact로 full baseline을 재현해야 Gate A를 통과한다.

## 14. 실험 산출물

초기 ID는 `s1-r01-e001-v0a-epoch-cnn`, `s1-r01-e002-v0b-cnn-bilstm-40to20`이다.

```text
researchers/r01/experiments/<experiment-id>/
├── experiment.md
├── config.yaml
├── environment.json
├── results.json
├── notes.md
└── artifacts.json
```

실행 전에 hypothesis, parent, 단일 변경점, 고정 요소, decision metric, compute, risk를 적는다. 모든 결과를 registry에 append한다. artifact manifest는 URI, SHA-256, size, producing experiment, access class, retention을 포함한다.

## 15. Baseline v0 완료 정의

### 데이터·평가

- [ ] dataset card와 rights status 존재
- [ ] recording/epoch manifest validation 통과
- [ ] label mapping과 30초 alignment 검사 통과
- [ ] subject/night/channel/content-hash overlap 0
- [ ] split, preprocessing, evaluator version/hash 고정
- [ ] locked test 접근 기록

### 모델·재현성

- [ ] v0a와 v0b smoke test 통과
- [ ] 세 seed validation 결과와 평균±표준편차 존재
- [ ] 40→20 stitching test 통과
- [ ] checkpoint에서 config/provenance 복원 가능
- [ ] 최소 두 사람의 reference baseline 재현

### 결과·보고

- [ ] canonical `results.json` 생성
- [ ] Accuracy, Macro-F1, κ, class metrics, confusion matrix 존재
- [ ] dataset/device/channel과 REM/Deep failure slice 보고
- [ ] 같은 split/evaluator에서 v0a와 v0b 비교
- [ ] 논문 수치와 공개 데이터 결과를 별도 표로 제시
- [ ] 한계, 실패 run, artifact URI/hash 기록

성능이 낮아도 무결성과 재현성 조건을 만족하면 v0는 완료될 수 있다. 높은 점수라도 leakage, alignment, provenance 또는 evaluator 조건을 위반하면 완료가 아니다.

## 16. v0 이후 확장

한 번에 한 축만 추가한다.

| 단계 | 변경 | 필수 비교 | 승격 조건 |
|---|---|---|---|
| v1 | realistic noise | clean/noise | real-domain 개선, clean 비열화 |
| v2 | two-view JS consistency | v1/v1+JS | 3-seed 또는 worst-domain 개선 |
| v3 | GRL domain adversarial | no-GRL/GRL | domain probe와 stage 동시 개선 |
| v4 | OPERA/AudioMAE | frozen/partial/full | v0b 대비 안정적 개선 |
| v5 | calibration/freeze | pre/post | ECE 개선, metric 비열화 없음 |
| v6 | mobile student KD | hard/logit/feature | device Pareto candidate |
| v7 | cloud temporal | logits/embedding | gain이 privacy 비용 정당화 |

논문형 consistency는 `L_supervised + lambda_js × JS(clean, noisy1, noisy2)`다. waveform mixing을 기본으로, Mel mixing을 별도 ablation으로 두며 대응 SNR은 -10~10 dB다.

논문형 UDA는 supervised, domain-adversarial, conditional-entropy, VAT loss를 분리 기록한다. target/home stage label을 UDA loss에 쓰지 않는다. long-context cache는 이 단계를 지연시키지 않는 후속 연구다.

## 17. 다음 구현자의 첫 실행

1. 실제 Git repository 위치를 확정한다.
2. TEAM.md에서 r01과 data/evaluation 책임자를 정한다.
3. D-01~D-08을 dataset card와 decision log에 답한다.
4. Section 4 skeleton과 dependency lock을 만든다.
5. label mapping과 metric fixture부터 구현한다.
6. recording manifest → epoch manifest → split → audit 순서로 진행한다.
7. 세 night의 처음·중간·끝 alignment를 사람이 확인한다.
8. feature fixture와 cache hash를 통과시킨다.
9. v0a smoke → seed 17 full → seed 23/42 순서로 실행한다.
10. window test 후 v0b를 같은 순서로 실행한다.
11. validation에서 후보를 고른 뒤 locked test를 한 번 실행한다.
12. Section 15와 Handbook Gate A checklist를 함께 닫는다.

데이터 권한, stable subject/night key, audio–label alignment 중 하나라도 불확실하면 모델 학습보다 해당 blocker 해결이 먼저다.
