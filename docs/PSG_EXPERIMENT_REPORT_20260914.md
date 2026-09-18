# PSG-only B0/B1 실험 내용 및 결과

보고 기준: 2026-09-14 실행 산출물. 이후 복사 상태나 새 실험 결과를 실시간으로 반영한 문서는 아니다.

## 1. 결론과 완료 범위

첫 실제 cache full 실험 `20260914T062144930455Z`가 `COMPLETE`로 종료됐다.
Train 191명 / validation 44명, 동일한 23,621 valid validation epochs에서
B0 Macro-F1 **0.363094 → B1 0.433996**, Cohen’s κ **0.216907 → 0.300808**로 개선됐다.

이번 단일 seed에서는 B1의 안정 구간 분류와 예측 전환율이 개선됐지만,
전환 구간 Macro-F1은 거의 그대로이며 REM/Deep 구분은 여전히 약하다.
이는 context 모델의 유용성을 지지하는 개발 결과다. 여러 seed와 독립 test에서 재현된 결론은 아니다.
**첫 full 실험의 상세 결과와 해석은 8절**에 정리했다.

아래 2–7절의 smoke 수치와 원본 복사 상태는 초기 설정 검증 기록으로 유지한다.
당시 실제 cache smoke는 모델별 2 updates, validation 926 epochs였고 B1은 majority baseline과 같았다.
이 smoke 결과와 이번 full 실험을 혼합하지 않는다.
Test split이 없으므로 모든 실제 데이터 성능은 validation-only다.
합성 EDF/RML 실험은 원본 처리 코드 검사이며 실제 수면 단계 예측 성능을 나타내지 않는다.

## 2. 실험 목적과 구성

가설은 동일한 frozen B0 acoustic embedding에 temporal context를 추가한 B1이 B0보다 validation Macro-F1을 높인다는 것이다.
초기 smoke의 판정 기준은 이 가설의 채택이 아니라 입력·학습·checkpoint·평가 연결과 무결성 확인이다.

```text
PSG 동기화 ambient Mic audio (30초)
  → log-Mel [1,48,1499]
  → B0 CNN → 192차원 embedding → Linear 4-class
                     ↓ freeze / subject별 cache
                  40 epochs
                     ↓
             B1 2-layer BiLSTM
                     ↓
             중앙 20 epochs 예측
                     ↓
    실제 validation epoch별 stitching 및 공통 evaluator
```

PSG 생체신호는 모델 입력으로 사용하지 않으며 PSG annotation이 정답 label을 제공한다.
canonical 순서는 `Wake, REM, Light, Deep`, version은 `wake-rem-light-deep-v1`이다.
기존 disk label `[0,1,2,3]`은 canonical `[0,2,3,1]`로 읽을 때 변환한다.

### 모델과 학습 설정

| 항목 | B0 | B1 |
| --- | --- | --- |
| 구조 | stride-2 Conv + depthwise blocks + global pooling | LayerNorm + 2-layer bidirectional LSTM + Linear |
| embedding / input dimension | 192 | 192 |
| hidden dimension | 해당 없음 | 128 (방향별) |
| 입력/출력 단위 | 30초 1 epoch → 4 logits | 40 epochs → 중앙 20 epochs × 4 logits |
| dropout | 0.2 | 0.2 |
| batch size | 16 | 32 |
| optimizer | AdamW | AdamW |
| learning rate | 0.0003 | 0.0003 |
| weight decay | 0.0001 | 0.0001 |
| loss | class-weighted CE, ignore_index=-100 | class-weighted CE, ignore_index=-100 |
| class weight | count^-0.5 / mean(weight) | count^-0.5 / mean(weight) |
| gradient clipping | 1.0 | 1.0 |
| seed | 20260910 | 20260910 |
| 실제 cache smoke optimizer steps | 2 | 2 |
| 예정 full 최대 epochs | 30 | 50 |
| full early stopping patience | 10 | 10 |
| 이번 실행 device | CPU | CPU |

설정의 `amp: true`는 CUDA 실행에서만 적용된다. 이번 CPU smoke에서는 AMP가 비활성화됐다.
B1의 stride는 20이며 야간 경계는 zero-pad하고 가짜 target은 loss와 metric에서 제외한다.
`subject_epoch_index`는 cache 조회용, `recording_start_seconds / 30`은 실제 시간축용으로 분리했다.
기존 manifest의 60초 간격 1곳은 B1에서 invalid gap으로 보존한다.

## 3. 데이터 범위

기존 `teacher_full.csv`의 subject split과 `audio_compact_full` Mel cache를 재사용했다.
원본 복사로 발견되는 V1/V2나 추가 V3 subject를 새 split에 자동 편입하지 않았다.

| 범위 | Train subjects | Val subjects | Test subjects | Manifest rows | Valid train epochs | Valid val epochs |
| --- | --- | --- | --- | --- | --- | --- |
| 전체 준비 데이터 | 191 | 44 | 0 | 135010 | 108543 | 23621 |
| 실제 cache smoke | 8 | 2 | 0 | 5288 | 4209 | 926 |

`Manifest rows`에는 invalid cache epoch가 포함되므로 실제 학습·평가 support와 구분한다.

| Class | 전체 train valid | 전체 val valid | Smoke train valid | Smoke val valid |
| --- | --- | --- | --- | --- |
| Wake | 8281 | 1959 | 282 | 46 |
| REM | 5362 | 881 | 116 | 23 |
| Light | 89473 | 19223 | 3667 | 776 |
| Deep | 5427 | 1558 | 144 | 81 |

전처리는 48 kHz ambient Mic의 epoch mean 제거 후 8 kHz로 resample하고,
48-bin log-Mel을 계산한다. n_fft=256, win_length=200, hop_length=160,
center=False, f_min=30 Hz, f_max=3900 Hz이다.
기존 전체 cache의 train normalization mean/std는 -40.8819716989 / 11.8089926024이며,
실제 cache smoke에서도 이를 사용했다. 이는 smoke 8명만으로 다시 계산한 통계가 아니다.

근거: [전체 preflight](../artifacts/ready_cached_20260914/cache_preflight.json),
[smoke preflight](../artifacts/setup_smoke_20260914/cache_preflight.json).

## 4. 실제 PSG-audio cache smoke 결과

Run: `setup_smoke_20260914`, 상태 `SMOKE_COMPLETE`.
B0/B1 모두 1회 validation을 수행했고 학습 optimizer update는 각각 2회다.
평가 대상 `(subject_id, epoch_index, target)` 집합이 동일함을 실행기가 검사했다.

### 전체 metric

| 모델 | Accuracy | Macro-F1 | Cohen’s κ |
| --- | --- | --- | --- |
| B0 | 0.049676 | 0.023663 | 0.000000 |
| B1 | 0.838013 | 0.227967 | 0.000000 |
| Majority (Light) | 0.838013 | 0.227967 | 0 (상수 예측으로 계산 가능; baseline JSON에는 미저장) |

Accuracy와 F1은 0–1 값이다. B1−B0 Macro-F1 차이는 0.204305이지만, B1이 majority baseline과 동일하므로 개선 효과로 해석하지 않는다.

### Class별 metric

| 모델 | Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- | --- |
| B0 | Wake | 0.049676 | 1.000000 | 0.094650 | 46 |
| B0 | REM | 0.000000 | 0.000000 | 0.000000 | 23 |
| B0 | Light | 0.000000 | 0.000000 | 0.000000 | 776 |
| B0 | Deep | 0.000000 | 0.000000 | 0.000000 | 81 |
| B1 | Wake | 0.000000 | 0.000000 | 0.000000 | 46 |
| B1 | REM | 0.000000 | 0.000000 | 0.000000 | 23 |
| B1 | Light | 0.838013 | 1.000000 | 0.911868 | 776 |
| B1 | Deep | 0.000000 | 0.000000 | 0.000000 | 81 |

### Confusion matrix

행은 정답, 열은 예측이며 순서는 `Wake, REM, Light, Deep`이다.

**B0 raw counts**

| 정답 / 예측 | Wake | REM | Light | Deep |
| --- | --- | --- | --- | --- |
| Wake | 46 | 0 | 0 | 0 |
| REM | 23 | 0 | 0 | 0 |
| Light | 776 | 0 | 0 | 0 |
| Deep | 81 | 0 | 0 | 0 |

**B1 raw counts**

| 정답 / 예측 | Wake | REM | Light | Deep |
| --- | --- | --- | --- | --- |
| Wake | 0 | 0 | 46 | 0 |
| REM | 0 | 0 | 23 | 0 |
| Light | 0 | 0 | 776 | 0 |
| Deep | 0 | 0 | 81 | 0 |

행 정규화 matrix도 각 `results.json`에 저장됐다. B0는 모든 행 `[1,0,0,0]`, B1은 모든 행 `[0,0,1,0]`이다.

### Temporal metric

| Metric | B0 | B1 |
| --- | --- | --- |
| transition_macro_f1 | 0.140625 | 0.166667 |
| stable_macro_f1 | 0.014381 | 0.230840 |
| true_transition_rate | 0.050230 | 0.050230 |
| predicted_transition_rate | 0.000000 | 0.000000 |
| transition_rate_error | 0.050230 | 0.050230 |

실제 시간축에서 연속한 valid epoch 쌍만 사용한다. transition/stable Macro-F1은 이전 정답 대비
현재 정답의 변화 여부로 현재 epoch를 분류하여 계산한다. 각 subset에서도 4개 class를 고정한다.
transition rate는 subject별 유효 인접 쌍의 변화 비율을 구한 뒤 subject 평균을 취한다.
rate error는 subject별 true/predicted rate의 절대 차이를 평균한다.
두 모델 모두 단일 class만 예측하므로 predicted transition rate가 0이다.

근거: [비교 결과](../artifacts/setup_smoke_20260914/comparison.json),
[B0 결과](../artifacts/setup_smoke_20260914/b0/evaluation/results.json),
[B1 결과](../artifacts/setup_smoke_20260914/b1/evaluation/results.json).

## 5. 원본 처리 검증

### 5.1 실제 원본 Mic 1 epoch probe

복사되어 읽을 수 있는 EDF에서 30초 ambient Mic를 읽고 CPU Mel 변환 후 기존 cache와 비교했다.

| 검사 | 결과 |
| --- | --- |
| Raw samples | 1440000 |
| Mel shape | [48, 1499] |
| Finite | True |
| 기존 cache 대비 mean absolute error | 0.006816 |
| 기존 cache 대비 max absolute error | 0.015930 |
| torch / torchaudio | 2.8.0+cpu / 2.8.0+cpu |

이 수치는 단일 epoch의 일치도 확인이며 전체 데이터의 audio–label 동기화나 byte-level 동일성을 보장하지 않는다. [Probe 결과](../artifacts/raw_feature_probe.json).

### 5.2 합성 EDF/RML end-to-end smoke

Run: `raw_fixture_smoke_20260914`, 상태 `SMOKE_COMPLETE`.

합성 신호와 4-class annotation으로 train/val 각각 1개 synthetic subject, 각 4 epochs를 만들었다.
원본 검사 → 새 Mel cache와 train normalization → B0 → embedding → B1 → 동일 validation 4 epochs 비교를 수행했다.
작은 dataset이므로 실제 optimizer update는 B0/B1 각각 1회였다.

| 모델 | Accuracy | Macro-F1 | Cohen’s κ |
| --- | --- | --- | --- |
| B0 | 0.250000 | 0.100000 | 0.000000 |
| B1 | 0.500000 | 0.375000 | 0.333333 |

위 숫자는 합성 입력에서 evaluator가 산출물을 만드는지 확인하기 위한 값이며 실제 수면 데이터 성능이 아니다.
합성 원본은 임시 디렉터리에서 생성 후 삭제됐으므로 당시 raw path는 현재 유효하지 않다.
남아 있는 모델·cache·평가 산출물로 결과를 확인할 수 있으며 원본부터 재실행하려면 fixture를 재생성해야 한다.
[합성 실험 비교 결과](../artifacts/raw_fixture_smoke_20260914/comparison.json).

### 5.3 실제 원본 전체 준비 검사

Run: `ready_raw_20260914`, 상태 `BLOCKED_RAW_COPY_OR_ALIGNMENT`.
다음은 2026-09-14 저장된 검사 시점의 snapshot이며 현재 복사 진행률과 다를 수 있다.

| 항목 | 수량 |
| --- | --- |
| 필요 EDF | 1243 |
| 검사 통과 EDF | 141 |
| 미복사 EDF | 1101 |
| 검증 실패 EDF | 1 |
| 필요 RML | 235 |
| 미복사 RML | 235 |

검증 실패 EDF 1개를 영구 손상으로 판정한 것은 아니다. 복사 진행 상태나 헤더·길이·정렬 등 검사 실패 원인은 재검사가 필요하다.
RML이 없으므로 annotation mismatch count 0을 전체 label 일치의 증거로 해석하지 않는다.
실제 원본 전체 변환·학습은 수행되지 않았고, 기존 cache 경로는 이와 별개로 preflight를 통과했다.
[원본 준비 보고서](../artifacts/ready_raw_20260914/raw_readiness.json).

## 6. 구현 보완과 검증

- manifest/cache label 일치 및 dataset 진입 시 split leakage 검사.
- embedding class order, B0/config/manifest/stats provenance와 파일 hash 검사.
- manifest 밖 embedding target 거부 및 stitching의 누락·추가·중복 검사.
- 실제 30초 시간축 보존과 invalid gap 마스킹.
- target이 전부 invalid인 batch 건너뛰기.
- true/predicted transition rate 및 subject 평균 집계 방식 기록.
- 지원하지 않는 B1 window 설정 거부, CUDA AMP, 학습 history와 단계별 실패 상태 기록.

설정 작업 당시 CPU 테스트 **24 passed**. 이번 보고서 작성에서는 새 학습이나 테스트를 재실행하지 않았다.
검증에는 class remap, 모델 shape, 경계 길이별 window coverage, mask, metric,
checkpoint logits round-trip, 동일 seed B1 재학습, 빈 target batch,
provenance 변조, label mismatch, leakage, 누락 prediction, 합성 EDF/RML 성공·실패 경로가 포함됐다.
GPU AMP의 실제 실행 및 전체 데이터 성능은 해당 CPU 검증에 포함되지 않는다. 이후 완료된 full 실험 결과는 8절에 별도로 기록한다.

## 7. 실행 기록과 재현

실제 cache smoke 실행 명령:

```bash
cd /home/sleep/researchers/choihy
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 /venv/main/bin/python \
  scripts/run_experiment.py --source cached --smoke --run --cpu \
  --output artifacts/setup_smoke_20260914
```

위 output은 이미 존재하므로 재실행 시 다른 새 폴더를 지정하거나 `--output`을 생략한다.
원본 복사 검사 명령은 다음과 같으며, 데이터가 불완전하면 exit code 2를 반환한다.

```bash
python scripts/run_experiment.py --source raw
```

학습 환경은 `/venv/main/bin/python`의 torch 2.14.0+cu130,
원본 CPU 전처리 환경은 `.venv-raw`의 torch/torchaudio 2.8.0+cpu다.
의존성과 설치 명령은 [실행 가이드](experiments/PSG_EXPERIMENT_QUICKSTART.md),
[학습 requirements](../requirements.txt), [테스트 requirements](../requirements-dev.txt),
[원본 requirements](../requirements-raw.txt)에 정리되어 있다.

### 실제 cache smoke provenance

**B0**

- `checkpoint_sha256`: `3f828ba2031c38b3472e7e35658bf2b7160cf5bd7631d4511c908d2f8b0c6d5e`
- `config_hash`: `9fe7c797dae1cd1b8f5f261407e9de79b784988c892a18b4b81dad1a02cd691e`
- `code_hash`: `c1df8e8de6915023441623de69a8978a0c1749a873921db3f4090fd512e710a6`

**B1**

- `checkpoint_sha256`: `ea1f5447f34cf69c191994fd369f24a2e24cb7d26e9e4679c2ea6383e12058c4`
- `config_hash`: `c2ce1dd4f45907adcd9b7a0ca11016490b0d5609f606cacca8e27fed41621d96`
- `code_hash`: `c1df8e8de6915023441623de69a8978a0c1749a873921db3f4090fd512e710a6`

`code_hash`는 평가 시점의 `src/psg_only/*.py` 내용 hash이며 전체 저장소 commit hash가 아니다.
checkpoint config에는 manifest 및 normalization hash가 저장되고,
B1 결과에는 사용한 B0 checkpoint와 embedding provenance가 기록된다.

## 8. 첫 full 실험 결과 및 해석

Run: `20260914T062144930455Z`. Seed: **20260910**, source: **cached**, smoke: **false**, 상태: **COMPLETE**.
B0 best encoder를 고정한 embedding으로 B1을 학습했다. Manifest와 validation epoch/target 일치를 실행기가 확인했다.

### 8.1 전체 성능

| 모델 | Accuracy | Macro-F1 | Cohen’s κ |
| --- | --- | --- |
| B0 | 74.76% | 0.363094 | 0.216907 |
| B1 | 76.70% | 0.433996 | 0.300808 |
| Majority: Light | 81.38% | 0.224337 | 별도 baseline JSON 미저장 |

B1−B0 차이는 Macro-F1 **+0.070902**, κ **+0.083901**, Accuracy **+1.95%p**다.

**해석:** validation의 Light 비율이 81.38%이므로 전부 Light로 예측해도 높은 Accuracy를 얻는다.
B1 Accuracy가 majority baseline보다 낮다는 이유만으로 B1을 열등하다고 평가하면 안 된다.
B1은 Macro-F1과 κ에서 실제 class 구분 능력을 보이지만, 수면 단계별 분류 성능은 고르지 않다.

### 8.2 Class별 개선과 남은 오류

| Class | Support | B0 F1 | B1 F1 | F1 차이 | B1 Precision | B1 Recall |
| --- | --- | --- | --- | --- | --- | --- |
| Wake | 1,959 | 0.463367 | 0.562986 | +0.099619 | 0.498426 | 0.646759 |
| REM | 881 | 0.044173 | 0.097470 | +0.053297 | 0.083001 | 0.118048 |
| Light | 19,223 | 0.853880 | 0.863607 | +0.009727 | 0.869412 | 0.857879 |
| Deep | 1,558 | 0.090956 | 0.211921 | +0.120965 | 0.298368 | 0.164313 |

네 class의 F1이 모두 개선됐다. 절대 F1 증가가 큰 class는 Deep과 Wake다.
그러나 REM F1은 0.0975, Deep F1은 0.2119로 아직 낮다.

- REM 881개 중 **681개(77.30%)**를 Light로 예측했고, 정답 REM으로 맞힌 것은 104개(11.80%)다.
- Deep 1,558개 중 **1,117개(71.69%)**를 Light로 예측했고, 정답 Deep으로 맞힌 것은 256개(16.43%)다.
- B1이 REM으로 예측한 1,253개 중 실제 REM은 104개뿐이다. REM precision도 8.30%로 낮다.

**해석:** REM/Deep이 Light에 흡수되는 문제가 남아 있다. REM은 recall뿐 아니라 precision도 낮으므로
소수 class weight를 더 키우는 것만으로 해결된다고 단정할 수 없다.
Class 불균형, acoustic embedding의 분리력, temporal context 부족은 후속 실험으로 구분할 가설이다.

B1 confusion matrix — 행: 정답, 열: 예측.

| 정답 / 예측 | Wake | REM | Light | Deep |
| --- | --- | --- | --- | --- |
| Wake | 1267 | 11 | 679 | 2 |
| REM | 91 | 104 | 681 | 5 |
| Light | 1073 | 1064 | 16491 | 595 |
| Deep | 111 | 74 | 1117 | 256 |

### 8.3 Temporal 성능의 의미

| Metric | B0 | B1 |
| --- | --- | --- |
| transition_macro_f1 | 0.378660 | 0.379130 |
| stable_macro_f1 | 0.351435 | 0.431280 |
| true_transition_rate | 0.052617 | 0.052617 |
| predicted_transition_rate | 0.147161 | 0.058020 |
| transition_rate_error | 0.097313 | 0.017948 |

**관측:** Stable Macro-F1은 0.3514→0.4313으로 증가했지만 transition Macro-F1은 0.3787→0.3791로 거의 같다.
실제 전환율은 약 5.26%, 예측 전환율은 B0 14.72%, B1 5.80%다.
개별 subject의 전환율 오차를 평균한 값도 0.0973→0.0179로 감소했다.

**해석:** B1은 안정 구간에서 더 일관된 예측을 만들고, B0의 과도한 단계 전환을 줄이는 방향으로 작동한 것으로 보인다.
다만 전환율이 정답에 가까워졌다고 해서 전환 시점을 정확히 찾았다는 뜻은 아니다.
전환 구간 F1이 거의 개선되지 않은 점은 남은 한계다. 단순한 smoothing과 더 나은 문맥 이해의 기여를 이 비교만으로 완전히 분리할 수는 없다.
또한 B0와 B1은 classifier 구조와 학습 과정도 다르므로, 개선 전부를 context 길이의 인과 효과로 단정하지 않는다.

### 8.4 학습 이력과 조기 종료

| 항목 | B0 | B1 |
| --- | --- | --- |
| 최대 설정 epochs | 30 | 50 |
| 실제 완료 epochs | 14 | 20 |
| Best epoch (1-based) | 4 | 10 |
| Best 시점 누적 optimizer steps | 27740 | 1770 |
| 종료 시점 누적 optimizer steps | 97090 | 3540 |

두 학습 모두 best 이후 10 epochs 동안 개선되지 않아 patience=10 규칙과 일치하게 종료됐다.
B0는 4번째 epoch에 최고점을 기록했고, 첫 epoch의 0.3624와 best 0.3631의 차이도 작다.
B1은 10번째 epoch까지 개선됐으나 이후 지속적인 향상은 없었다.

**해석:** 현재 설정에서 학습 epoch 수만 늘리는 것을 우선할 근거는 약하다.
B0의 정체·변동은 최적화 문제나 과적합 가능성을 점검할 이유가 되지만, 로그에 train loss가 없어 원인을 확정할 수 없다.
후속 B0 실험에서는 train loss를 기록하고 LR 3e-4→1e-4 변경을 한 축으로 검증할 수 있다.

### 8.5 이번 결과로 판단할 수 있는 범위

- 확인된 사실: 단일 seed, 동일 validation set에서 B1의 Macro-F1·κ·class별 F1과 전환율 오차가 B0보다 개선됐다.
- 남은 불확실성: 여러 seed 재현성, subject별 성능 편차, 독립 test 및 실제 가정 환경 일반화.
- Validation으로 best checkpoint를 선택했으므로 이 점수는 독립적인 최종 성능 추정치가 아니다.
- 원본 복사 미완료 상태의 기존 cache 실험이며, 전체 원본 audio–label 정렬 재검증을 완료한 결과는 아니다.

### 8.6 다음 실험으로 연결

현재 B0 embedding을 고정하고 **B1 input 40→80 epochs(20분→40분)**만 변경하는 실험을 준비했다.
중앙 target 20, stride20, seed, 모델 크기, loss, LR 및 최대 update 예산은 유지한다.
이는 더 긴 context가 REM/Deep 구분과 전환 구간 성능까지 개선하는지 확인하기 위한 실험이다.
좋은 후보가 나오면 같은 B0에서 40/80 context를 paired temporal seeds로 반복한다.
실행·판정 조건은 [80→20 실험 가이드](experiments/B1_80TO20_EXPERIMENT.md)를 따른다.

근거: [실험 상태](../artifacts/experiments/20260914T062144930455Z/experiment.json),
[전체 비교 결과](../artifacts/experiments/20260914T062144930455Z/comparison.json),
[B0 학습 이력](../artifacts/experiments/20260914T062144930455Z/b0/history.jsonl),
[B1 학습 이력](../artifacts/experiments/20260914T062144930455Z/b1/history.jsonl).

## 9. 남은 실험과 다음 단계

> **2026-09-16 상태 노트.** 아래 1–2번(B1 80→20 실험)은 완료됐다 —
> [B1 80→20 결과](PSG_B1_80TO20_REPORT_20260914.md), 개선 미확인(0.4340 → 0.4319).
> 3번(원본 복사 후 재처리)은 완료됐다 — 287명 EDF 1,537개(984.7 GB) 이전 및 헤더 대조 정상, RML 287 파싱 정상.
> 4번(다중 seed)은 choihy 트랙에서 여전히 미충족이다.
> 이후 실험 계보는 Transformer → Conformer 순이며,
> 현재 방향은 [다음 실험 방향 제언 rev.2](PSG_RESEARCH_DIRECTION_20260916.md)를 따른다.


1. 완료한 첫 full B0/B1을 기준으로, frozen embedding을 재사용하는 B1 80→20 context 실험을 수행한다.
2. 새 B1을 기존 40→20 B1과 동일 validation에서 비교하고 REM/Deep F1과 transition metric의 개선을 확인한다.
3. 원본 복사 완료 후 raw 검사와 전체 재처리를 별도 실행으로 기록한다.
4. 성능 개선 주장은 충분한 학습과 여러 seed 결과를 확보한 뒤 판단한다. 현재 test 결과는 없다.

기존 B0/B1 전체 파이프라인을 새 run으로 재실행하는 명령 (`GPU_INDEX`는 실제 배정받은 번호):

```bash
python scripts/run_experiment.py --source cached --run --gpu GPU_INDEX
```

### 산출물 목록

| 구분 | 경로 | 상태 |
| --- | --- | --- |
| 실제 cache smoke | [setup_smoke_20260914](../artifacts/setup_smoke_20260914/experiment.json) | SMOKE_COMPLETE |
| 합성 원본 smoke | [raw_fixture_smoke_20260914](../artifacts/raw_fixture_smoke_20260914/experiment.json) | SMOKE_COMPLETE |
| 기존 cache full 준비 | [ready_cached_20260914](../artifacts/ready_cached_20260914/experiment.json) | 최초 full 준비 snapshot |
| 첫 실제 cache full | [20260914T062144930455Z](../artifacts/experiments/20260914T062144930455Z/experiment.json) | COMPLETE |
| 실제 원본 full 준비 | [ready_raw_20260914](../artifacts/ready_raw_20260914/experiment.json) | BLOCKED_RAW_COPY_OR_ALIGNMENT |

이 보고서는 2026-09-14 보완된 파이프라인의 산출물을 기준으로 작성했다.
이전 `smoke_b0`, `smoke_b1` 등의 결과는 새 provenance 계약 적용 전이므로 이 비교 표에 혼합하지 않았다.
