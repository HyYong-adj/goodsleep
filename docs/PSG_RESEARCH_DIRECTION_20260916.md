# 다음 실험 방향 제언 — 2026-09-16 (rev. 2)

본 문서는 새 학습을 실행하지 않았다. 기존 실험 산출물의 재집계, 원본 annotation 감사,
그리고 **팀 내 병행 트랙(byoungjun) 대조**만 수행했다.
실험 결과 보고서가 아니라 **트랙 위치와 다음 실험 축을 결정하기 위한 판단 문서**다.

> **rev. 2 변경 사유 (2026-09-16 오후):** 초판은 choihy 트랙만 보고 작성했다.
> 이후 `researchers/byoungjun`의 병행 트랙을 확인한 결과, 초판이 "다음 실험"으로 제안한 항목 중
> 상당수가 **이미 3 seed로 답이 나와 있었다**. 또한 2026-09-15 제품 정의 변경으로
> 초판이 전제한 teacher/student 구도가 바뀌었다. 해당 부분을 §5–§7에서 정정한다.
> 초판에서 유지되는 결론은 REM 데이터 제약(§3)과 "acoustic representation이 병목"(§2)이다.

## 1. 결론

1. **현재 choihy 트랙은 teacher 학습 단계가 아니다.** 형식상 Stage 1 teacher lane(비인과·모바일 제약 없음)에
   속하지만, 성능이 byoungjun의 *student*(0.523)보다 낮고(0.4833), 모델 규모도 1.9M params로
   이미 온디바이스 예산 안에 들어간다. 실제 KD teacher는 `bj-e001`(PSG 신호 입력, macro F1 0.672)로 별도 존재한다.
2. **teacher lane의 헤드룸은 이미 측정됐고 거의 0이다.** `bj-e022`(양방향·2배 폭)와 `bj-e037`(full-night GRU, 3 seed)이
   "시간 문맥 포화, 병목은 epoch 표현·데이터"를 확립했다. 초판이 제안한 long-context 축은 재실행 가치가 없다.
3. **choihy 트랙의 고유 자산은 Conformer epoch encoder 하나다.** byoungjun의 진단이 "병목은 epoch 표현"인데
   정작 그쪽 encoder는 평범한 Conv2d CNN(`CompactAudioEncoder`)이다. 이 빈칸을 채우는 것이 choihy의 기여 지점이다.
4. **REM 데이터 제약은 양 트랙에서 독립 확인됐다.** 4-stage Macro-F1 0.60은 현재 코호트에서 구조적으로 어렵다.
5. 따라서 다음 행동은 새 아키텍처 실험이 아니라 **트랙 조율 + 결손 요소 이식**이다.

---

## 2. 실험 궤적

모두 **단일 seed(20260910), validation-only**(44 subjects / 23,621 valid epochs), best-epoch 기준이다.

| 실험 | Macro-F1 | Accuracy | κ | Wake F1 | REM F1 | Light F1 | Deep F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Majority (Light) | 0.2243 | 0.8138 | – | – | – | – | – |
| B0 CNN 단독 | 0.3631 | 0.7476 | 0.2169 | 0.4634 | 0.0442 | 0.8539 | 0.0910 |
| B0 LR 1e-4 | 0.3662 | 0.7930 | 0.2296 | 0.4649 | 0.0049 | 0.8826 | 0.1123 |
| └ 그 embedding의 B1 40→20 | 0.3960 | 0.6999 | 0.2243 | 0.4823 | 0.0892 | 0.8172 | 0.1953 |
| B1 CNN+BiLSTM 40→20 | 0.4340 | 0.7670 | 0.3008 | 0.5630 | 0.0975 | 0.8636 | 0.2119 |
| B1 80→20 (40분) | 0.4319 | 0.7719 | 0.3023 | 0.5704 | 0.0961 | 0.8670 | 0.1940 |
| Transformer40 | 0.4069 | 0.7464 | 0.2461 | 0.5166 | 0.1223 | 0.8489 | 0.1398 |
| Transformer80 | 0.4086 | 0.7742 | 0.2578 | 0.5001 | 0.1618 | 0.8691 | 0.1035 |
| Conformer 단독 (C1) | 0.4127 | 0.7304 | 0.2301 | 0.4828 | 0.1319 | 0.8391 | 0.1969 |
| **Conformer + BiLSTM40** | **0.4833** | **0.7989** | **0.3565** | **0.5722** | **0.2322** | **0.8836** | **0.2451** |

### 2.1 패턴

- **Temporal 축 3전 3패.** context 2배(−0.0021), Transformer40(−0.0271), Transformer80(−0.0232).
- **Encoder 축 1전 1승.** Conformer 교체가 단일 실험 최대인 **+0.0493**.
- 이 패턴은 `bj-e037`(full-night GRU 3 seed: causal 0.517 / bidirectional 0.517 / epoch 단독 0.390)과
  **독립적으로 같은 결론**에 도달한다: 병목은 temporal이 아니라 epoch 표현이다.

### 2.2 수치 해석 시 필수 보정

byoungjun 트랙이 36 run에서 실측한 값이며, choihy 수치에도 동일하게 적용된다.

| 편향원 | 크기 | 의미 |
| --- | ---: | --- |
| best-epoch 선택 부풀림 | **+0.049** | best epoch F1 − 인접 epoch 평균 |
| 고정 val-44 낙관 편향 | **+0.027** | 고정 val 0.509 vs 5-fold CV 평균 0.482 |
| fold 간 표준편차 | 0.035 | seed 표준편차 0.017보다 큼 |

즉 **0.4833은 절대값으로 신뢰할 수 없다.** seed 쌍 비교·paired 분석만 유효하다.
choihy 트랙도 평가 기준을 5-fold CV(`byoungjun/manifests/cv5/`)로 옮겨야 한다.

### 2.3 미기록 실험

`artifacts/experiments/b0_lr1e4_20260914T154714`는 상태 `COMPLETE`인데 보고서가 없다.
B0는 0.3631 → 0.3662로 소폭 올랐으나 downstream B1이 0.4340 → **0.3960**으로 악화됐다
(REM F1은 B0 단계에서 0.0442 → 0.0049로 붕괴). 전략 문서 §7.2의 부정 결과 기록 규칙에 따라 보고서가 필요하다.

---

## 3. REM 데이터 제약 — 양 트랙 독립 확인

### 3.1 감사 결과

Validation의 Light 비중 81.4%가 생리학적으로 비정상이라 원본까지 역추적했다.

**Label mapping은 정상이다.** [src/psg_only/raw.py:29](../src/psg_only/raw.py#L29)의
`{'Wake':0, 'NonREM1':1, 'NonREM2':1, 'NonREM3':2, 'REM':3, 'NotScored':-100}`와
[src/psg_only/constants.py](../src/psg_only/constants.py)의 canonical 재배열 `(0,2,3,1)`은 AASM 규약과 일치한다.
`UserStaging`만 읽고 `MachineStaging`을 배제하는 XPath도 올바르다.

V3 전체 287 subjects UserStaging duration 가중 집계:

| | N2 | N1 | Wake | N3 | REM | NotScored |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| UserStaging | 73.22% | 9.50% | 7.71% | 4.80% | **4.77%** | 0.00% |
| MachineStaging (참고) | 26.72% | 17.96% | 49.87% | 3.22% | 2.09% | 0.15% |

| 항목 | 값 |
| --- | ---: |
| REM이 밤 전체에 0인 subject | **102 / 287 (35.5%)** |
| REM < 5%인 subject | 187 / 287 (65.2%) |
| Subject별 REM 비율 중앙값 | **0.026** |
| 녹음 길이 중앙값 (최소 / 최대) | **4.45 h** (1.27 / 9.00) |

교차 확인:

- **라벨이 잘린 것이 아니다.** 표본 8 subject에서 staging span / EDF 총 길이 비율 **0.91–1.00**.
  staging은 녹음 전체를 덮고 있고, 녹음 자체가 4~5시간으로 짧다.
- **스코어러 오류가 아니다.** UserStaging REM=0인 102명 중 **79명은 MachineStaging도 REM=0**.

### 3.2 독립 확인과 메커니즘

byoungjun 트랙이 원본 서버 대조로 동일 결론에 도달했다(`reports/label_audit/`):
Light 82.3% / Wake 7.7% / Deep 5.2% / REM 4.7%, 판독 길이 중앙값 4.4 h, **235명 중 90명 REM 0**.
본 문서의 287명 기준 102명과 일관된다(235는 train+val 부분집합).

그쪽에서 **메커니즘까지 규명했다**(`subject_covariates_20260915.csv`):

> 중증 OSA(**AHI 중앙값 54**), **split-night 57명**, RML 이벤트(각성·무호흡·코골이) 대량
> → REM 희소는 데이터 특성.

`test_locked` 49명도 REM 4.8% / Deep 2.6%로 동일하다. 즉 test에서도 개선되지 않는다.

### 3.3 의미

현재 최고 모델에서도 REM 정답의 **73.78%가 Light로**, Deep 정답의 **70.03%가 Light로** 분류된다.
4-stage Macro-F1 ≥ 0.60에 도달하려면 REM F1이 약 0.5여야 하나 현재 0.2322다.
**이 격차는 encoder/context 교체로 메울 수 있는 성질이 아니다.**

byoungjun 트랙의 결론도 동일하다: "4-stage 0.60은 KD/구조 개선보다 **full-night 판독 데이터 확보 또는
3-stage 중심 협의**가 현실적 경로." PSG teacher조차 REM F1 0.58 / precision 0.51로,
KD로 전달할 REM 지식 자체가 약하다.

한계: 본 감사는 V3 `UserStaging` XML 집계이며 PSG-Audio 원 논문 보고값과 대조한 결과는 아니다.
"데이터가 잘못됐다"가 아니라 **"이 코호트는 REM이 희박하다"**가 확인된 사실이다.

---

## 4. 성능 격차 분석 — byoungjun 트랙과의 차이는 어디서 오는가

### 4.1 격차의 실제 크기

| 지점 | Macro-F1 | 조건 |
| --- | ---: | --- |
| byoungjun baseline (e004) | 0.452 ± 0.005 (3 seed) | 평범한 CNN encoder, combo 없음 |
| **choihy Conformer + BiLSTM40** | **0.4833** (1 seed) | combo·KD·augmentation **전부 없음** |
| byoungjun combo (e021) | 0.506 ± 0.019 (3 seed) | +0.054 |
| byoungjun skgkd2step (e027) | 0.523 ± 0.016 (3 seed) | +0.017 |
| byoungjun homeaug + SK/gated KD | 0.548 (1 seed) | 파형 augmentation 추가 |

**부가 요소가 없는 지점끼리 비교하면 choihy 쪽이 0.452 → 0.4833으로 앞선다.**
격차는 모델 품질이 아니라 **누적된 부가 요소의 유무**에서 온다.

비교 한계: byoungjun 수치는 공용 evaluator 재산출 전이고, causal TCN(RF 127) vs 비인과 BiLSTM40이라
통제 비교가 아니다. 지표(indicative)로만 사용한다.

### 4.2 확인된 차이 (코드 대조)

| # | 요소 | byoungjun | choihy | 근거 |
| --- | --- | --- | --- | --- |
| 1 | encoder-temporal 학습 | **end-to-end 공동 학습** | **frozen 2단계** | `AdamW(model.parameters())` vs `requires_grad_(False)` |
| 2 | 정규화 | **피험자(밤)별 mean/std** | 학습셋 전역 통계 1쌍 | `audio_sequence_data.py:61` vs `train_stats.json` |
| 3 | Augmentation | SpecAugment (freq 8 / time 150 / gain σ0.25) + 파형 augmentation | **없음** | `train_audio_temporal.py:75` |
| 4 | 시각 특징 | 경과시간 `[h, h²]` → embedding 가산 | **없음** | `audio_temporal.py:97,128` |
| 5 | KD | PSG teacher(0.672)로부터 gated KD | **없음** | `bj-e025/e027` |
| 6 | 보조 손실 | `local_head` epoch 단위 deep supervision | 없음 | `audio_temporal.py:125` |
| 7 | LR 스케줄 | cosine + warmup 3 | **`scheduler: none`** | `configs/t40_transformer.yaml:41` |
| 8 | Label smoothing | 0.05 | 없음 | — |
| 9 | Class weight | inverse-sqrt + **REM ×2** | `count^-0.5`만 | — |

### 4.3 가장 큰 두 가지로 보는 이유

**(1) Frozen 2단계 병목.** choihy의 B0는 "고립된 30초 분류"에 유용한 특징만 학습하고 동결된다.
temporal 모델은 그 특징을 재조합할 수만 있다. byoungjun의 encoder는 시퀀스 목적함수에서 gradient를 받으므로
**문맥 속에서만 의미 있는 특징**도 학습할 수 있다. temporal 추가 이득이 갈리는 것과 일관된다.

| 트랙 | epoch 단독 | + temporal | 이득 |
| --- | ---: | ---: | ---: |
| choihy (frozen) | 0.4127 | 0.4833 | **+0.0706** |
| byoungjun (end-to-end) | 0.390 | 0.517 | **+0.127** |

**(2) Augmentation 부재로 인한 과적합.** choihy 전 모델이 심하게 과적합 중이다.

| 모델 | Best epoch | Train loss 첫→끝 | **Validation loss 첫→끝** |
| --- | ---: | --- | --- |
| C1 Conformer | **2** (12에서 종료) | 0.930 → 0.361 | 1.460 → **2.509** |
| B1 (새 embedding) | 24 | 0.620 → 0.265 | 1.195 → **1.735** |
| T40 | 6 | 0.629 → 0.321 | 1.511 → **2.041** |
| T80 | 17 | 0.644 → 0.223 | 1.504 → **2.511** |

Conformer가 **2번째 epoch에서 best**에 도달한 것은 정규화 수단이 전무하다는 신호다.

보조로, **night_norm과 시각 특징은 choihy의 약한 클래스를 직접 겨냥한다.**
전역 정규화는 235개의 서로 다른 방·마이크 배치·환자 거리를 잡음 변량으로 남긴다.
시각 특징은 N3가 전반부, REM이 후반부에 몰리는 거시구조 prior를 거의 공짜로 주입한다.

---

## 5. 트랙 위치 — 지금은 teacher 학습 단계가 아니다

### 5.1 형식과 실질의 괴리

형식상 choihy 트랙은 Stage 1 teacher lane에 속한다(B1은 미래 epoch을 쓰는 비인과 모델, 모바일 제약 없음).
그러나 teacher로 기능하지 않는다.

| 근거 | 내용 |
| --- | --- |
| 성능 역전 | choihy 0.4833 < byoungjun **student** 0.523 (807k params, causal). teacher가 student보다 약하면 teacher가 아니다 |
| 규모 | Conformer 1,905,124 params ≈ fp32 7.6 MB → **이미 50 MB 온디바이스 예산 안**. 앙상블·foundation·full-night·multi-view 등 teacher 권한 미사용 |
| 실제 KD teacher | `bj-e001`(PSG 신호 입력) macro F1 **0.672**, κ 0.561, REM F1 0.58 — 이미 존재하고 KD 타깃으로 사용 중 |

### 5.2 teacher lane의 헤드룸은 이미 측정됐다

| 실험 | 설정 | 결과 |
| --- | --- | ---: |
| `bj-e022` ub | 양방향 + 2배 폭 (비인과 상한) | 0.518 ≈ combo 0.506 |
| `bj-e037` 시퀀스 탐침 | full-night GRU, 3 seed | causal 0.517 ± 0.006 / **bidirectional 0.517 ± 0.002** |
| `bj-e037` 참조 | epoch 단독 | 0.390 |

원문 결론: **"시간 문맥은 포화. 병목은 epoch 표현·데이터."**
teacher lane을 정의하는 "비인과 + 더 큰 모델" 권한을 이미 행사했고 수익이 ~0이었다.
현재 라벨에서 오디오 모델 상한은 **0.52–0.55**로 추정돼 있다.

### 5.3 제품 정의 변경 (2026-09-15)

byoungjun `STATUS.md`에 업체 확정 사항이 기록돼 있다.

> **아침에 밤 전체 그래프만 필요. 실시간 불필요 → 비인과 허용, 서버 후보정 불필요(단말 내 처리).**

[PSG-only 연구 방향 handoff](<agent_handoff_psg_only_research_direction (1).md>)가 전제한
*"post-wake 비인과 teacher → causal edge student"* 구도가 사라졌다.
단말에서 비인과로 처리하므로 **별도의 큰 teacher를 둘 이유 자체가 줄었다.**
해당 handoff 문서의 §9~§10(long-context, caching 연구)은 `bj-e037`로 이미 답이 나왔다.

---

## 6. 권장 방향

### 6.1 우선 행동은 실험이 아니라 조율

choihy 트랙의 고유 자산은 **Conformer epoch encoder 하나**다.
byoungjun의 진단이 "병목은 epoch 표현"인데 정작 그쪽 encoder는 평범한 Conv2d CNN이다
([byoungjun/src/sleep_kd/audio_temporal.py:14](../../byoungjun/src/sleep_kd/audio_temporal.py#L14) `CompactAudioEncoder`).
**choihy가 동일 split에서 CNN → Conformer +0.049를 실측한 유일한 트랙이다.**

> **권고: choihy의 기여를 "독립 teacher"가 아니라 "epoch encoder 모듈"로 재정의하고,
> byoungjun 파이프라인에 이식하는 실험으로 전환한다.**

### 6.2 교환해야 할 것

| 요소 | 효과 | choihy 보유 |
| --- | --- | :---: |
| combo (night_norm + SpecAugment + 시각특징 + RF 127) | +0.054 (단일 최대) | ✗ |
| wave home augmentation | 복합 가정조건 0.344 → **0.481** | ✗ |
| gated KD + SK 사전학습 | +0.017 (3 seed 일관) | ✗ |
| 5-fold CV 평가 체계 (`manifests/cv5/`) | 고정 val 낙관 편향 제거 | ✗ |
| **Conformer epoch encoder** | **+0.049** | **✓ (유일 보유)** |

### 6.3 조율 후 실험 우선순위

**T1. Conformer encoder 이식 (단일 변수).** byoungjun combo 파이프라인의 `CompactAudioEncoder`를
Conformer로 교체하고 **end-to-end로** 학습한다. frozen 2단계를 버리는 것이 핵심이다.
동일 5-fold CV에서 combo parent와 paired 비교한다. 이것이 choihy 트랙의 기여를 검증하는 유일한 실험이다.

**T2. 과적합 대책 단독 검증.** T1과 섞지 말고, 현재 choihy 라인에 SpecAugment / cosine+warmup /
label smoothing을 넣어 C1의 best-epoch-2 문제가 해소되는지 확인한다. Conformer의 실제 용량을 알 수 있다.

**T3. Foundation encoder (AudioMAE / OPERA).** 전략 §5.3에서 **P0**이나 **양 트랙 모두 미시도**다.
encoder가 병목이라는 진단이 양쪽에서 일치하므로, pretrained encoder는 가장 강한 미탐색 레버다.
비용: 현재 front-end가 8 kHz / 48-mel / f_max 3900 Hz이므로 16 kHz 별도 cache가 필요하다.

**T4. Tracheal 채널 upper-bound 진단.** EDF 채널 19가 `Tracheal` 48 kHz인데 현재 채널 18 `Mic`만 쓴다.

| index | label | sample rate |
| ---: | --- | ---: |
| 10 | `Snore` | 500 Hz |
| 18 | `Mic` (ambient) | 48,000 Hz |
| 19 | `Tracheal` | 48,000 Hz |

접촉 마이크라 배포에는 못 쓰지만 **REM 한계가 "음향 접근성"인지 "라벨"인지 가르는 진단**이 된다.
tracheal 모델이 REM F1 0.5를 내면 cross-modal distillation이 실제 카드가 되고,
못 내면 §3의 결론을 독립적으로 확정한다. 동일 EDF 내 채널이라 정렬 비용이 없다.
단, split 규약상 **같은 night의 모든 채널은 같은 split에 유지**해야 한다(전략 §4.2).

### 6.4 재실행하지 말 것

| 항목 | 이유 |
| --- | --- |
| Long-context / full-night 확장 | `bj-e037`이 3 seed로 포화 확인 (causal 0.517 = bidirectional 0.517) |
| 비인과 대형 모델 상한 탐색 | `bj-e022` ub 0.518 ≈ combo 0.506 |
| HMM / Viterbi 후처리 | `bj-e006` 폐기 — Light 82% prior가 소수 클래스를 덮음 |
| 게이트 없는 logit KD | `bj-e023` 폐기 — REM −0.10 |
| 기본 loss/imbalance 스윕 | REM class weight ×2, label smoothing 0.05, inverse-sqrt 이미 수행됨 |

> **초판 정정:** 초판 §4의 Tier 0 (c) "loss/imbalance 스윕"은 상당 부분 이미 수행된 항목이었다.
> Tier 1 (d)(e)의 방향(temporal → representation)은 유효하나, 초판은 이를 새 발견처럼 제시했다.
> 실제로는 `bj-e037`이 먼저 3 seed로 확립했으며 본 트랙은 독립 재확인에 해당한다.

---

## 7. 일정 및 범위 리스크

전략 문서 §8 기준 **Stage 1은 9월 말 teacher freeze, 10월 초 Stage 2 시작**, 최종 납기 11월 27일이다.

| Stage 1 exit gate 요건 (§5.4) | 현재 상태 (팀 전체) |
| --- | --- |
| Macro-F1 ≥ 0.60 / κ ≥ 0.50 / Acc ≥ 0.60 | **0.523 / 0.390 / 0.793** (byoungjun 최고, 3 seed) — Acc만 달성 |
| 최소 3 seed 재현 | byoungjun 충족 / **choihy 미충족(전 실험 단일 seed)** |
| 주요 부정 결과 문서화 | byoungjun 충족 / choihy 부분 충족(§2.3) |
| Robustness / noise / device 검증 | byoungjun 수행 중(`bj-e048`, homeaug) / **choihy 전무** |
| Probability calibration | **양 트랙 전무** |
| Foundation model 비교 (P0) | **양 트랙 전무** |
| 평가 harness / split freeze | 충족 — `psg_audio_subject_split_v1.csv` (sha256 고정), test_locked 49명 미사용 |

### 7.1 강건성 — 검사실 성능은 실사용 추정치가 아니다

`bj-e048`이 측정한 기존 최고 모델(skgkd2step)의 교란 하 성능:

| 조건 | clean | SNR 10 dB | 기기 EQ | 복합(가정) |
| --- | ---: | ---: | ---: | ---: |
| skgkd2step (mel) | 0.544 | 0.355 | 0.422 | **0.301** |
| homeaug sev1.0 (3 seed) | 0.517 | **0.479** | **0.498** | **0.481** |

**"검사실 조건에서만 성립하는 모델"**이라는 것이 그쪽 결론이다.
choihy 트랙은 augmentation이 전무하므로 강건성 측정조차 되어 있지 않다.
주의: 평가 교란이 학습 augmentation과 같은 합성 계열이라 in-distribution 강건성이다.

### 7.2 타깃 도메인 평가셋 부재

계약 지표는 **"held-out real-world smartphone evaluation set"**에서 측정하도록 되어 있다(전략 §2.2).
그러나 `/home/sleep/data`에는 `psg_audio`만 있고, Multimodal OSA는 데이터 README에 `NOT ASSESSED`다.
byoungjun 쪽도 "실제 스마트폰 파일럿 녹음 필요(`bj-e049`)", "실제 생활 소음 데이터셋 상업 라이선스 검토 전"으로 기록했다.
**이 평가셋 없이는 Stage 1 exit gate가 원리적으로 닫히지 않는다.** 모델링과 병렬 트랙으로 즉시 착수해야 한다.

### 7.3 목표 지표 재협의

byoungjun `STATUS.md` next experiments에 **"bj-e033 과제 측과 3-stage 주 지표 협의"**가 이미 올라 있다.
choihy 트랙의 §3 감사 결과는 이 협의를 뒷받침하는 독립 근거다. 별도 제안 대신 해당 협의에 합류하는 것이 맞다.

---

## 8. 요약

> **(1) 지금은 teacher 학습 단계가 아니다.** teacher lane의 상한은 이미 측정됐고(0.52–0.55),
> 실제 KD teacher는 `bj-e001`로 별도 존재하며, 제품 정의 변경으로 teacher/student 구분 자체가 약해졌다.
> **(2) choihy 트랙의 고유 자산은 Conformer epoch encoder다.** 부가 요소 없는 지점끼리는 오히려 앞선다
> (0.4833 vs 0.452). 이를 byoungjun 파이프라인에 **end-to-end로 이식**하는 것이 다음 실험이다.
> **(3) 나머지는 조율 문제다.** combo·augmentation·KD·5-fold CV를 가져오고,
> long-context·후처리·기본 loss 스윕은 재실행하지 않는다.

---

## 9. 감사 재현 방법

§3의 원본 annotation 감사는 다음으로 재현한다. 학습·GPU 불필요.

```bash
cd /home/sleep/researchers/choihy
python3 - <<'PY'
import xml.etree.ElementTree as ET, glob, collections, statistics as s
ns = '{http://www.respironics.com/PatientStudy.xsd}'

def dist(stage_node):
    tr = sorted((float(x.attrib['Start']), x.attrib['Type'])
                for x in stage_node.findall(f'{ns}Stage'))
    c = collections.Counter()
    for i, (t, ty) in enumerate(tr):
        end = tr[i + 1][0] if i + 1 < len(tr) else t + 30
        c[ty] += end - t
    return c

agg, rem_frac, durations, zero_rem = collections.Counter(), [], [], 0
for f in sorted(glob.glob('/home/sleep/data/psg_audio/V3/APNEA_RML/*.rml')):
    st = ET.parse(f).getroot().find(
        f'.//{ns}ScoringData/{ns}StagingData/{ns}UserStaging/{ns}NeuroAdultAASMStaging')
    if st is None:
        continue
    c = dist(st)
    tot = sum(c.values())
    if not tot:
        continue
    agg.update(c); durations.append(tot / 3600); rem_frac.append(c.get('REM', 0) / tot)
    zero_rem += c.get('REM', 0) == 0

tot = sum(agg.values())
print('subjects        :', len(rem_frac))
print('stage pct       :', {k: '%.2f%%' % (100 * v / tot) for k, v in sorted(agg.items())})
print('zero-REM        : %d (%.1f%%)' % (zero_rem, 100 * zero_rem / len(rem_frac)))
print('REM frac median : %.3f' % s.median(rem_frac))
print('duration median : %.2f h' % s.median(durations))
PY
```

EDF 채널 구성(§6.3 T4)은 `.venv-raw/bin/python`에서 `pyedflib.EdfReader(...).getSignalLabels()`로 확인한다.

### 근거 파일

**choihy 트랙**

- [첫 B0/B1 실험 보고서](PSG_EXPERIMENT_REPORT_20260914.md)
- [B1 80→20 결과](PSG_B1_80TO20_REPORT_20260914.md)
- [Transformer 결과](PSG_TRANSFORMER_REPORT_20260915.md)
- [Conformer 결과](PSG_CONFORMER_REPORT_20260916.md)
- [PSG-only 연구 방향 handoff](<agent_handoff_psg_only_research_direction (1).md>) — §5.3 참조, 전제 일부 무효
- 미기록 실행: `artifacts/experiments/b0_lr1e4_20260914T154714/comparison.json`

**byoungjun 트랙** (경로: `/home/sleep/researchers/byoungjun/`)

- `STATUS.md` — 제품 정의 변경, `bj-e037` 시퀀스 탐침, `bj-e048` 강건성, tier-5, 5-fold CV
- `RESULTS_SUMMARY.md` — 방법별 기여도, 무효·유해 판정 목록, 데이터 제약
- `KD_DESIGN.md` — KD 설계
- `reports/label_audit/` — 라벨 감사, `subject_covariates_20260915.csv`
- `src/sleep_kd/audio_temporal.py` — `CompactAudioEncoder`, 시각 특징
- `src/sleep_kd/audio_sequence_data.py` — `night_norm`

**공통**

- [모델 개발 전략](MODEL_DEVELOPMENT_STRATEGY.md) — §2.2 계약 지표, §5.3 우선순위, §5.4 exit gate, §7.2 승인 규칙, §8 일정
- 고정 split: `/home/sleep/data/splits/psg_audio_subject_split_v1.csv`
