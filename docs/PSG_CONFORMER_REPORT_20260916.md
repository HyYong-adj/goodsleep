# Conformer epoch encoder 실험 결과 — 2026-09-16

> **2026-09-16 오후 후속 노트 (트랙 대조).** 아래 결과는 유효하나, 팀 내 병행 트랙
> (`researchers/byoungjun`)과 대조하면 해석이 달라지는 부분이 있다.
>
> - **§6의 후속 방향 1(추가 seed)은 여전히 유효하다.** 본 실험은 단일 seed다.
> - **§6이 "미실험"으로 남긴 Conformer + full-night는 재실행 가치가 낮다.** `bj-e037`이 3 seed로
>   full-night GRU를 측정해 causal 0.517 ± 0.006 / bidirectional 0.517 ± 0.002, epoch 단독 0.390을 얻었고,
>   "시간 문맥은 포화, 병목은 epoch 표현·데이터"로 결론냈다. 본 실험의 encoder 가설과 같은 방향이다.
> - **Macro-F1 0.483261은 절대값으로 신뢰할 수 없다.** byoungjun 트랙이 36 run에서 실측한
>   best-epoch 선택 부풀림은 **+0.049**이며, 고정 val-44는 5-fold CV 평균보다 **0.027** 낙관적이다.
>   seed 쌍 비교·paired 분석만 유효하다.
> - **비교 맥락:** byoungjun의 동급(부가 요소 없는) baseline은 0.452 ± 0.005, combo 적용 시 0.506 ± 0.019다.
>   본 실험의 0.483은 combo·KD·augmentation을 하나도 쓰지 않은 상태의 값이다.
>
> 종합 판단은 [다음 실험 방향 제언 rev.2](PSG_RESEARCH_DIRECTION_20260916.md) 참조.

## 1. 결론

**Conformer encoder + 기존 BiLSTM40은 이번 단일 seed validation에서 기존 CNN + BiLSTM40보다 개선됐다.**
최종 Macro-F1은 **0.433996 → 0.483261 (+0.049266, +4.93 percentage points)**이다.
Encoder 단독도 0.363094 → 0.412662로 개선됐다. 기존 Transformer context 교체와 달리
이번에는 downstream의 네 클래스 F1이 모두 상승했으며, 가장 큰 기여는 REM이다.

이는 **음향 encoder 교체가 유망하다는 screening 결과**이지 통계적 우위나 배포 승인은 아니다.
기존 CNN parent는 비교 기준으로 보존하고, Conformer + BiLSTM40은 재현성 확인 대상 후보로 기록한다.

## 2. 실행 상태와 비교 조건

- Run: `conformer_epoch_20260915T054210274368058`, `COMPLETE`, `smoke=false`.
- 실행: 2026-09-15 05:42:13–06:49:53 UTC, GPU 3, seed `20260910`.
- 보고서 확인일: 2026-09-16. 실행 날짜와 보고서 날짜를 구분한다.
- 전체 실행 시작→완료: 4,060.66초, 약 **67분 41초**. 학습뿐 아니라 평가·cache 생성 등도 포함한다.
- Parent: `20260914T062144930455Z`. 저장 prediction을 재집계한 비교이며 parent 재학습은 아니다.
- Train 191 / validation 44 subjects, 유효 epoch 108,543 / 23,621. Test 없음.
- 동일 manifest, label 순서, train-only 정규화·class weights, weighted CE를 유지했다.
- C1: CNN subsampling + 2-block Conformer(d192, heads4, FFN768, kernel31), mean pooling.
- C1 best checkpoint를 freeze하여 새 embedding을 만들고 기존 2-layer BiLSTM40→20을 학습했다.
- C1: LR 3e-4, batch16, 최대30 epochs. B1: LR 3e-4, batch32, 최대50 epochs. 양쪽 patience10.

확인 과정에서 parent B0/B1 및 candidate C1/B1의 저장 prediction을 현재 evaluator로 재집계했다.
네 결과 모두 같은 44 subjects·23,621 유효 epoch의 `(subject, epoch, target)`과 일치하며,
Macro-F1·Accuracy·κ·temporal 지표·confusion matrix가 `comparison.json`과 일치했다.
새 학습이나 checkpoint 변경은 수행하지 않았다.

근거: [실행 상태](../artifacts/experiments/conformer_epoch_20260915T054210274368058/experiment.json),
[비교 지표](../artifacts/experiments/conformer_epoch_20260915T054210274368058/comparison.json),
[입력·비교 무결성](../artifacts/experiments/conformer_epoch_20260915T054210274368058/preflight.json),
[실행 설정](experiments/CONFORMER_EPOCH_EXPERIMENT.md).

## 3. 전체 및 클래스별 성능

| 모델 | Macro-F1 | Accuracy | κ | Wake F1 | REM F1 | Light F1 | Deep F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CNN 단독 (B0) | 0.3631 | 0.7476 | 0.2169 | 0.4634 | 0.0442 | 0.8539 | 0.0910 |
| Conformer 단독 (C1) | 0.4127 | 0.7304 | 0.2301 | 0.4828 | 0.1319 | 0.8391 | 0.1969 |
| CNN + BiLSTM40 | 0.4340 | 0.7670 | 0.3008 | 0.5630 | 0.0975 | 0.8636 | 0.2119 |
| Conformer + BiLSTM40 | **0.4833** | **0.7989** | **0.3565** | **0.5722** | **0.2322** | **0.8836** | **0.2451** |

- Encoder 단독 Macro-F1 차이: **+0.049568**. Accuracy는 −0.017188로 하락했다.
- Downstream Macro-F1 차이: **+0.049266**. Accuracy +0.031878, κ +0.055668.
- Conformer embedding에 BiLSTM40을 추가한 효과: Macro-F1 **+0.070600**.
- Validation의 Light 비중은 81.38%다. 항상 Light로 예측하면 Accuracy 0.8138이나
  Macro-F1은 0.2243이므로, Accuracy 단독으로 모델을 선택하지 않는다.

### Downstream에서 무엇이 바뀌었는가

| 클래스 | Precision 기존→Conformer | Recall 기존→Conformer | F1 변화 |
| --- | ---: | ---: | ---: |
| Wake | 0.4984 → 0.5318 | 0.6468 → 0.6192 | +0.0092 |
| REM | 0.0830 → 0.2120 | 0.1180 → 0.2565 | **+0.1347** |
| Light | 0.8694 → 0.8746 | 0.8579 → 0.8927 | +0.0200 |
| Deep | 0.2984 → 0.4150 | 0.1643 → 0.1739 | +0.0332 |

REM F1 상승은 전체 Macro-F1 증가분의 약 **68.3%**를 설명한다(클래스 F1 차이의 산술 기여).
REM 정답 881개 중 정분류가 **104 → 226개**로 늘고 Light→REM 오분류는 1,064 → 740개로 줄었다.
다만 REM의 650/881개(73.78%)는 여전히 Light로 분류된다.

Deep 정분류는 **256 → 271개**로 증가폭이 작다. Light→Deep 오분류가 595 → 360개로 줄어
precision 개선이 두드러지지만, Deep recall은 **17.39%**에 머문다.
실제 Deep의 1,091/1,558개(70.03%)를 Light로 예측하므로 Deep 문제가 해결됐다고 볼 수 없다.
Wake도 F1은 오르지만 recall은 64.68% → 61.92%로 낮아지는 trade-off가 있다.

근거: [C1 평가](../artifacts/experiments/conformer_epoch_20260915T054210274368058/c1/evaluation/results.json),
[B1 평가](../artifacts/experiments/conformer_epoch_20260915T054210274368058/b1/evaluation/results.json).

## 4. Temporal 성능

| 모델 | Transition Macro-F1 | Stable Macro-F1 | 예측 전환율 | Subject별 전환율 절대오차 평균 |
| --- | ---: | ---: | ---: | ---: |
| CNN 단독 | 0.3787 | 0.3514 | 14.72% | 9.73 pp |
| Conformer 단독 | 0.4249 | 0.3989 | 19.72% | 14.58 pp |
| CNN + BiLSTM40 | 0.3791 | 0.4313 | 5.80% | 1.79 pp |
| Conformer + BiLSTM40 | **0.4254** | **0.4780** | **5.49%** | **1.61 pp** |

정답 전환율은 subject 평균 **5.26%**다. Conformer 단독은 더 자주 예측이 바뀌지만,
BiLSTM40을 거치면 전환율이 정답에 가까워지고 transition/stable 단계 분류 모두 개선된다.
전환율은 인접한 유효 epoch pair에서 구한 subject별 비율의 평균이며 결측을 가로지르지 않는다.
절대오차 평균은 subject별 오차를 평균한 값이므로 두 평균 전환율의 단순 차이와 다르다.
Transition Macro-F1은 **정답이 전환된 위치의 4-class 단계 분류 F1**이지 전환 검출 binary F1이 아니다.

## 5. 학습 이력과 비용

| 항목 | C1 | 새 embedding의 BiLSTM40 |
| --- | ---: | ---: |
| Best epoch (1-based) | 2 | 24 |
| 종료 epoch | 12 | 34 |
| Best optimizer steps | 13,870 | 4,248 |
| 종료 optimizer steps | 83,220 | 6,018 |
| Train loss 첫→마지막 | 0.930 → 0.361 | 0.620 → 0.265 |
| Validation loss 첫→마지막 | 1.460 → 2.509 | 1.195 → 1.735 |

양쪽 모두 best 이후 10 epochs 개선이 없어 early stopping했다.
C1은 특히 2번째 epoch에서 best에 도달한 뒤 train loss는 줄고 validation loss는 커져
과적합과 일치하는 양상을 보인다. 이 관찰만으로 LR이나 용량이 원인이라고 확정하지 않는다.
평가와 cache는 마지막 모델이 아니라 validation Macro-F1 기준 best checkpoint를 사용했다.

C1 trainable parameters는 **1,905,124**, 학습+validation은 **3,712.20초(61.87분)**,
peak allocated GPU memory는 **349.79 MiB**다. 이 값은 GPU 전체 점유량이나 reserved memory가 아니다.
B1 개별 학습 시간·peak memory는 해당 run metadata에 없어 미측정으로 남긴다.
전체 67분 41초에서 C1 시간을 뺀 값을 B1 학습 시간으로 해석하지 않는다.

근거: [C1 이력](../artifacts/experiments/conformer_epoch_20260915T054210274368058/c1/history.jsonl),
[B1 이력](../artifacts/experiments/conformer_epoch_20260915T054210274368058/b1/history.jsonl),
[C1 자원](../artifacts/experiments/conformer_epoch_20260915T054210274368058/c1/run_metadata.json),
[B1 metadata](../artifacts/experiments/conformer_epoch_20260915T054210274368058/b1/run_metadata.json).

## 6. 기존 실험과 연결한 판단

기존 CNN embedding의 Transformer40/80은 Macro-F1 **0.4069/0.4086**으로
CNN + BiLSTM40(0.4340)을 넘지 못했다. 이번에는 temporal 모델을 BiLSTM40으로 유지한 채
encoder를 교체해 0.4833에 도달했다. 따라서 현재 관찰은 “BiLSTM이 구형이라 성능이 낮다”보다는
“30초 내부 음향 표현의 개선이 도움이 된다”는 가설에 더 부합한다.

다만 교체에는 subsampling·모델 용량·연산 구조 차이도 포함되므로 Conformer attention만의
인과적 효과를 분리한 결과는 아니다. Conformer embedding + Transformer 또는 full-night는 미실험이다.
본 결과는 비인과적/post-wake PSG-audio validation이며 실시간 edge나 home smartphone 성능을 입증하지 않는다.

후속 검증 방향(이번 작업에서는 설정 변경·실행하지 않음):

1. 같은 split과 고정 설정에서 CNN parent와 Conformer 후보를 추가 seed로 paired 재실험하여
   encoder부터 downstream까지의 재현성을 확인한다. 유의성·신뢰구간은 아직 계산하지 않았다.
2. REM/Deep→Light 오류를 recording 단위로 점검한다. Epoch를 독립 표본으로 취급한 검정보다는
   subject/night 단위 paired 분석을 사용하며, 특정 recording에만 개선이 집중되는지 확인한다.
3. Conformer의 이른 과적합을 확인한 뒤 LR 3e-4→1e-4 같은 단일 변수 실험을 별도로 검토한다.
   Encoder 튜닝과 context/temporal 구조 변경을 한 실험에 섞지 않는다.

Validation으로 checkpoint와 후속 실험을 선택하므로 이 수치는 개발 결과다.
독립 test 평가, 다중 seed 결과, 통계적 우위, 공식 후보 승인은 아직 없다.

관련 기록: [Transformer 결과](PSG_TRANSFORMER_REPORT_20260915.md),
[실험 문서 목록](experiments/README.md).
