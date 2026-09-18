# Transformer context 실험 결과와 후속 계획 — 2026-09-15

> 2026-09-16 후속 확인: Conformer encoder + 기존 BiLSTM40 정규 실험이 완료됐으며
> Macro-F1 0.483261을 기록했다. [Conformer 결과 보고서](PSG_CONFORMER_REPORT_20260916.md) 참조.
> 아래 parent 유지 판단과 다음 실험 계획은 Transformer 비교 당시의 기록이다.

## 1. 결론

**이번 설정에서는 Transformer가 기존 BiLSTM을 대체할 만큼 개선되지 않았다.**
현재 parent는 CNN embedding + BiLSTM 40→20으로 유지한다.
Transformer는 REM과 정답 전환 위치의 단계 분류를 개선했으나 Wake/Deep에서 손실이 있었다.

실험: `transformer_pair_20260914T164821314552775`, COMPLETE, smoke=false,
GPU3, seed20260910, validation-only.
2026-09-14 16:48:24–16:51:18 UTC에 실행됐다.
같은 frozen B0 embedding과 train191/val44 subjects, valid epochs108,543/23,621을 사용했다.

근거: [실행 상태](../artifacts/experiments/transformer_pair_20260914T164821314552775/experiment.json),
[비교 지표](../artifacts/experiments/transformer_pair_20260914T164821314552775/comparison.json).

## 2. 전체·class별 결과

| 모델 | Macro-F1 | Accuracy | κ | Wake F1 | REM F1 | Light F1 | Deep F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BiLSTM40 | **0.4340** | 0.7670 | 0.3008 | 0.5630 | 0.0975 | 0.8636 | **0.2119** |
| BiLSTM80 | 0.4319 | 0.7719 | **0.3023** | **0.5704** | 0.0961 | 0.8670 | 0.1940 |
| Transformer40 | 0.4069 | 0.7464 | 0.2461 | 0.5166 | 0.1223 | 0.8489 | 0.1398 |
| Transformer80 | 0.4086 | **0.7742** | 0.2578 | 0.5001 | **0.1618** | **0.8691** | 0.1035 |

Macro-F1 차이:

- T40−R40: **−0.027086**.
- T80−R80: **−0.023219**.
- Transformer80−40: **+0.001732**.
- BiLSTM80−40: **−0.002135**.
- Architecture×context interaction: **+0.003866**, 단일 seed여서 유의한 상호작용으로 확정하지 않는다.

Transformer80은 Deep recall이6.55%다. 실제 Deep1,558개 중102개만 맞혔고,
1,297개를 Light로 분류했다. REM F1 상승이 Wake/Deep 손실을 상쇄하지 못했다.
Accuracy만 기준으로 Transformer80을 선택하지 않는다.

## 3. Temporal 결과의 해석

| 모델 | Transition Macro-F1 | Stable Macro-F1 | 예측 전환율 |
| --- | ---: | ---: | ---: |
| BiLSTM40 | 0.3791 | 0.4313 | 5.80% |
| BiLSTM80 | 0.3717 | 0.4307 | 4.91% |
| Transformer40 | 0.4075 | 0.3958 | 10.94% |
| Transformer80 | 0.4102 | 0.3958 | 9.52% |

정답 전환율은5.26%다. Transformer의 예측은 실제보다 자주 바뀐다.
Transition Macro-F1은 정답이 직전 epoch와 달라진 위치에서의 **4-class 단계 분류 F1**이며
전환 검출 자체의 binary F1이 아니다. 이 지표 상승만으로 전체 수면 시계열이 좋아졌다고 해석하지 않는다.
결측을 가로지르는 pair는 평가에서 제외한다.

## 4. 학습 이력과 비용

| 항목 | T40 | T80 |
| --- | ---: | ---: |
| Best epoch (1-based) | 6 | 17 |
| 종료 epoch | 16 | 27 |
| Best optimizer steps | 1,062 | 3,009 |
| 종료 optimizer steps | 2,832 | 4,779 |
| Train loss 첫→마지막 | 0.629→0.321 | 0.644→0.223 |
| Validation loss 첫→마지막 | 1.511→2.041 | 1.504→2.511 |
| 모델 parameters | 694,148 | 694,148 |
| 학습+validation 시간 | 62.68초 | 98.01초 |
| Peak allocated GPU memory | 약105.9MiB | 약142.8MiB |

Train loss는 감소하고 validation loss는 증가하는 양상으로 과적합과 일치한다.
Dropout과 모델 상태가 다르므로 두 loss의 절대 차이만으로 원인을 확정하지 않는다.
Validation CE와 Macro-F1은 다른 목적함수이며, best checkpoint는 계획대로 Macro-F1로 선택했다.

짧은 실행 시간은 encoder를 재학습하지 않고 cached embedding만 학습했기 때문이다.
Conformer를 포함하는 음향 encoder 학습의 비용으로 일반화하지 않는다.

근거: [T40 이력](../artifacts/experiments/transformer_pair_20260914T164821314552775/T40/history.jsonl),
[T80 이력](../artifacts/experiments/transformer_pair_20260914T164821314552775/T80/history.jsonl),
[T40 자원](../artifacts/experiments/transformer_pair_20260914T164821314552775/T40/run_metadata.json),
[T80 자원](../artifacts/experiments/transformer_pair_20260914T164821314552775/T80/run_metadata.json).

## 5. 가설별 판단과 한계

- “BiLSTM이 구형이라80→20 효과가 없었다”: 이번 비교로 지지되지 않음.
- “Transformer가 추가 문맥을 더 잘 활용한다”: 작은 양의 차이는 있으나 단일 seed 근거가 약함.
- “Frozen CNN 표현이 병목이다”: 가능한 가설이며 encoder를 바꿔 검증해야 함.
- “Full-night가 무효다”: 이번 실험에서 다루지 않았으므로 판단 불가.

동일 학습 설정을 적용한 screening이지 각 모델에 최적 튜닝을 적용한 비교가 아니다.
위치 정보·mask·모델 용량 차이를 포함한 모델 교체 효과이며 attention 연산만의 효과가 아니다.
독립 test나 home smartphone 환경에서의 일반화 결과가 아니다.

## 6. 다음 실험: Conformer epoch encoder → 같은 BiLSTM40

**주 질문:** 30초 내부 음향 표현을 바꾸면 REM/Deep과 최종 단계 분류가 개선되는가?

1. C1: 기존48-Mel 입력으로 CNN subsampling + Small Conformer를 single-epoch 학습.
2. C1 best checkpoint를 고정하고 별도 경로에192차원 embedding 생성.
3. 기존과 같은 BiLSTM40→20을 새 embedding으로 학습.
4. C1 vs 기존 B0(0.363094), 새 embedding의 B1 vs 기존 B1(0.433996)을 각각 비교.

Single-epoch 성능과 최종 temporal 성능을 구분한다.
C1 단독 성능이 오르지 않아도 downstream이 개선될 수 있으므로 첫 screening은 두 단계 모두 평가한다.
데이터·label·전처리·weighted CE와 각 단계의 기존 학습 budget을 고정한다.
Conformer 효과에는 CNN subsampling 구조 변화와 모델 용량 변화도 포함된다.

실제 설정과 실행 명령은 [Conformer 실험 문서](experiments/CONFORMER_EPOCH_EXPERIMENT.md)를 따른다.
한 seed에서 유망한 후보가 나올 경우 추가 seed로 재현성을 확인한다.
보조 후보인 T40 LR3e-4→1e-4는 별도 단일 변수 실험으로 남기며 이번 실행에 섞지 않는다.
