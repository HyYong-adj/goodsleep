# Phase 2 — encoder 단계 combo 주력 요소 이식 (night_norm, SpecAugment)

**작성일:** 2026-09-16
**상태:** 실행 중 (`artifacts/experiments/phase2_20260916`, GPU 2)
**Parent:** `artifacts/experiments/conformer_epoch_20260915T054210274368058` (C1 0.412662 / B1 0.483261)
**관련:** [Phase 1 결과](../PSG_PHASE1_REPORT_20260916.md) — 음성, combo 미검증

## 목적

[Phase 1](PHASE1_EMBEDDING_OPTIONS.md)은 참조 트랙 combo(+0.054)를 시험하지 않았다.
겹치는 요소가 시각 특징 하나뿐이었고, **주력인 night_norm과 SpecAugment는 encoder 재학습이 필요해 미뤘다.**
Phase 2가 실제 combo 검증이다.

B1 설정은 parent 그대로 고정하여 **encoder 단계 변경만** 측정한다.

## 변형

| ID | 변경 | 근거 |
| --- | --- | --- |
| **C0** | 없음 (대조군, 동일 GPU) | GPU 교란 제거 — 아래 §무결성 참조 |
| C2 | night_norm (피험자별 mean/std) | 마이크 감도·방 소음 차이 제거 |
| C3 | SpecAugment (freq 8 / time 150 / gain σ0.25) | 과적합 대책 |
| **C4** | **C2 + C3** | **채택 판정 대상** |

Seed 20260910 단일 screening. 유망하면 3 seed로 확장한다.
런당 약 70분(C1 62분 + embedding + B1), 4런 약 4.7시간.

### night_norm이 제거하는 변량

전역 통계 1쌍(mean −40.88 / std 11.81)으로 235개 녹음을 정규화하고 있었다. 실측 분포:

| | 최소 | 중앙값 | 최대 | 표준편차 |
| --- | ---: | ---: | ---: | ---: |
| 피험자별 mean | **−68.39** | −40.83 | **−26.77** | 5.19 |
| 피험자별 std | 6.14 | 10.90 | 14.52 | — |

평균이 **41.6 dB 범위**로 퍼져 있다. mean −68.4인 녹음은 전역 통계로 정규화하면
전체가 약 2.3σ 이동한 채 encoder에 들어간다. 이를 흡수하는 데 용량이 쓰이고 있었을 가능성이 있다.

`night_stats.json`이 소스 캐시 235/235에 이미 존재하며, 3개 피험자에 대해 직접 재계산해
일치를 확인했다. **소스 캐시(타 연구자 소유)에는 쓰지 않는다.**

### 배포상 단서

night_norm은 **밤 전체**를 요구하므로 실시간 인과 경로에서는 쓸 수 없다.
2026-09-15 확정된 제품 정의(아침에 밤 전체 그래프, 비인과 허용, 단말 내 처리)에서는 허용된다.
라벨을 쓰지 않는 입력 신호만의 정규화이므로 라벨 누수는 아니다.

## 무결성 — 왜 C0 대조군이 필요한가

`cache_embeddings.py`를 수정했으므로 parent C1 체크포인트로 embedding을 재생성해 대조했다.
그 과정에서 **비트 단위 재현이 GPU 간에 성립하지 않음**을 확인했다.

| 조건 | 결과 |
| --- | --- |
| 같은 GPU(2) 두 번 실행 | 해시 **완전 일치** |
| parent(GPU 3) vs 현재(GPU 2) | 요소 약 **0.05%** 차이, 최대 절대차 0.00195 (임베딩 범위 ±6.2) |

따라서 게이트 기준을 두 번 완화했고 근거를 남긴다.

1. 비트 동일 → **불가능**(GPU 간 커널 선택 차이). 같은 GPU 재실행 해시 일치로 코드 무결성은 별도 확인.
2. 1 ULP → **2 ULP**. 한 요소가 1 ULP를 넘었으나 값이 −6.05e-05로 float16 subnormal 경계였다.
   임베딩 범위 ±6에서 이 크기는 의미가 없다. 2 ULP 초과는 여전히 실패로 본다.

**결과: 통과** (6명, 2 ULP 초과 0건).

이 발견 때문에 **C0 대조군을 추가했다.** parent는 GPU 3에서 학습됐으므로, C0 없이
C2/C3/C4를 parent와 비교하면 모든 Δ가 GPU 효과와 교란된다. 단일 seed screening에서는
GPU 교란이 seed 잡음과 구별되지 않는다.

> 부수 함의: 기존 보고서들이 기록한 checkpoint 해시는 **GPU 종속**이다.
> 다른 GPU에서의 재현 검증에 해시 동일성을 기준으로 쓰면 안 된다.

## 구현

| 위치 | 변경 |
| --- | --- |
| `src/psg_only/data.py` | `night_statistics()`, `CachedMelEpochDataset(night_norm=...)` |
| `src/psg_only/train.py` | `spec_augment()`, C1 학습 루프 적용(train split 한정), night_norm 배선 |
| `src/psg_only/evaluate.py` | **`evaluate_b0`가 night_norm 반영** — 누락 시 학습/평가 정규화 불일치 |
| `scripts/cache_embeddings.py` | night_norm 반영 + provenance에 `normalization` 기록 |
| `tests/test_phase2_options.py` | 신규 9건 |

정규화를 소비하는 세 지점(학습·평가·embedding 생성)이 모두 같은 설정을 따르도록 맞췄다.
provenance의 `normalization` 필드가 서로 다른 정규화의 캐시가 섞이는 것을 막는다.

테스트: 44 + 11(Phase 1) + 9(Phase 2) = **64건 통과**.

## 실행

```bash
cd /home/sleep/researchers/choihy
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  nohup /venv/main/bin/python -u scripts/run_phase2.py \
  --output artifacts/experiments/phase2_20260916 \
  > artifacts/logs/phase2_20260916.log 2>&1 < /dev/null &
```

## 판정 기준

주지표 Macro-F1, **C0 대비** 차이로 판단한다(parent 대비가 아니다).

- 채택: C4의 Δ ≥ +0.01, 그리고 REM·Deep F1이 하락하지 않을 것
- 단일 seed이므로 채택 시 3 seed 확장이 선행 조건이며, 이 단계에서 우위 확정은 하지 않는다
- C1 단독 성능과 downstream B1을 분리 보고한다 (참조: B0 LR 1e-4는 C1↑ / B1↓였다)

## 한계

- 단일 seed screening. validation-only, `test_locked` 미접근.
- 참조 트랙 combo에는 RF 127도 포함되나 여기서는 다루지 않는다(context 축은 종결됨).
- 4-stage Macro-F1 0.60 격차는 이 Phase로 좁혀지지 않는다. REM 데이터 제약은 그대로다.
