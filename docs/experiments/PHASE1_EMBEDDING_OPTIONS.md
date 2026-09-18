# Phase 1 — 동결 Conformer embedding 위 부가 요소 이식

**작성일:** 2026-09-16
**상태:** 실행 중
**Parent:** `artifacts/experiments/conformer_epoch_20260915T054210274368058` (Conformer + BiLSTM40, Macro-F1 0.483261)
**작업 위치:** `/home/sleep/researchers/choihy`
**관련 문서:** [다음 실험 방향 제언 rev.2](../PSG_RESEARCH_DIRECTION_20260916.md) §6.3

## 목적

병행 트랙(`researchers/byoungjun`)이 확보한 부가 요소 중 **동결 embedding 위에서 검증 가능한 네 가지**를
choihy 파이프라인에 이식하고 단일 변수로 측정한다. Encoder 재학습이 필요한 요소(night_norm, SpecAugment)와
구조 변경(end-to-end)은 Phase 2/3로 분리한다.

Parent의 Conformer embedding 캐시(235명)를 그대로 재사용하므로 런당 약 100초다.

## 변형

| ID | 변경 | 근거 |
| --- | --- | --- |
| A0 | 없음 (parent 재현) | 무결성 게이트 |
| A1 | 시각 특징 `[h, h²]` → embedding 가산 | N3 전반부 / REM 후반부 거시구조 prior |
| A2 | cosine LR + warmup 3 | 현재 `scheduler: none` |
| A3 | label smoothing 0.05 | 경계 epoch 라벨 불확실성 |
| A4 | REM class weight ×2 | 현재 `count^-0.5`만 |
| **A5** | **A1–A4 결합** | **채택 판정 대상** |

Seeds: `20260910, 20260911, 20260912` (전략 §7.2의 3 seed 요건).
총 6 × 3 = 18런.

### ⚠️ 단독 런을 폐기 근거로 쓰지 않는다

참조 트랙 기록: *"night_norm / SpecAugment / 시각 특징 / RF 127 **단독**: 각각 −0.01 ~ +0.01 → 단독 무효,
combo로만 사용"*. 네 요소는 결합해야 +0.054였다.

따라서 A1–A4 단독 런은 **귀속(attribution)용**이고, 채택 판정은 **A5**로 한다.
단일 변수 원칙(전략 §7.2)과 이 경험적 사실이 충돌하므로, 양쪽을 모두 돌려 해소한다.

## 무결성 게이트

A0 seed 20260910은 parent Macro-F1 `0.48326135281572924`를 **정확히** 재현해야 한다.
재현되지 않으면 코드 변경이 학습 경로를 바꿨다는 뜻이므로 러너가 `FAILED_PARENT_REPRODUCTION`으로 중단한다.

**결과: 통과.** 재현값 `0.48326135281572924` (차이 0), best epoch 23(0-indexed), 종료 34 epoch —
parent 기록(best epoch 24 1-based, 종료 34)과 일치.

## 구현

옵션은 전부 **기본값 off**로 추가해 기존 체크포인트가 그대로 로드된다.

| 위치 | 변경 |
| --- | --- |
| `src/psg_only/models.py` | `B1Model(time_feature=...)`, `window_hours()` |
| `src/psg_only/train.py` | `class_weights(multipliers=...)`, `build_scheduler()`, label smoothing, hours 전달, B1 비용 metadata |
| `src/psg_only/evaluate.py` | `time_feature` 체크포인트 복원, 평가 시 hours 전달 |
| `tests/test_phase1_options.py` | 신규 11건 |

`window_hours`는 `target_indexes[:, 0]`(실제 30초 격자의 첫 target epoch)에서 입력 위치를 역산한다.
패딩 위치는 0으로 둔다 — epoch 0 자체도 0.0이라 값이 겹치지만, 모델은 `input_valid`로 패딩을 따로 구분한다
(embedding이 0으로 마스킹된다). 참조 구현과 같은 동작이다.

테스트: 기존 44건 + 신규 11건 = **55건 통과**.

## 실행

```bash
cd /home/sleep/researchers/choihy
mkdir -p artifacts/logs
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  nohup /venv/main/bin/python -u scripts/run_phase1.py \
  --output artifacts/experiments/phase1_20260916 \
  > artifacts/logs/phase1_20260916.log 2>&1 &
```

GPU: byoungjun 트랙이 GPU 3에서 5-fold CV 체인을 돌리고 있어 **GPU 2**를 사용한다.
(참조 트랙 메모: 다중 GPU는 NUMA 노드 0의 0/1/2 권장, GPU 3 횡단 시 2.4–3배 느림.)

집계:

```bash
/venv/main/bin/python scripts/analyze_phase1.py artifacts/experiments/phase1_20260916
```

## 판정 기준

주지표는 Macro-F1이며 **같은 seed 끼리의 차이(paired)**만 근거로 쓴다.

- 채택: A5의 3 seed paired Δ가 전부 양수이고 평균 Δ ≥ +0.01
- 보류: 부호가 갈리거나 평균 Δ < +0.01
- REM/Deep F1이 하락하면 Macro-F1이 올라도 채택하지 않는다 (참조 트랙의 KD 실패 사례와 동일 기준)

절대값은 보고하되 판단 근거로 쓰지 않는다. 참조 트랙 실측상 best-epoch 선택 부풀림은 **+0.049**,
고정 val-44는 5-fold CV 평균보다 **0.027** 낙관적이다. 러너가 런마다 인접 epoch 평균을 함께 기록한다.

## 한계

- validation-only. `test_locked` 49명은 미접근.
- 동결 embedding 위 결과이므로 **end-to-end(Phase 3) 전환 후 그대로 전이된다는 보장이 없다.**
- 참조 트랙 수치는 CNN encoder 위에서 측정된 것이라 Conformer 위에서 재현될지는 별개 문제다.
- 4-stage Macro-F1 0.60은 이 Phase로 도달하지 못한다. REM 데이터 제약은 그대로다.
