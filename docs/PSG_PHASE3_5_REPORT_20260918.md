# Phase 2–5 누적 결과 — 2026-09-18

**범위:** 2026-09-16 ~ 09-18, validation-only(44 subjects / 23,621 valid epochs), `test_locked` 49명 미접근.
**관련:** [Phase 1 결과](PSG_PHASE1_REPORT_20260916.md) · [Phase 2 자동 집계](PSG_PHASE2_REPORT_20260916.md) · [방향 제언 rev.2](PSG_RESEARCH_DIRECTION_20260916.md)

## 1. 결론

**최고 기록은 end-to-end 공동 학습의 0.5310이다.** 동결 2단계 최고(D3 0.5071 ± 0.0094, 3 seed)보다
원시 +0.024, 부풀림 보정 +0.020 높다. 다만 **단일 seed이며 재현 확인 중**이다.

확인된 이득은 세 가지다.

| 레버 | 이득 | 근거 |
| --- | ---: | --- |
| SpecAugment (encoder 단계) | +0.024 | C3 vs C0 대조군 |
| gated KD (PSG teacher) | +0.010 | D2−D0, D3−D1 |
| **end-to-end 공동 학습** | **+0.024** | Phase 3b vs D3 |

확인되지 않은 것: night_norm(부풀림 보정 후 소멸), 시각 특징(해로움), label smoothing, REM class weight,
cosine LR 단독(무효 — 단 KD와 결합 시 안정화).

**4-stage 목표 0.60과는 여전히 0.07 이상 떨어져 있다.** REM F1 최고가 0.268이며 데이터 제약이 원인이다(§5).

## 2. 전체 궤적

| 단계 | 설정 | Macro-F1 | 부풀림 | 보정 | seed |
| --- | --- | ---: | ---: | ---: | :---: |
| parent (GPU3) | Conformer + 동결 BiLSTM40 | 0.4833 | — | — | 1 |
| C0 | 동일 GPU 대조군 | 0.4716 | +0.0024 | 0.4692 | 1 |
| C2 | + night_norm | 0.4952 | +0.0300 | 0.4652 | 1 |
| C3 | + SpecAugment | 0.5052 | +0.0113 | 0.4939 | 1 |
| C4 | + 둘 다 | 0.4912 | +0.0104 | 0.4808 | 1 |
| D0 | = C3, 3 seed 재측정 | 0.4952 ± 0.0089 | +0.0111 | 0.4841 | 3 |
| D2 | + gated KD | 0.5053 ± 0.0027 | +0.0118 | 0.4935 | 3 |
| **D3** | **+ cosine + gated KD** | **0.5071 ± 0.0094** | +0.0139 | 0.4932 | 3 |
| P3 (1차) | end-to-end, 예산 부족 | 0.4765 | **+0.0868** | 0.3897 | 1 |
| **P3b** | **end-to-end + SpecAugment + cosine** | **0.5310** | +0.0182 | **0.5128** | 1 |

## 3. Phase 2 — encoder 단계 (C0 대조군 대비)

| | C1 encoder | B1 | Δ vs C0 | Wake | REM | Light | Deep | κ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C0 | 0.4127 | 0.4716 | 기준 | 0.564 | 0.197 | 0.873 | 0.253 | 0.334 |
| C2 night_norm | 0.4166 | 0.4952 | +0.0236 | 0.591 | 0.266 | 0.864 | 0.260 | 0.345 |
| **C3 SpecAugment** | 0.4260 | **0.5052** | **+0.0336** | 0.606 | 0.268 | 0.875 | 0.272 | 0.362 |
| C4 둘 다 | **0.4458** | 0.4912 | +0.0196 | 0.596 | **0.147** | 0.888 | 0.333 | 0.371 |

**세 가지를 짚어야 한다.**

**(a) C4는 사전 등록 기준을 통과하지 못했다.** 판정 대상으로 등록한 것은 C4(결합)였고 기준은
"Δ ≥ +0.01 **이면서** REM·Deep 미하락"이었다. REM F1이 0.197 → 0.147로 떨어져 **미달**이다.
C3이 통과한 것은 사후 관찰이므로 그만큼 증거가 약하고, 이것이 3 seed 확인(Phase 4 D0)을 먼저 한 이유다.

**(b) 결합이 단독보다 나빴다(C4 < C3).** 참조 트랙의 "단독 무효, 결합해야 효과" 패턴과 반대다.
encoder가 다르면 combo 필요성도 달라진다는 뜻으로 읽힌다.

**(c) C1↑ / B1↓ 탈동조가 또 나왔다.** C4는 단일 epoch 성능이 0.4458로 가장 좋은데 downstream은
C3보다 낮다. `b0_lr1e4`에서 관찰된 것과 같은 패턴이며, **encoder 단독 지표로 downstream을 예측할 수 없다.**

## 4. Phase 4 — 동결 embedding 위 KD (3 seed, C3 embedding)

| | 평균 | sd | D0 대비 paired Δ | 전 seed 동일부호 |
| --- | ---: | ---: | ---: | :---: |
| D0 C3 기준 | 0.4952 | 0.0089 | 기준 | — |
| D1 + cosine | 0.4968 | 0.0023 | +0.0017 | 아니오 |
| D2 + gated KD | 0.5053 | 0.0027 | +0.0101 | 아니오 |
| **D3 + cosine + KD** | **0.5071** | 0.0094 | **+0.0119** | **예** |

D3는 네 클래스 모두 하락이 없다(Wake +0.002, REM +0.008, Light +0.011, Deep +0.027).

**KD가 이득의 대부분이다** (D2−D0 = +0.0101, D3−D1 = +0.0103). cosine은 단독 무효이나
KD와 결합 시 부호를 안정화한다(D2는 한 seed 음수, D3는 3 seed 전부 양수).

**게이트가 작동했다.** 참조 트랙에서 게이트 없는 KD는 REM을 −0.10 깎았으나 여기서는 REM +0.008이다.

### Teacher 타깃 변환

원본(`byoungjun/cache/teacher_targets`)은 **legacy 순서**(Wake, Light, Deep, REM)이고
**캐시 축**(subject_epoch_index)에 색인돼 있었다. canonical 순서 + 실제 epoch 시간축으로 변환했고,
**argmax 일치율 0.8951**로 검증했다(순서를 틀리면 ~0.10으로 떨어진다). 235명 / 132,164 epoch.

## 5. Phase 3 / 3b — end-to-end

| | epochs | Macro-F1 | 부풀림 | 보정 | Wake | REM | Light | Deep | κ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| P3 1차 (예산 부족) | 16 | 0.4765 | **+0.0868** | 0.3897 | 0.561 | 0.113 | 0.890 | 0.343 | 0.372 |
| **P3b** | 47 | **0.5310** | +0.0182 | **0.5128** | 0.577 | 0.268 | 0.880 | **0.399** | **0.395** |

**1차 결과를 폐기 근거로 쓰지 않은 것이 옳았다.** 1차는 16 epoch 조기종료·스케줄 없음으로 미학습이었고,
best epoch 7의 0.4765는 부풀림 +0.0868인 **단일 epoch 스파이크**였다(ep6 0.400 → ep7 0.477 → ep8 0.379).
예산(50 epochs, patience 12)과 정규화(SpecAugment, cosine+warmup)를 붙이자 0.5310으로 올랐고,
마지막 8 epoch이 0.50~0.53에 머물러 **안정된 수준**임을 보인다.

**Deep F1 0.399는 전 실험 최고**다(D3 0.265). 동결 2단계에서 encoder가 배우지 못하던
문맥 의존 특징을 공동 학습이 확보한 것과 일관된다.

비용: 1.80 h, 47 epochs, peak 2,634 MiB, 2,631,528 params.

## 6. 방법론에서 확인된 것

**(a) 선택 부풀림은 regime마다 다르다.** best epoch − 인접 epoch 평균:

| regime | 부풀림 |
| --- | ---: |
| 동결 embedding B1 | +0.011 ~ +0.016 |
| encoder 재학습 | +0.010 ~ +0.030 |
| end-to-end (미학습) | **+0.087** |
| end-to-end (충분 학습) | +0.018 |

참조 트랙 실측 +0.049는 전 regime에 적용할 상수가 아니다. **불안정한 실행일수록 부풀림이 크므로,
부풀림 자체가 결과 신뢰도의 지표로 쓸 수 있다.**

**(b) GPU 간 비트 재현은 성립하지 않는다.** 같은 GPU 재실행은 해시가 완전히 일치하나,
GPU가 다르면 요소의 약 0.05%가 float16 2 ULP 이내에서 달라진다. 그 작은 차이가 downstream 학습을
발산시켜 C0의 B1은 parent보다 0.0117 낮았다. **기존 보고서의 checkpoint 해시는 GPU 종속이며,
다른 GPU에서의 재현 검증 기준으로 쓸 수 없다.** Phase 2에 C0 대조군을 넣은 이유다.

**(c) 단일 seed 수치를 대표값으로 쓰면 안 된다.** C3의 0.5052는 3 seed 중 운 좋은 하나였고
평균은 0.4952 ± 0.0089였다.

## 7. 구현

| 위치 | 추가 |
| --- | --- |
| `src/psg_only/models.py` | `time_feature`, `window_hours()` |
| `src/psg_only/data.py` | `night_statistics()`, `night_norm`, `MelWindowDataset` |
| `src/psg_only/train.py` | `spec_augment()`, `build_scheduler()`, class weight multipliers, label smoothing |
| `src/psg_only/endtoend.py` | `EndToEndModel` (신규) |
| `src/psg_only/evaluate.py` | night_norm·time_feature 반영 |
| `scripts/` | `run_phase1/2/3/4.py`, `build_teacher_cache.py`, `analyze_*.py` |
| `tests/` | 신규 28건 (총 **72건 통과**) |

옵션은 전부 기본값 off로 추가해 기존 체크포인트가 그대로 로드된다.
무결성 게이트 2종(A0 parent 비트 재현, embedding 재생성 2 ULP)을 통과했다.

구현 중 잡은 버그 둘 — `evaluate_b0`가 night_norm을 무시(학습/평가 정규화 불일치),
end-to-end 루프에서 지역변수 `batch`가 dataloader 튜플을 가림. 둘 다 정규 실행 전에 잡았다.

## 8. 남은 격차

| 지표 | 현재 최고 | 목표 |
| --- | ---: | ---: |
| Macro-F1 | 0.5310 (보정 0.5128) | 0.60 |
| κ | 0.3951 | 0.50 |
| Accuracy | 0.7987 | 0.60 (달성) |
| REM F1 | 0.268 | — |

REM 제약은 그대로다. V3 287명 중 **102명(35.5%)이 밤 전체에 REM 0**, 전체 REM 4.77%,
녹음 중앙값 4.45 h, 중증 OSA(AHI 중앙값 54)·split-night 57명.
`test_locked` 49명도 REM 4.8%이므로 test에서도 개선되지 않는다.

## 9. 진행 중 / 다음

**진행 중 (2026-09-18 04:58 시작)**

- Phase 5: **end-to-end + gated KD** 3 seed (GPU 3) — 확인된 두 최대 레버의 결합
- Phase 3b 재현: seed 2개 추가 (GPU 1) — 0.5310이 단일 seed이므로

**다음 후보**

1. 승자의 5-fold CV 평가. 고정 val-44는 CV 평균보다 약 0.027 낙관적이다.
2. end-to-end 위 night_norm 재시험. 동결 경로에서는 부풀림에 가려졌으나 공동 학습에서는 다를 수 있다.
3. KD α/T 스윕 (현재 0.3 / 2.0 고정).
4. 목표 지표 재협의. 4-stage 0.60은 이 코호트에서 구조적으로 어렵다.

> GPU 2는 현재 하드웨어 오류 상태다(`Unable to determine the device handle for GPU2`).
