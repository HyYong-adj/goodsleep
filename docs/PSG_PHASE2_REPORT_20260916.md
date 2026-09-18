# Phase 2 결과 — encoder 단계 combo (night_norm, SpecAugment)

실행: `phase2_20260916`, 상태 **COMPLETE**, device cuda, seed [20260910]

Validation-only, 단일 seed screening. `test_locked` 미접근.

## 전체

| Variant | 변경 | C1 (단일 epoch) | B1 (downstream) | B1 Δ vs C0 | κ | Acc |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| C0 | 대조군 (옵션 없음, 동일 GPU) | 0.4127 | **0.4716** | 기준 | 0.3340 | 0.7827 |
| C2 | night_norm | 0.4166 | **0.4952** | +0.0236 | 0.3453 | 0.7743 |
| C3 | SpecAugment | 0.4260 | **0.5052** | +0.0336 | 0.3621 | 0.7903 |
| C4 | night_norm + SpecAugment | 0.4458 | **0.4912** | +0.0196 | 0.3710 | 0.8102 |

## 클래스별 B1 F1

| Variant | Wake | REM | Light | Deep | REM recall | Deep recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C0 | 0.5637 | 0.1965 | 0.8730 | 0.2530 | 0.2327 | 0.2015 |
| C2 | 0.5910 | 0.2664 | 0.8638 | 0.2595 | 0.3201 | 0.2445 |
| C3 | 0.6060 | 0.2683 | 0.8748 | 0.2717 | 0.3190 | 0.2323 |
| C4 | 0.5960 | 0.1474 | 0.8883 | 0.3331 | 0.1146 | 0.2914 |

## 판정

기준: C4 의 C0 대비 Δ ≥ +0.01 **이면서** REM·Deep F1 미하락. 단일 seed 이므로 채택 시 3 seed 확장이 선행 조건.

- **C2** (night_norm): ΔMacro-F1 +0.0236, ΔREM +0.0699, ΔDeep +0.0065 → 임계 충족
- **C3** (SpecAugment): ΔMacro-F1 +0.0336, ΔREM +0.0717, ΔDeep +0.0186 → 임계 충족
- **C4** (night_norm + SpecAugment): ΔMacro-F1 +0.0196, ΔREM -0.0491, ΔDeep +0.0801 → 임계 미달

## 비용

| Variant | C1 학습 | 전체 |
| --- | ---: | ---: |
| C0 | 63 min | 66 min |
| C2 | 70 min | 73 min |
| C3 | 72 min | 75 min |
| C4 | 115 min | 119 min |

> C0 는 같은 GPU 대조군이다. parent(GPU 3) B1 은 0.483261 이었으나 C0 는 다른 값이 나온다 — GPU 간 비트 재현이 성립하지 않기 때문이며, 따라서 **판정은 parent 가 아니라 C0 대비**로 한다.

---

# Phase 3 결과 — end-to-end 공동 학습

구조는 parent 와 같고 **학습 방식만** 바꿨다(동결 2단계 -> 공동 학습).
비교 대상은 같은 GPU 의 Phase 2 C0 대조군이다.

- Macro-F1 **0.4765** / κ 0.3721 / Acc 0.8110
- parent B1(0.483261) 대비 -0.0067  *(참고용 — GPU 가 다르므로 C0 대비로 판단할 것)*
- 파라미터 2,631,528 / 16 epochs / 0.72 h / peak 2628 MiB

| 클래스 | F1 | recall |
| --- | ---: | ---: |
| Wake | 0.5605 | 0.6309 |
| REM | 0.1134 | 0.0726 |
| Light | 0.8896 | 0.9063 |
| Deep | 0.3427 | 0.2786 |

**C0 대비 ΔMacro-F1 = +0.0050** (C0 0.4716)

단일 seed, validation-only. 채택 판단 전 3 seed 확장 필요.
