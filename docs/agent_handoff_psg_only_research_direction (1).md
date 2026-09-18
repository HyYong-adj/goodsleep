# Agent Handoff: PSG-Only Sleep Staging Research Direction

> **2026-09-16 상태 노트 — 아래 전제 중 일부는 더 이상 유효하지 않다.**
> 본 문서는 2026-09-10 작성됐다. 이후 확정된 사항을 반영하면 다음과 같다.
>
> | 문서의 전제 | 현재 상태 |
> | --- | --- |
> | §9–§10 long-context 가설 (20분 → full-night) | **답이 나왔다.** `bj-e037` full-night GRU 3 seed: causal 0.517 / bidirectional 0.517 / epoch 단독 0.390. `bj-e022` ub(양방향·2배 폭) 0.518 ≈ combo 0.506. **시간 문맥 포화** |
> | §10 long-context cache 연구 | 전제(긴 문맥이 성능을 올린다)가 성립하지 않으므로 **착수 조건 미충족** |
> | post-wake 비인과 teacher → causal edge student 구도 | **2026-09-15 제품 정의 변경으로 무효.** 업체 확정: "아침에 밤 전체 그래프만 필요, 실시간 불필요 → 비인과 허용, 서버 후보정 불필요(단말 내 처리)" |
> | §8 synthetic noise consistency | byoungjun 트랙이 선행 수행 중(`bj-e048`, homeaug). 복합 가정조건 0.344 → 0.481 |
> | §13 4-stage를 주 task로 고정 | **재협의 예정.** REM 4.77% / 287명 중 102명 REM 0 → `bj-e033`에서 3-stage 주 지표 협의가 올라와 있다 |
>
> 여전히 유효한 것: §3 canonical task 정의, §4 subject-disjoint split 규칙, §6 필수 metric, §15 단일 변수 원칙.
>
> 현재 방향은 [다음 실험 방향 제언 rev.2](PSG_RESEARCH_DIRECTION_20260916.md)를 따른다.

## 1. Objective

현재 보유 가능한 데이터 조건을 기준으로, **HomeSleepNet 방법론을 직접 재현하기보다 PSG와 동기화된 audio + PSG sleep-stage label만으로 학습 가능한 supervised sleep staging baseline을 먼저 구축한다.**

핵심 방향은 다음과 같다.

> **PSG-audio only source-domain training에서 시작하여, synthetic acoustic domain generalization과 long-term sleep context modeling을 통해 real-world smartphone sleep staging으로 확장한다.**

주의할 점:

- 여기서 PSG는 모델 입력이 아니라 **sleep-stage ground truth를 제공하는 역할**이다.
- 실제 모델 입력은 **PSG 검사 중 동기화되어 녹음된 audio**여야 한다.
- 만약 현재 데이터가 EEG/EOG/EMG 등의 PSG signal만 있고 audio가 없다면 HomeSleepNet/SoundSleepNet 계열 audio baseline을 구현할 수 없다.
- 현재 baseline은 `HomeSleepNet reproduction`이 아니라 **`PSG-only supervised baseline` 또는 `SoundSleepNet-like baseline`**으로 정의한다.

---

## 2. Why PSG-Only Baseline First

HomeSleepNet의 학습은 크게 세 가지 component로 구성된다.

```text
HomeSleepNet training
├── Supervised sleep-stage learning
│   └── Hospital PSG + synchronized audio + stage labels
│
├── Unsupervised domain adaptation
│   └── Hospital audio + unlabeled home smartphone audio
│
└── Consistency training
    └── Hospital audio + synthetic home noise
```

현재 실제 home smartphone audio가 없다면:

- `Supervised learning`: 가능
- `Domain adaptation`: 원 논문 방식으로는 불가능
- `Consistency training`: synthetic environmental/home noise를 확보하면 가능

따라서 첫 번째 milestone은 **PSG-audio만 이용한 supervised baseline을 완성하는 것**이다.

---

## 3. Canonical Task Definition

### Input

```text
30-second synchronized sleep audio
```

### Output classes

```text
0 = Wake
1 = REM
2 = Light   # N1 + N2
3 = Deep    # N3
```

### Canonical mapping

```text
Wake  -> Wake
REM   -> REM
N1    -> Light
N2    -> Light
N3    -> Deep
```

### Evaluation unit

```text
30-second epoch
```

### Mandatory split rule

반드시 **subject-disjoint / participant-night-disjoint split**을 사용한다.

금지:

```text
random epoch split
same subject across train/test
same night or channel derivative across different splits
```

---

## 4. Baseline Definition

Baseline은 두 단계로 나눈다.

---

### B0. Single-Epoch CNN Baseline

목적:

- data loading 검증
- audio-label alignment 검증
- label mapping 검증
- training/evaluation pipeline smoke test
- class imbalance 확인

구조:

```text
30 sec audio
    ↓
Mel spectrogram
    ↓
CNN encoder
    ↓
Linear classifier
    ↓
Wake / REM / Light / Deep
```

Loss:

```text
Cross Entropy only
```

B0에서는 성능 자체보다 pipeline correctness가 중요하다.

---

### B1. SoundSleepNet-Like Supervised Sequence Baseline

HomeSleepNet이 사용한 기반 모델 구조를 참고하여 sequence context를 추가한다.

논문 설정:

```text
40 input epochs
× 30 sec
= 20 min context
```

입력 40 epoch에서 가운데 20 epoch를 예측하는 many-to-many 방식이다.

구조 예시:

```text
40 × 30 sec audio
      ↓
Mel spectrogram sequence
      ↓
CNN acoustic / epoch encoder
      ↓
epoch embeddings
      ↓
Temporal encoder
(BiLSTM / TCN / Transformer)
      ↓
middle 20 epoch predictions
      ↓
Wake / REM / Light / Deep
```

B1을 이후 모든 실험의 **main parent baseline**으로 사용한다.

---

## 5. Recommended Initial Configuration

```yaml
task:
  num_classes: 4
  labels:
    - Wake
    - REM
    - Light
    - Deep

audio:
  sample_rate: 16000
  epoch_seconds: 30

feature:
  type: log_mel

sequence:
  input_epochs: 40
  output_epochs: 20

training:
  loss: cross_entropy
  supervised: true
  domain_adaptation: false
  consistency: false
```

추후 context length는 config로 조절 가능해야 한다.

예:

```yaml
sequence:
  input_epochs: 40
```

→ 이후 80, 120, 240, full-night 등으로 확장 가능하도록 구현한다.

---

## 6. Baseline Completion Criteria

Baseline v0가 완료되었다고 판단할 조건:

- PSG audio와 PSG stage label의 30초 alignment 검증
- 4-stage canonical mapping 적용
- subject-disjoint split 고정
- reproducible config
- B0 학습 완료
- B1 학습 완료
- validation inference 완료
- shared evaluator 사용
- confusion matrix 생성
- per-class metric 저장

필수 metric:

```text
Macro-F1
Cohen's kappa
Accuracy

Wake F1
REM F1
Light F1
Deep F1

Confusion Matrix
```

Accuracy보다 **Macro-F1과 Cohen's kappa를 우선 해석한다.**

---

## 7. Recommended Research Direction After Baseline

Baseline 이후 연구는 다음 순서로 진행하는 것을 권장한다.

```text
B0
Single epoch CNN
        ↓
B1
PSG-only supervised sequence model
        ↓
--------------------------------
Baseline complete
--------------------------------
        ↓
B2
Noise robustness / consistency training
        ↓
B3
Long-context temporal modeling
        ↓
B4
Efficient long-context modeling
        ↓
B5
Cross-device / cross-dataset domain generalization
        ↓
High-performance teacher
```

---

# 8. Research Axis 1: Synthetic Home Noise Consistency

실제 home PSG 또는 home smartphone audio가 없어도 consistency training은 가능하다.

핵심 아이디어:

```text
clean PSG audio
        ↓
      model
        ↓
    prediction P

clean PSG audio
 + environmental noise
        ↓
     noisy audio
        ↓
       model
        ↓
    prediction Q
```

목표:

```text
P ≈ Q
```

Loss 예시:

```text
L_total
=
L_CE
+
lambda * L_consistency
```

가능한 consistency loss:

```text
Jensen-Shannon divergence
KL divergence
MSE between logits
```

Noise 후보:

```text
fan
HVAC
traffic
room noise
speech
TV
bedding movement
white/pink noise
microphone/device response
```

SNR range 예시:

```text
-10 dB ~ +10 dB
```

연구 질문:

> 실제 home target-domain data 없이 synthetic acoustic corruption만으로 source-only model의 robustness를 얼마나 높일 수 있는가?

---

# 9. Research Axis 2: Long-Context Sleep Modeling

SoundSleepNet/HomeSleepNet 계열 기본 context:

```text
40 epochs × 30 sec = 20 min
```

하지만 sleep stage는 독립적인 30초 classification이 아니라 temporal process이다.

개념적으로:

```text
P(y_t | x_t)
```

보다

```text
P(
  y_t |
  x_t,
  neighboring acoustic features,
  previous sleep stages,
  sleep-cycle history
)
```

가 더 적절하다.

따라서 baseline 이후 가장 먼저 확인할 핵심 실험:

```text
20 min
→ 40 min
→ 1 hour
→ 2 hour
→ full-night
```

실험 표 예시:

| Experiment | Context | Model | Objective |
|---|---:|---|---|
| B1 | 20 min | CNN + BiLSTM | CE |
| E1 | 40 min | CNN + BiLSTM | CE |
| E2 | 1 hour | CNN + TCN/Transformer | CE |
| E3 | 2 hour | CNN + Transformer | CE |
| E4 | full-night | Temporal model | CE |

확인할 metric:

```text
overall Macro-F1
REM F1
Deep F1
transition accuracy
transition violation rate
stage duration plausibility
```

특히 다음을 확인한다.

> context가 길어질수록 REM / Deep / stage boundary 성능이 실제로 좋아지는가?

---

# 10. Long-Context Cache 연구는 Baseline 이후

현재 장기적으로 고려 중인 caching 아이디어는 baseline 단계에서 바로 적용하지 않는다.

먼저 다음 가설을 검증해야 한다.

```text
Longer context
→ better sleep staging?
```

만약

```text
20 min < 1 h < 2 h < full-night
```

순으로 성능이 향상되지만 계산량과 memory가 급증한다면, 그때 cache 연구의 필요성이 생긴다.

향후 연구 질문:

> 긴 sleep history 중 어떤 epoch 정보를 유지하고 어떤 정보를 압축해도 현재 stage prediction 성능을 보존할 수 있는가?

Video long-context 연구와 대응:

```text
Video                      Sleep staging
------------------------------------------------
motion                     stage transition
dynamic region             REM/Wake transition
static region              stable sleep stage
frame cache                epoch feature cache
long-term frame memory     long-term sleep history
```

예상 방향:

```text
최근 epoch
→ high-resolution representation

오래된 epoch
→ compressed representation / cache

transition probability high
→ frequent update

stable sleep region
→ aggressive compression
```

단, 이는 B1/B3 이후 진행한다.

---

# 11. Research Axis 3: Domain Generalization Instead of Home UDA

HomeSleepNet의 UDA는 실제 home smartphone audio를 target domain으로 사용한다.

현재 home audio가 없다면 동일한 UDA를 구현하는 것은 적절하지 않다.

따라서 연구 프레이밍은:

```text
Unsupervised Domain Adaptation
```

보다

```text
Source-Only Domain Generalization
```

에 가깝게 잡는다.

가능한 domain variation:

```text
dataset
microphone
audio channel
SNR
noise type
recording device
sample rate
room simulation
```

추후 실험:

```text
domain randomized augmentation
device response simulation
RIR / MIR augmentation
domain adversarial training using available device/channel labels
leave-one-dataset-out evaluation
leave-one-device-out evaluation
```

주의:

`hospital vs home` target domain이 없는데 arbitrary channel A/B를 이용해 GRL을 넣는 것은 HomeSleepNet replication이 아니다.

그 경우 연구 명칭은 명확하게:

```text
device/channel domain-invariant representation learning
```

으로 정의한다.

---

# 12. Foundation Models Should Come After Stable Baseline

OPERA, AudioMAE 등 pretrained audio encoder를 실험할 수 있지만 최초 baseline에는 넣지 않는 것을 권장한다.

이유:

```text
pretrained representation gain
+
temporal context gain
+
augmentation gain
```

이 동시에 섞이면 어느 요소가 성능 향상에 기여했는지 해석하기 어렵다.

권장 순서:

```text
B0 CNN
↓
B1 CNN + temporal
↓
context / noise experiments
↓
OPERA / AudioMAE replacement
↓
controlled ablation
```

---

# 13. Four-Stage Should Be the Main Task

원 HomeSleepNet 논문의 대표 결과는 3-stage이지만 현재 프로젝트의 target은:

```text
Wake
REM
Light
Deep
```

4-stage이다.

따라서 모든 baseline과 실험의 기본 task를 4-stage로 고정한다.

특히 주의할 confusion:

```text
Light ↔ Deep
REM ↔ Light
Wake ↔ Light
```

HomeSleepNet 계열에서도 Light와 Deep 구분은 어려운 문제로 보고되었다.

따라서 단순 accuracy보다 per-class F1 분석이 필수다.

---

# 14. Recommended Experiment Roadmap

## Phase 0 — Foundation

### E0-1
Data manifest 생성

```text
subject_id
night_id
epoch_idx
audio_path
label
dataset
channel
device
sample_rate
```

### E0-2
30-second alignment 검증

### E0-3
4-stage label mapping 검증

### E0-4
subject-disjoint split 생성 및 고정

### E0-5
shared evaluator 구축

---

## Phase 1 — Baseline

### B0
Single epoch CNN

```text
30 sec
→ Mel
→ CNN
→ stage
```

### B1
Sequence supervised baseline

```text
40 epochs
→ acoustic encoder
→ temporal encoder
→ middle 20 predictions
```

---

## Phase 2 — Robustness

### R1
Noise augmentation only

### R2
Noise + consistency loss

### R3
RIR / MIR / device simulation

---

## Phase 3 — Temporal Context

### T1
20 min

### T2
40 min

### T3
1 hour

### T4
2 hour

### T5
full-night

비교:

```text
BiLSTM
TCN
Transformer
```

---

## Phase 4 — Long-Context Efficiency

Long-context 실험에서 성능 향상이 확인된 경우에만 진행한다.

후보:

```text
hierarchical temporal encoder
memory compression
multi-scale context
KV cache
state-space model
transition-aware caching
```

---

## Phase 5 — Representation / Domain Generalization

후보:

```text
OPERA
AudioMAE
self-supervised adaptation
domain randomization
channel/device invariance
```

---

# 15. Key Experimental Principles

모든 실험은 한 번에 major factor 하나만 바꾼다.

예:

```text
B1
↓
B1 + noise

B1
↓
B1 + longer context

B1
↓
B1 + AudioMAE
```

다음과 같이 동시에 여러 요소를 변경하는 실험은 지양한다.

```text
B1
→ AudioMAE
→ noise
→ transformer
→ long context
→ focal loss
```

이 경우 성능 변화 원인을 분석할 수 없다.

---

# 16. Naming Recommendation

현재 모델:

```text
PSG-Only Supervised Baseline
```

또는:

```text
SoundSleepNet-Like Supervised Baseline
```

추천하지 않는 명칭:

```text
HomeSleepNet reproduction
```

이유:

실제 HomeSleepNet의 핵심 요소 중 하나인 home-domain adaptation data가 없기 때문이다.

---

# 17. Core Research Story

최종 연구 스토리는 다음과 같이 정리한다.

## Problem

실제 smartphone home sleep staging은 home environment에서 동작해야 하지만, labeled home PSG + smartphone audio 데이터 확보가 어렵다.

## Available Data

```text
PSG-aligned sleep audio + PSG stage labels
```

## Baseline

```text
PSG-only supervised sleep staging
```

## Research Question 1

> 실제 home data 없이 synthetic noise consistency로 acoustic domain robustness를 확보할 수 있는가?

## Research Question 2

> 기존 20분 context를 넘어 장기 sleep history를 사용하면 REM/Deep 및 transition prediction이 향상되는가?

## Research Question 3

> 긴 temporal context가 효과가 있다면, 이를 효율적으로 압축/cache하면서 성능을 유지할 수 있는가?

## Final Goal

```text
PSG-only training
    ↓
robust acoustic representation
    ↓
long-context sleep modeling
    ↓
efficient temporal memory
    ↓
high-performance teacher
    ↓
future smartphone deployment / distillation
```

---

# 18. Immediate Agent Tasks

Agent가 지금 바로 수행해야 할 우선순위:

1. 사용 가능한 dataset에 **synchronized audio가 실제 존재하는지 확인**
2. PSG stage annotation 형식 확인
3. subject / night / channel metadata 확인
4. 30초 epoch alignment pipeline 작성
5. 4-stage mapping 구현
6. subject-disjoint split 생성
7. log-Mel feature pipeline 구현
8. B0 single-epoch CNN smoke test
9. B1 40→20 sequence baseline 구현
10. Macro-F1 / kappa / per-class F1 / confusion matrix evaluation
11. baseline 결과 확보 후 noise consistency 실험 설계
12. 이후 context length ablation 설계

---

# 19. Current Decision

현재 시점의 권장 결정:

```text
DO:
PSG-audio supervised 4-stage baseline first

DO:
B0 → B1 순서로 구현

DO:
40-input / middle-20 prediction을 main temporal baseline으로 고려

DO:
baseline 완료 후 synthetic noise consistency 실험

DO:
그 다음 long-context ablation

DO NOT:
home audio가 없는 상태에서 HomeSleepNet UDA를 억지로 재현

DO NOT:
baseline 이전에 cache / foundation model / domain adversarial 방법을 모두 동시에 도입
```

---

# 20. One-Sentence Summary for Agent

> **우선 PSG와 동기화된 audio 및 PSG sleep-stage label만으로 4-stage supervised SoundSleepNet-like baseline(B0 single-epoch CNN, B1 40→20 temporal model)을 재현 가능하게 구축하고, 이후 synthetic home-noise consistency와 long-context temporal modeling을 순차적으로 검증하며, long context의 효과가 확인된 이후에만 caching/domain-generalization 연구로 확장한다.**
