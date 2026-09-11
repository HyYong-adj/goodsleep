# Agent 실행 명세: PSG-Audio B0/B1 및 후속 실험

> 이 문서는 구현 agent에게 그대로 전달하는 작업 지시서다. 설명 문서가 아니라 파일 변경, 테스트, 실행, 중단 조건을 정의한다.

## 1. 최종 목표

`/home/sleep/researchers/choihy` 안에 PSG 동기화 ambient audio만 입력으로 사용하는 4-stage supervised pipeline을 구현한다.

1. B0: 30초 single-epoch acoustic classifier
2. B1: B0 embedding 40개를 입력하여 중앙 20개를 예측하는 temporal classifier
3. canonical validation evaluator와 결과 schema
4. B0 대비 B1의 동일 split, 동일 seed 비교

PSG 생체신호는 입력이 아니라 ground-truth sleep-stage label 제공에만 사용한다.

## 2. 변경 허용 범위

작성 가능:

- `/home/sleep/researchers/choihy/src/`
- `/home/sleep/researchers/choihy/scripts/`
- `/home/sleep/researchers/choihy/configs/`
- `/home/sleep/researchers/choihy/tests/`
- `/home/sleep/researchers/choihy/artifacts/`의 작은 metadata
- `/home/sleep/researchers/choihy/docs/`

읽기 전용 참조:

- `/home/sleep/researchers/byoungjun/`
- `/home/sleep/researchers/cloud9sm/`
- `/home/sleep/researchers/psg_audio_baseline_pipeline/`
- `/home/sleep/data/`
- `/home/sleep/eval/`

공용 `data/`, `eval/`, `common/`, `integration/`과 다른 연구자의 파일은 수정하지 않는다. 공용 승격은 구현·검증 후 별도 review 대상으로 남긴다.

## 3. 시작 전 반드시 읽을 파일

1. `/home/sleep/AGENTS.md`
2. `docs/agent_handoff_psg_only_research_direction (1).md`
3. `docs/agent_handoff_homesleepnet.md`
4. `/home/sleep/researchers/psg_audio_baseline_pipeline/src/sleepteacher/data/psg_audio.py`
5. `/home/sleep/researchers/psg_audio_baseline_pipeline/src/sleepteacher/data/psg_audio_v3_dataset.py`
6. `/home/sleep/researchers/cloud9sm/src/fullnight_sleep/data.py`
7. `/home/sleep/researchers/cloud9sm/src/fullnight_sleep/models.py`
8. `/home/sleep/researchers/cloud9sm/src/fullnight_sleep/metrics.py`

## 4. 확인된 현재 상태

### 재사용할 실제 자산

- manifest: `/home/sleep/researchers/byoungjun/manifests/teacher_full.csv`
- manifest metadata: `/home/sleep/researchers/byoungjun/manifests/teacher_full.json`
- Mel cache: `/home/sleep/researchers/byoungjun/cache/audio_compact_full`
- train normalization: `/home/sleep/researchers/byoungjun/cache/audio_compact_full/train_stats.json`

| 항목 | 확인된 값 |
|---|---|
| subjects | train 191, val 44, test 0 |
| rows | 135,010 |
| train class count | Wake 8,426; Light 91,528; Deep 5,515; REM 5,483 |
| val class count | Wake 1,997; Light 19,622; Deep 1,558; REM 881 |
| cache item | `mels.npy`, `labels.npy`, `valid.npy`, `stats.json` |
| Mel shape | subject별 `[T,48,1499]` |
| Mel dtype | float16 on disk |
| sample rate | 8,000 Hz |
| train mean/std | -40.8819716989 / 11.8089926024 |

### 사용하지 않을 경로

`/disk4/personal/sjhoney/psg_audio/V3/APNEA_EDF`와 `APNEA_RML`은 현재 호스트에 없다. raw EDF가 필요한 작업은 경로가 제공될 때까지 중단한다.

### 현재 제약

- `/home/sleep/eval`은 README와 skeleton만 있고 evaluator가 구현되지 않았다.
- system Python에는 torch와 pyedflib가 없어 기존 pipeline test가 collection 단계에서 실패한다.
- 현재 manifest에는 test subject가 없다. train/validation만 사용하고 test 성능을 주장하지 않는다.
- GPU 실행 전 가용 GPU를 확인하고 사용자에게 GPU index를 배정받아야 한다.

## 5. 클래스 순서 migration

프로젝트 canonical 순서:

```python
CANONICAL_CLASSES = ("Wake", "REM", "Light", "Deep")
```

기존 manifest/cache의 legacy 순서:

```python
LEGACY_CLASSES = ("Wake", "Light", "Deep", "REM")
LEGACY_LABEL_TO_CANONICAL = (0, 2, 3, 1)
CANONICAL_LOGITS_FROM_LEGACY = (0, 3, 1, 2)
```

원본 manifest, cache, checkpoint를 재작성하지 않는다. 새 dataset adapter가 label을 읽을 때 canonical index로 바꾼다. 새 prediction, checkpoint, metric은 canonical 순서만 저장하고 `class_order_version: wake-rem-light-deep-v1`을 포함한다.

먼저 다음을 test한다.

- legacy label `[0,1,2,3]` → canonical `[0,2,3,1]`
- legacy logits column → `[0,3,1,2]`
- label name/index round-trip
- checkpoint의 class order 누락 시 명시적 실패

## 6. 생성할 구조

```text
/home/sleep/researchers/choihy/
├── pyproject.toml
├── configs/{b0_epoch,b1_40to20}.yaml
├── src/psg_only/
│   ├── __init__.py
│   ├── constants.py
│   ├── config.py
│   ├── data.py
│   ├── windows.py
│   ├── models.py
│   ├── metrics.py
│   ├── train.py
│   └── evaluate.py
├── scripts/
│   ├── validate_inputs.py
│   ├── train_b0.py
│   ├── cache_embeddings.py
│   ├── train_b1.py
│   └── evaluate.py
├── tests/
│   ├── test_constants.py
│   ├── test_data.py
│   ├── test_windows.py
│   ├── test_models.py
│   └── test_metrics.py
└── artifacts/                 # Git에는 작은 metadata만
```

다른 연구자 directory를 `sys.path`에 넣지 않는다. 필요한 최소 동작을 독립 모듈로 구현하되 참조한 파일과 차이를 module docstring에 기록한다.

## 7. dataset 계약

### CachedMelEpochDataset

- CSV의 `subject_id`를 string dtype으로 읽는다.
- 허용 split은 `train`, `val`뿐이다.
- `subject_epoch_index`로 `{cache_root}/{subject_id}/mels.npy`를 조회한다.
- `valid.npy`가 false면 target은 -100이다.
- manifest row와 cache index 불일치는 실패한다.
- float16 Mel을 float32 `[1,48,1499]`로 변환한다.
- `train_stats.json` mean/std로 normalize한다.
- legacy label을 canonical index로 remap한다.
- 반환: `{x, y, subject_id, epoch_index, valid}`.

### CachedSubjectSequenceDataset

- subject별 `subject_epoch_index` 정렬을 강제한다.
- index 감소나 중복은 실패한다.
- 전체 night의 Mel 또는 cached embedding, label, valid mask를 반환한다.
- 누락 epoch를 압축 제거하지 말고 invalid mask를 유지한다.

### validate_inputs.py

다음을 non-zero exit 기준으로 검사한다.

- manifest required columns
- train/val subject overlap 0
- cache/manifest subject set과 array 길이
- Mel `[T,48,1499]`, label/valid `[T]`
- finite Mel의 표본 검사
- train/val 네 class support
- train normalization metadata
- test split이 없음을 경고로 명시

## 8. B0 모델

`cloud9sm/src/fullnight_sleep/models.py`의 `AcousticEncoder`를 구조 참고로 사용한다.

```text
Input [B,1,48,1499]
Conv2d 1→32, k3, stride2
Depthwise blocks 32→64→96→128→160, 각각 stride2
Conv2d 160→192, k1
AdaptiveAvgPool2d(1)
embedding [B,192]
Linear 192→4
```

B0 loss:

```text
w[c] = count[c]^-0.5
w = w / mean(w)
loss = CrossEntropy(logits, target, weight=w, ignore_index=-100)
```

기본 config:

```yaml
seed: 20260910
data:
  manifest: /home/sleep/researchers/byoungjun/manifests/teacher_full.csv
  cache_root: /home/sleep/researchers/byoungjun/cache/audio_compact_full
  stats: /home/sleep/researchers/byoungjun/cache/audio_compact_full/train_stats.json
  allowed_splits: [train, val]
model:
  embedding_dim: 192
  dropout: 0.2
train:
  epochs: 30
  batch_size: 16
  learning_rate: 0.0003
  weight_decay: 0.0001
  early_stopping_patience: 10
  grad_clip_norm: 1.0
  amp: true
```

smoke는 8 train subjects, 2 val subjects, 최대 2 optimizer steps다. tiny overfit은 별도의 작은 고정 subset에서 수행한다.

## 9. embedding cache

B0 best checkpoint의 encoder를 freeze하고 subject별로 생성한다.

```text
artifacts/embeddings_b0_v1/{subject_id}/
├── embeddings.npy   # [T,192], float16
├── labels.npy       # canonical int64
├── valid.npy        # bool
└── metadata.json
```

metadata는 B0 checkpoint SHA-256, source cache logical name, class order/version, shape/dtype, subject ID, config hash를 포함한다. checkpoint/config hash가 다르면 기존 cache를 덮어쓰지 말고 새 version directory를 만든다.

## 10. B1 40→20 모델

window는 10 left context + 20 target + 10 right context다.

- input `[B,40,192]`
- target positions `[10:30]`
- stride 20
- night 경계를 넘지 않음
- 없는 boundary는 zero-pad하고 `input_valid=false`
- 실제 target만 `target_valid=true`
- 같은 실제 epoch를 두 번 평가하지 않음

```text
LayerNorm(192)
BiLSTM(input=192, hidden=128, layers=2, bidirectional, dropout=0.2)
Dropout(0.2)
Linear(256→4)
Output [B,20,4]
```

B1에서는 B0 encoder를 freeze하고 embedding cache만 쓴다. temporal context 효과를 acoustic encoder 변화와 분리하기 위함이다.

```yaml
seed: 20260910
data:
  manifest: /home/sleep/researchers/byoungjun/manifests/teacher_full.csv
  embedding_root: artifacts/embeddings_b0_v1
  input_epochs: 40
  target_epochs: 20
  left_context: 10
  stride: 20
model:
  input_dim: 192
  hidden_dim: 128
  layers: 2
  bidirectional: true
  dropout: 0.2
train:
  epochs: 50
  batch_size: 32
  learning_rate: 0.0003
  weight_decay: 0.0001
  early_stopping_patience: 10
  grad_clip_norm: 1.0
  amp: true
```

## 11. canonical evaluator

B0/B1이 모두 `src/psg_only/metrics.py`를 호출한다.

필수 metric:

- accuracy
- Macro-F1 with labels `[0,1,2,3]`
- Cohen's kappa
- class별 precision, recall, F1, support
- raw/row-normalized confusion matrix
- transition/stable Macro-F1
- true/predicted transition rate와 absolute error
- 같은 validation set의 majority-class baseline

class support가 0이면 formal 4-stage result를 `invalid`로 기록한다.

prediction schema:

```text
subject_id, epoch_index, target, prediction,
prob_wake, prob_rem, prob_light, prob_deep,
valid, model_version, class_order_version
```

`results.json`은 schema version, experiment ID, seed, code/config/checkpoint hash, logical data name, subject/epoch count, metrics, class order, artifact location을 포함한다. raw EDF path는 저장하지 않는다.

## 12. 구현 순서

### Task 0 — 환경·입력 검증

1. Python/venv를 확인한다.
2. dependency 설치가 필요하면 시스템 환경을 임의 변경하지 않는다.
3. `validate_inputs.py`를 구현하고 실행한다.
4. test split 0을 report한다.

### Task 1 — constants/data/metrics

```bash
python -m pytest tests/test_constants.py tests/test_data.py tests/test_metrics.py -q
```

### Task 2 — B0

```bash
python scripts/train_b0.py --config configs/b0_epoch.yaml --smoke
python scripts/train_b0.py --config configs/b0_epoch.yaml --seed 20260910
python scripts/evaluate.py --checkpoint <B0_RUN>/best.pt --output <B0_RUN>/evaluation --split val
```

GPU index를 배정받은 뒤에만 `CUDA_VISIBLE_DEVICES=<assigned>`로 실행한다.

### Task 3 — embedding

```bash
python scripts/cache_embeddings.py \
  --checkpoint <B0_RUN>/best.pt \
  --output artifacts/embeddings_b0_v1
```

### Task 4 — B1

```bash
python -m pytest tests/test_windows.py tests/test_models.py -q
python scripts/train_b1.py --config configs/b1_40to20.yaml --smoke
python scripts/train_b1.py --config configs/b1_40to20.yaml --seed 20260910
python scripts/evaluate.py --checkpoint <B1_RUN>/best.pt --output <B1_RUN>/evaluation --split val
```

### Task 5 — 공정 비교

동일 validation subjects와 evaluator로 B0/B1을 비교한다. B1 stitching 후 모든 valid validation epoch가 정확히 한 번 존재하는지 assert한다.

## 13. 필수 테스트

| 테스트 | 합격 기준 |
|---|---|
| class migration | legacy label/logit 변환 정확 |
| manifest | required columns와 split 검증 |
| leakage | train/val subject 교집합 시 실패 |
| cache lookup | epoch index와 array 정렬 |
| feature | `[1,48,1499]`, float32, finite |
| normalization | 고정 mean/std 적용 |
| B0 shape | logits `[B,4]`, embedding `[B,192]` |
| window lengths | 1,19,20,21,39,40,41에서 누락/중복 0 |
| B1 shape | logits `[B,20,4]` |
| mask | padding이 loss/metric에 기여하지 않음 |
| metrics | hand-computed fixture와 일치 |
| save/load | 같은 input logits 일치 |
| determinism | 동일 seed smoke 재실행 일치 |

## 14. 완료 정의

- [ ] 다른 연구자의 파일을 수정하지 않았다.
- [ ] input validation report가 있다.
- [ ] canonical class migration test가 통과한다.
- [ ] B0/B1 smoke와 checkpoint reload가 통과한다.
- [ ] B0 embedding cache provenance가 완전하다.
- [ ] 40→20 stitching에 누락/중복이 없다.
- [ ] B0/B1 validation 결과가 동일 evaluator로 생성됐다.
- [ ] Macro-F1, kappa, per-class F1, confusion matrix, transition metric이 있다.
- [ ] majority-class baseline을 같은 validation set에 보고한다.
- [ ] test가 없으므로 결과를 validation-only/development로 표시한다.
- [ ] 실행하지 못한 GPU/full run은 실행한 것처럼 쓰지 않고 blocker와 명령을 남긴다.

## 15. baseline 이후 실험

B1 완료 후 한 번에 한 축만 바꾼다.

1. R1: noise augmentation only
2. R2: clean + two noisy views의 JS consistency
3. T1: 20분 B1
4. T2: 40분
5. T3: 1시간
6. T4: 2시간
7. T5: full-night
8. long-context gain 확인 후 hierarchical memory/cache
9. 실제 target domain 확보 후 home UDA

long-context 비교는 같은 frozen B0 embedding, split, seed, optimizer budget을 사용하고 Macro-F1, REM/Deep F1, transition/stable Macro-F1, transition-rate error를 보고한다.

## 16. agent의 최종 보고 형식

1. 생성·수정한 파일
2. 재사용한 read-only 데이터와 코드 근거
3. 실행한 명령과 결과
4. 통과·실패한 테스트
5. B0/B1 validation 비교 또는 실행 blocker
6. class order, split, data availability의 남은 위험
7. 다음 한 개 실험

구현 범위를 임의로 UDA, distillation, mobile deployment까지 넓히지 않는다.
