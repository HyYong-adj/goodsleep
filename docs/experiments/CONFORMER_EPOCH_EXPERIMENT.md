# Conformer epoch encoder + BiLSTM40 실험

**작성일:** 2026-09-15  
**상태:** 정규 학습 COMPLETE (2026-09-15). 결과 확인·문서 갱신: 2026-09-16.  
**Parent:** `artifacts/experiments/20260914T062144930455Z`  
**작업 위치:** `/home/sleep/researchers/choihy`  
**관련 결과:** [Transformer 결과 및 후속 계획](../PSG_TRANSFORMER_REPORT_20260915.md)

## 정규 실험 결과

Run `conformer_epoch_20260915T054210274368058`은 GPU3에서 정상 완료됐다.
동일 validation 44 subjects·23,621 유효 epochs 기준 C1 Macro-F1은 **0.412662**
(기존 B0 대비 +0.049568), downstream BiLSTM40은 **0.483261** (기존 대비 +0.049266)이다.
REM 개선이 가장 크지만 Deep recall은 17.39%로 낮다. 단일 seed 개발 결과이며 우위 확정은 아니다.
실제 학습 이력·클래스별 지표·한계는 [Conformer 결과 보고서](../PSG_CONFORMER_REPORT_20260916.md)를 따른다.
아래 명령은 완료된 실행을 재개하는 명령이 아니라 새 run을 생성하는 명령이다.

## 실험 설계

C1에서 음향 encoder를 학습하고 freeze한 뒤, 새 embedding 위에 기존 BiLSTM40을 학습한다.
Conformer는30초 내부, BiLSTM은epoch 사이 문맥을 담당한다.

| 구성 | 시작 설정 |
| --- | --- |
| 입력 | 기존 정규화 log-Mel [B,1,48,1499] |
| CNN | 3×3/stride2/padding1 Conv + BN + SiLU, channels32→64→96 |
| Subsampling | [B,96,6,188] → [B,188,576] → Linear(576,192) |
| Conformer | 2 blocks, d192, heads4, FFN768, conv kernel31, dropout0.2 |
| Position / pooling | sinusoidal absolute position / mean pooling |
| Head | Dropout0.2 + Linear(192,4) |
| C1 optimizer | AdamW LR3e-4, weight decay1e-4 |
| C1 budget | batch16, 최대30 epochs, patience10, AMP, clip1.0 |
| B1 구조 | 기존 LayerNorm + 2-layer BiLSTM(hidden128/방향) + Linear4 |
| B1 context | 40→20, 좌우10, stride20 |
| B1 optimizer/budget | AdamW LR3e-4, weight decay1e-4, batch32, 최대50 epochs, patience10 |
| Loss / seed | 기존 count^-0.5 평균 정규화 weighted CE / 20260910 |

비교 parent는 기존 CNN B0 Macro-F1 0.363094 / BiLSTM40 Macro-F1 0.433996이다.
LR1e-4 B0 실험이나 Transformer를 이번 모델에 섞지 않는다.
Macro-F1을 주지표로 κ·REM/Deep recall/F1·transition 지표를 함께 평가한다.
처음에는1-seed validation screening이며 공식 owner ID나 stage-gate 승인을 추정하지 않는다.

## GPU3 nohup 실행

검증 환경은 `/venv/main/bin/python`, torch2.14.0+cu130이다. 추가 라이브러리 설치는 없다.
아래 명령은 GPU3이 사용 중이면30초 간격으로 기다린 뒤 정규 학습을 시작한다.

```bash
cd /home/sleep/researchers/choihy
mkdir -p artifacts/logs
PSG_RUN_ID="conformer_epoch_$(date -u +%Y%m%dT%H%M%S%N)"

nohup env OMP_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /venv/main/bin/python -u scripts/run_conformer_experiment.py \
  --run --gpu 3 --wait-for-gpu \
  --output "artifacts/experiments/$PSG_RUN_ID" \
  > "artifacts/logs/$PSG_RUN_ID.log" 2>&1 < /dev/null &
echo $! > "artifacts/logs/$PSG_RUN_ID.pid"

tail -f "artifacts/logs/$PSG_RUN_ID.log"
```

`tail -f`는 Ctrl+C로 닫아도 작업이 계속된다.
대기 중에는 로그와 experiment.json에 `WAITING_FOR_GPU`가 표시된다.
대기 기능은 기존 process를 종료하지 않으며, GPU 예약 scheduler를 대신하지 않는다.
대기하지 않고 사용 중이면 즉시 종료하려면 `--wait-for-gpu`를 제외한다.

실행 순서:

1. Parent checkpoint·prediction·manifest·train stats와 C1/B1 학습 조건을 검증한다.
2. GPU가 비기를 기다리고, 대기 후 data/hash를 다시 확인한다.
   준비/대기 중 source가 바뀌면 새 run을 요구하며 버전이 섞인 학습을 시작하지 않는다.
3. C1 single-epoch 학습·best checkpoint 평가.
4. C1을 freeze/eval로 고정하여 새192차원 embedding cache 생성.
5. 같은 BiLSTM40을 새 cache로 학습·평가.
6. Single-epoch와 downstream B1의 parent 비교를 각각 저장한다.

C1이 단독으로 parent보다 낮아도 첫 screening에서는 B1까지 진행한다.
기술적 실패나 무결성 오류에서는 중단하고 FAILED 사유를 기록한다.
자동 resume은 지원하지 않으며 재실행은 새 run ID를 사용한다.
기존 B0/B1 checkpoint와 embedding은 덮어쓰지 않는다.

## 학습 없이 preflight / smoke

전체 데이터의 입력·비교 조건만 확인:

```bash
/venv/main/bin/python scripts/run_conformer_experiment.py
```

CPU smoke:

```bash
OMP_NUM_THREADS=4 /venv/main/bin/python scripts/run_conformer_experiment.py \
  --smoke --run --cpu
```

GPU3 smoke를 따로 실행할 경우:

```bash
OMP_NUM_THREADS=4 /venv/main/bin/python scripts/run_conformer_experiment.py \
  --smoke --run --gpu 3 --wait-for-gpu
```

Smoke는 train8명/val2명, C1/B1 각각 최대2 updates이며,
선택한 subjects의 embedding을 실제로 생성한다. Smoke 점수는 성능 비교용이 아니다.
구현 준비 시 GPU3이 점유되어 **GPU smoke는 실행하지 않았고 CPU end-to-end smoke를 검증했다**.

## 산출물과 판정

```text
artifacts/experiments/<PSG_RUN_ID>/
  experiment.json            # PREPARING/PREPARED/WAITING_FOR_GPU/RUNNING/COMPLETE 등
  experiment.yaml            # 전체 실험 설정
  preflight.json             # 데이터·parent 출처·비교 무결성
  environment.json           # 환경, 정밀도, source hash
  code/                      # 해당 실행의 source snapshot
  tracked_changes.patch
  c1.yaml
  c1/best.pt
  c1/history.jsonl
  c1/run_metadata.json       # parameters, 실제 시간·updates, GPU 메모리
  c1/evaluation/
  embeddings/<subject_id>/   # embeddings/labels/valid.npy + metadata.json
  cache_metadata.json
  b1.yaml
  b1/best.pt
  b1/history.jsonl
  b1/evaluation/
  comparison.json
```

`comparison.json`의 `single_epoch`와 `bilstm40`에 parent/candidate 지표와
Macro-F1 차이가 각각 저장된다. Parent는 저장 prediction의 checksum·checkpoint 출처를 검증하고
현재 evaluator로 다시 집계한 값이다. Parent CNN 전체 재학습이나 재추론을 자동 수행하지 않는다.

모든 정규 비교는 같은 validation44명·23,621 valid epochs를 평가한다.
Canonical label 순서, train-only 정규화·class weights, weighted CE, ignore_index=-100을 유지한다.
Cache는 원래 recording 시간축의 결측을 보존하며, metadata에 encoder_type과 checkpoint hash를 기록한다.
후속 B1은 checkpoint hash가 가리키는 C1 cache를 사용하므로 기존 CNN cache와 섞이지 않는다.

FP32 연산에서 TF32를 비활성화하고 Conformer attention은 math SDPA를 사용한다.
C1/B1 학습은 AMP, validation·cache 생성은 FP32 모델 추론이며 embedding은float16으로 저장한다.
C1과 B1의 checkpoint 선택은 validation Macro-F1이다. Loss는 weighted CE 합/valid weight 합으로 집계한다.

C1 budget은 최대30 epochs, 최대208,050 updates(6,935×30)이다.
B1 budget은 최대50 epochs이며 기존과 같은 target20/stride20을 사용한다.
실제 epoch 수는 early stopping으로 달라진다.
C1은 Mel부터 학습하므로 이전 cached Transformer의 수분 실행 시간을 예상 소요 시간으로 쓰지 않는다.
정규 pipeline 시작→완료는 약67분41초였다(평가·cache 생성 등 포함).
C1 학습+validation은 약61분52초이며 B1 개별 학습 시간은 기록되지 않았다.

## 검증 기록

- 전체 preflight: `conformer_epoch_20260915T052926549662Z`, PREPARED.
- 실제 데이터 CPU smoke: `conformer_epoch_20260915T053115217987Z`, SMOKE_COMPLETE.
- C1/B1 각각2 updates, val2명·926 valid epochs의 평가 대상 일치.
- Conformer 학습 가능 parameter 수: **1,905,124**.
- Shape/subsampling/gradient/checkpoint/cache/결측/BiLSTM 연결과 기존 CNN cache 호환성 검증.
- GPU 대기 및 source 변경 감지 테스트 포함, **전체44개 테스트 통과**.
- 정규 학습 및 GPU smoke는 최초 준비 작업 중 실행하지 않았다. 이후 정규 GPU3 실행은 위 run으로 완료됐다.

설정: [c1_conformer_epoch.yaml](../../configs/c1_conformer_epoch.yaml).
구현: [Conformer 모델](../../src/psg_only/conformer.py),
[실행기](../../scripts/run_conformer_experiment.py),
[cache 생성](../../scripts/cache_embeddings.py),
[테스트](../../tests/test_conformer.py).
