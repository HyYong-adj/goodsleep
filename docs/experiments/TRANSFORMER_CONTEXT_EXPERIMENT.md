# Transformer 40→20 / 80→20 실행 세팅

**준비일:** 2026-09-14  
**범위:** 인계서의 첫 단계 T40/T80. Conformer와 full-night 학습은 후속 단계.  
**Parent:** `artifacts/experiments/20260914T062144930455Z`  
**상태:** 정규 실험 COMPLETE. 결과와 후속 계획은 [2026-09-15 보고서](../PSG_TRANSFORMER_REPORT_20260915.md) 참조.  
**작업 위치:** `/home/sleep/researchers/choihy`; 공식 rXX owner 배정은 이 문서에서 추정하지 않는다.

## 1. 모델과 통제 조건

```text
동일 frozen B0 embedding [B,40 또는 80,192]
→ 입력 LayerNorm + 실제 recording epoch의 sinusoidal position
→ 2-layer Pre-LN Transformer (d192, heads4, FFN512, GELU)
→ final LayerNorm
→ 중앙20 positions
→ Dropout0.2 + Linear(192,4)
```

| 항목 | T40 | T80 |
| --- | --- | --- |
| 입력 / 중앙 출력 | 40→20, 좌우10 | 80→20, 좌우30 |
| 비교할 기존 모델 | R40 BiLSTM | R80 BiLSTM |
| 기존 validation Macro-F1 | 0.433995671 | 0.431861106 |
| trainable parameter | 694,148 | 동일 |
| Batch / dropout | 32 / 0.2 | 동일 |
| AdamW LR / weight decay | 3e-4 / 1e-4 | 동일 |
| 최대 epoch / patience | 50 / 10 | 동일 |
| 최대 update 상한 | 8,850 (177×50; 무효 batch skip 시 감소) | 동일 |
| Loss | train valid 고유 epoch의 count^-0.5를 평균 정규화한 weighted CE | 동일 |
| Seed / stride | 20260910 / 20 | 동일 |
| AMP / gradient clip | 활성 / 1.0 | 동일 |

설정: [T40](../../configs/t40_transformer.yaml), [T80](../../configs/t80_transformer.yaml).
두 설정의 차이는 context와 대응 comparator/alias뿐이다.
모델 초기화와 window shuffle seed를 T40/T80에 동일하게 적용한다.
위치 정보와 mask 방식도 포함한 모델 교체 비교이며 attention만 분리한 실험은 아니다.

B0 LR1e-4 실험은 완료됐다. B0 Macro-F1은0.366173이지만 downstream B1은0.395976으로,
기존 B1보다 낮았다. 이번 비교는 계획대로 기존 parent embedding을 사용한다.
새 B0 학습이나 embedding 생성은 수행하지 않는다.

## 2. GPU3 nohup 실행

검증한 Python은 `/venv/main/bin/python`, torch는 `2.14.0+cu130`이다.
새 라이브러리 설치는 필요하지 않다. 아래 명령이 정규 학습을 실제로 시작한다.

```bash
cd /home/sleep/researchers/choihy
mkdir -p artifacts/logs
PSG_RUN_ID="transformer_pair_$(date -u +%Y%m%dT%H%M%S%N)"

nohup env OMP_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /venv/main/bin/python -u scripts/run_transformer_pair.py \
  --run --gpu 3 \
  --output "artifacts/experiments/$PSG_RUN_ID" \
  > "artifacts/logs/$PSG_RUN_ID.log" 2>&1 < /dev/null &
echo $! > "artifacts/logs/$PSG_RUN_ID.pid"

tail -f "artifacts/logs/$PSG_RUN_ID.log"
```

`tail -f`는 Ctrl+C로 닫아도 학습이 계속된다.
실행기는 CUDA_VISIBLE_DEVICES=3을 설정하고 다음 순서를 따른다.

1. 설정·checkpoint·cache hash·class weights·평가 대상 검증.
2. GPU3 사용 상태 확인. 기존 compute process가 있으면 종료하며 해당 process를 중단하지 않는다.
3. 기존 BiLSTM R40/R80 checkpoint 재평가 및 저장 결과와 일치 검사.
4. T40 학습 → best checkpoint 재로드 → validation → R40 비교.
5. T80 학습 → best checkpoint 재로드 → validation → R80 비교.
6. T40/T80 context 차이와 architecture×context 차이의 비교표 저장.

출력 폴더가 이미 존재하면 덮어쓰지 않고 실패한다.
T40 실패 시 T80을 자동 진행하지 않는다. T80 실패 시 완료된 T40 산출물은 남는다.
현재 실행기는 checkpoint resume을 지원하지 않으므로 재실행은 새 run ID를 사용한다.

## 3. 학습 없이 준비 / 짧은 smoke

전체 데이터 preflight만 수행하며 GPU 학습을 시작하지 않는다.

```bash
/venv/main/bin/python scripts/run_transformer_pair.py
```

GPU3 smoke는 train8명/val2명, 각 모델 최대2 updates만 수행한다.

```bash
OMP_NUM_THREADS=4 /venv/main/bin/python scripts/run_transformer_pair.py \
  --smoke --run --gpu 3
```

CPU smoke도 `--gpu 3` 대신 `--cpu`로 가능하다.
Smoke 점수는 성능 비교에 사용하지 않는다.
Smoke의 baseline은 저장된 full-run predictions를 같은 val2명으로 제한한 참조이며,
정규 실행에서는 baseline checkpoint를 전체 validation에 다시 추론한다.

## 4. 무결성·수치 정밀도

- 전체 데이터: train191/val44 subjects, valid epochs108,543/23,621.
- Canonical 순서: Wake, REM, Light, Deep. Label remap은 기존 loader 계약을 따른다.
- 중앙 target20/stride20과 실제 recording 시간축을 유지하며 중간 결측을 압축하지 않는다.
- 입력 invalid는 attention key/value에서, target invalid는 loss/metric에서 제외한다.
- 입력이 전부 invalid인 row는 attention 전에 제외하여 NaN을 방지한다.
- 모든 비교의 `(subject_id, epoch_index, target)` 집합과 최종 prediction coverage를 검증한다.
- FP32 연산에서는 CUDA matmul/cuDNN TF32를 비활성화한다.
  기본 TF32로는 R40의23,621개 중1개 예측이 달랐으며,
  TF32 비활성화 및 CPU 추론에서는 R40/R80 저장 지표가 모두 재현됐다.
  이 설정을 새 실행기에 고정하고 환경·checkpoint config에 기록한다.
- Transformer attention은 짧은 window에 적합한 math SDPA를 사용한다.
  학습은 AMP이며 validation은 FP32다. 학습과 validation loss는 weighted CE 합/유효 weight 합으로 기록한다.
- 기존 checkpoint 선택은 validation Macro-F1이다.
  재로드한 best checkpoint 점수가 선택 시 점수와 다르면 실행을 실패 처리한다.

한 seed의 validation screening이며 통계적 우월성이나 home 일반화를 주장하지 않는다.
현재 runner는 기존 비교 checkpoint와 같은 seed를 요구한다.
추가 seed 비교는 해당 seed의 BiLSTM comparator를 확보한 뒤 별도 실험으로 확장한다.

## 5. 결과 위치

```text
artifacts/experiments/<PSG_RUN_ID>/
  experiment.json                 # 단계, 상태, 시작/종료, 오류
  T40.yaml, T80.yaml               # 경로를 확정한 설정
  preflight.json                  # provenance, support, 예산 상한
  environment.json                # Python/torch/CUDA/정밀도/code hash
  code/                           # 사용한 source snapshot
  tracked_changes.patch           # 기존 tracked dirty 변경
  R40/evaluation/                 # full 실행에서 baseline 재평가
  R80/evaluation/
  T40/
    best.pt
    resolved_config.json          # class weights/provenance 포함
    history.jsonl
    run_metadata.json             # parameter, GPU 메모리, 실제 시간/updates
    evaluation/results.json
    evaluation/predictions.csv
    evaluation/timing.json
    baseline_comparison.json
  T80/                            # T40과 같은 산출물
  comparison.json                 # R40/R80/T40/T80 및 차이
```

단계별 상태는 `PREPARING → PREPARED → RUNNING → COMPLETE`다.
Smoke는 `SMOKE_COMPLETE`, 오류는 `FAILED`, KeyboardInterrupt는 `STOPPED`로 기록한다.
OS가 process를 강제 종료하면 최종 상태를 기록하지 못할 수 있다.

별도 checkpoint 재평가:

```bash
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=4 /venv/main/bin/python scripts/evaluate.py \
  --checkpoint "artifacts/experiments/$PSG_RUN_ID/T40/best.pt" \
  --output "artifacts/experiments/$PSG_RUN_ID/T40/reevaluation"
```

## 6. 준비 검증 기록

- 전체 preflight: `transformer_pair_20260914T163707113792Z`, `PREPARED`.
- 첫 GPU3 smoke: `transformer_pair_20260914T163903738672Z`, `SMOKE_COMPLETE`.
- 최종 정밀도 설정의 GPU3 smoke: `transformer_pair_20260914T164453041708Z`, `SMOKE_COMPLETE`.
- 관련25개 테스트 및 전체39개 테스트 통과.
- GPU3에서 TF32 비활성화 후 기존 R40/R80의 저장 Macro-F1 재현.
- Smoke에서694,148 parameters, peak allocated GPU memory 약101MiB(T40)/140MiB(T80).
  CUDA 전체 점유량이나 full 학습 시간의 측정값으로 해석하지 않는다.

구현: [모델](../../src/psg_only/transformer_model.py),
[학습·dataset·평가](../../src/psg_only/transformer.py),
[순차 실행기](../../scripts/run_transformer_pair.py),
[테스트](../../tests/test_transformer.py).
