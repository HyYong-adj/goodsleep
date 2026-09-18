# B1 context 20분 → 40분 실험

상태: 설정·전체 embedding preflight·CPU 2-step smoke 완료. Full GPU 실험은 아직 시작하지 않았다.

## 가설과 비교 조건

같은 frozen B0 embedding에서 context를 늘리면 REM/Deep 구분이 개선되는지 확인한다.
Parent: `artifacts/experiments/20260914T062144930455Z`의 B1 40→20.
Parent validation Macro-F1 0.433996, κ 0.300808, REM F1 0.097470, Deep F1 0.211921.

| 항목 | Parent | 이번 실험 |
| --- | --- | --- |
| Input | 40 epochs / 20분 | 80 epochs / 40분 |
| Target | 중앙 20 epochs | 중앙 20 epochs |
| 좌/우 context | 각각 10 | 각각 30 |
| Output slice | [10:30] | [30:50] |
| Stride | 20 | 20 |
| Seed | 20260910 | 동일 |
| Embedding | Parent B0 | 동일 cache 재사용 |
| Model | hidden128, layers2, dropout0.2 | 동일 |
| Training | AdamW, LR0.0003, batch32, weight decay0.0001 | 동일 |
| 최대 budget | 50 epochs, patience10 | 동일 |

전체 preflight: train 191명 / 5,640 windows / 108,543 valid epochs,
val 44명 / 1,225 windows / 23,621 valid epochs.
Epoch당 최대 177 updates, 최대 8,850 updates이며 early stopping에 따른 실제 update 수는 달라질 수 있다.
Context를 늘리므로 동일 update 수라도 GPU 시간은 같지 않다.

## GPU 3 nohup 실행

```bash
cd /home/sleep/researchers/choihy
mkdir -p artifacts/logs
PSG_RUN_ID="b1_80to20_$(date -u +%Y%m%dT%H%M%S)"
nohup env CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /venv/main/bin/python -u scripts/run_b1_context.py \
  --config configs/b1_80to20.yaml \
  --output "artifacts/experiments/$PSG_RUN_ID" \
  > "artifacts/logs/$PSG_RUN_ID.log" 2>&1 < /dev/null &
echo $! > "artifacts/logs/$PSG_RUN_ID.pid"
echo "Run: $PSG_RUN_ID"
```

로그 확인(같은 shell):

```bash
tail -f "artifacts/logs/$PSG_RUN_ID.log"
```

새 shell에서는 위에 출력된 Run ID로 파일명을 지정한다.
`CUDA_VISIBLE_DEVICES=3`으로 물리 GPU 3만 노출하므로 프로세스 내부의 `cuda:0` 표기는 정상이다.
이 명령은 B0 학습이나 embedding 생성을 다시 수행하지 않는다.

## 산출물과 판정

- `experiment.json`: 사전 실험 가설, 단계, COMPLETE/FAILED 상태
- `b1.yaml`, `preflight.json`: 설정과 parent provenance 검증
- `b1/best.pt`, `b1/history.jsonl`: best checkpoint, 학습 이력
- `b1/evaluation/results.json`, `predictions.csv`: 새 B1 결과
- `comparison.json`: parent B1와 동일 epoch/target 확인 및 metric 비교

주요 지표는 Macro-F1이고 κ, REM/Deep precision·recall·F1, transition/stable Macro-F1,
transition-rate error를 함께 본다. 한 seed의 결과로 확정하지 않고 후보가 개선되면 paired seed 반복으로 확인한다.
평가 범위는 validation-only이며 test 성능 주장은 하지 않는다.

## 검증 근거

- [전체 embedding preflight](../../artifacts/b1_80to20_preflight/preflight.json)
- [CPU smoke 완료](../../artifacts/b1_80to20_smoke/experiment.json): train8/val2, 2 updates, validation926 epochs
- CPU smoke Macro-F1 0.265051은 실행 검증용이며 full parent 성능과 비교하지 않는다.
- 40/80 context 경계 coverage, 중앙 target 위치, 80 checkpoint 재로딩 및 잘못된 설정 검사 추가.
- 기존 checkpoint에 context 설정이 없으면 40→20 기본값을 사용한다.
