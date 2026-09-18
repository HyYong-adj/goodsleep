# B0 learning rate 3e-4 → 1e-4 실험

## 목적과 통제 조건

Parent: `artifacts/experiments/20260914T062144930455Z`.
B0 LR만 0.0003에서 0.0001로 변경한다. 새 B0 best checkpoint로 embedding을 새 실행 폴더에 생성하고,
B1 40→20을 동일한 설정으로 다시 학습한다. 기존 B0/B1 학습 결과나 embedding을 덮어쓰지 않는다.

| 항목 | 이번 설정 |
| --- | --- |
| B0 LR | 0.0001 |
| B0 최대 epochs / patience | 30 / 10 |
| B0 batch / dropout | 16 / 0.2 |
| B1 context | 40→20, stride20, 좌우10 |
| B1 LR / 최대 epochs / patience | 0.0003 / 50 / 10 |
| Seed | 20260910 |
| Split / normalization | Parent manifest와 기존 train stats 고정 |
| Loss / optimizer | 기존 class-weighted CE / AdamW |

가설: 더 낮은 B0 LR이 acoustic representation과 최종 B1 validation Macro-F1을 개선한다.
Baseline은 B0 Macro-F1 0.363094, B1 Macro-F1 0.433996이다.
Macro-F1을 주 지표로 κ, REM/Deep precision·recall·F1, transition Macro-F1을 함께 확인한다.
B0만 개선된 경우와 최종 B1까지 개선된 경우를 구분한다.
성공 후보는 추가 seed로 재현성을 확인한다. test가 없으므로 validation-only 개발 결과다.

## GPU 3 nohup 명령

현재 활성화된 conda 환경의 `python`을 사용한다.

```bash
cd /home/sleep/researchers/choihy
mkdir -p artifacts/logs
PSG_RUN_ID="b0_lr1e4_$(date -u +%Y%m%dT%H%M%S)"
nohup env OMP_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  python -u scripts/run_experiment.py \
  --source cached \
  --b0-config configs/b0_lr1e4.yaml \
  --parent-run artifacts/experiments/20260914T062144930455Z \
  --run --gpu 3 \
  --output "artifacts/experiments/$PSG_RUN_ID" \
  > "artifacts/logs/$PSG_RUN_ID.log" 2>&1 < /dev/null &
echo $! > "artifacts/logs/$PSG_RUN_ID.pid"
echo "Run: $PSG_RUN_ID"
tail -f "artifacts/logs/$PSG_RUN_ID.log"
```

실행기가 각 학습·평가 subprocess에 `CUDA_VISIBLE_DEVICES=3`을 설정한다.
순서: preflight → B0 학습/평가 → 새 embedding → B1 학습/평가 → parent 비교.
`tail -f`를 Ctrl+C로 닫아도 nohup 학습은 계속된다.

## 기록과 판정

실행 폴더 `artifacts/experiments/<Run ID>/`:

- `experiment.json`: parent, 변경 LR, seed, 예산, 단계와 COMPLETE/FAILED 상태
- `b0.yaml`, `b1.yaml`: resolved 설정
- `b0/history.jsonl`, `b1/history.jsonl`: epoch별 train_loss, validation_loss, validation_macro_f1
- `b0/evaluation/`, `b1/evaluation/`: 모델별 지표·prediction
- `comparison.json`: 이번 run 내부 B0/B1 비교
- `parent_comparison.json`: 첫 full run과 새 run의 B0/B1 각각 비교 (full만 생성)

Loss는 단순 batch 평균이 아니라 weighted CE 합을 valid target weight 합으로 나눈 값이다.
Train loss는 train mode에서 update 전 각 batch loss를 누적하므로 학습 중 서로 다른 파라미터 상태의 평균이다.
Validation loss는 epoch 종료 모델의 eval mode에서 계산한다. Dropout/BatchNorm 모드 차이가 있어 두 loss의
절대값 차이만으로 과적합을 판정하지 않는다. 추세와 validation Macro-F1을 함께 해석한다.
Checkpoint 선택 기준은 validation Macro-F1이며 loss는 진단 기록에만 추가했다.

Parent에는 train/validation loss가 기록되지 않았으므로 과거 loss와의 수치 비교는 불가능하다.
낮은 LR에서는 수렴이 느릴 수 있으므로 동일 최대 epoch/patience 아래의 비교라는 한계를 남긴다.
더 많은 epochs나 scheduler는 동시에 추가하지 않는다.

## 준비 및 smoke 명령

```bash
python scripts/run_experiment.py --source cached \
  --b0-config configs/b0_lr1e4.yaml \
  --parent-run artifacts/experiments/20260914T062144930455Z

python scripts/run_experiment.py --source cached \
  --b0-config configs/b0_lr1e4.yaml \
  --parent-run artifacts/experiments/20260914T062144930455Z \
  --smoke --run --cpu
```

준비 과정에서는 실제 GPU full 학습을 시작하지 않는다. Smoke는 모델별 최대 2 updates로 동작만 확인한다.

## 최종 결과 및 해석

Run: `b0_lr1e4_20260914T154714`, 상태 **COMPLETE**, seed 20260910.
동일 validation epoch/target 비교를 통과했다. 평가 범위는 validation-only다.

**결론: B0 Macro-F1은 소폭 상승했지만 최종 B1 성능은 하락했다. LR 1e-4를 새 baseline으로 채택하지 않고 기존 B0 LR 3e-4 + B1 40→20을 유지한다.**

### 전체 지표

| 모델 | 지표 | 기존 | LR 변경 후 | 차이 |
| --- | --- | --- | --- | --- |
| B0 | accuracy | 0.747555 | 0.793023 | +0.045468 |
| B0 | macro_f1 | 0.363094 | 0.366173 | +0.003079 |
| B0 | cohen_kappa | 0.216907 | 0.229551 | +0.012644 |
| B1 | accuracy | 0.767029 | 0.699928 | -0.067101 |
| B1 | macro_f1 | 0.433996 | 0.395976 | -0.038019 |
| B1 | cohen_kappa | 0.300808 | 0.224257 | -0.076551 |

Accuracy는 0–1 단위다. B0 Accuracy는 상승했으나 B1 Macro-F1은 **0.433996→0.395976 (−0.038019)**, κ는 **0.300808→0.224257**로 하락했다.

### Class별 F1

| 모델 | Class | 기존 F1 | 새 F1 | 차이 |
| --- | --- | --- | --- | --- |
| B0 | Wake | 0.463367 | 0.464879 | +0.001512 |
| B0 | REM | 0.044173 | 0.004862 | -0.039311 |
| B0 | Light | 0.853880 | 0.882600 | +0.028720 |
| B0 | Deep | 0.090956 | 0.112349 | +0.021393 |
| B1 | Wake | 0.562986 | 0.482272 | -0.080714 |
| B1 | REM | 0.097470 | 0.089181 | -0.008288 |
| B1 | Light | 0.863607 | 0.817201 | -0.046406 |
| B1 | Deep | 0.211921 | 0.195251 | -0.016670 |

**B0 해석:** 전체 Macro-F1의 작은 증가가 모든 class의 개선을 뜻하지 않는다. REM F1은 0.044173→0.004862로 크게 낮아졌다. 실제 REM 881개 중 3개만 맞혔고 834개(94.67%)를 Light로 예측했다. Light F1 상승이 REM 악화를 가리는 형태다.

**B1 해석:** 네 class의 F1이 모두 하락했다. REM 정답 수는 104→122, Deep은 256→296으로 늘었지만 precision은 각각 0.0830→0.0658, 0.2984→0.2008로 감소했다. 소수 class recall 증가를 오탐 증가가 상쇄했다. 실제 Light를 REM으로 예측한 수는 1,064→1,576, Deep으로 예측한 수는 595→1,155로 늘었다.

새 B0의 single-epoch F1 상승이 downstream embedding 품질 향상으로 이어진다고 볼 수 없다. 다만 단일 seed의 결과만으로 embedding 손상의 원인이나 일반적인 LR 우열을 확정하지 않는다.

### B1 temporal 지표

| 지표 | 기존 | 새 B1 |
| --- | --- | --- |
| transition_macro_f1 | 0.379130 | 0.384420 |
| stable_macro_f1 | 0.431280 | 0.387490 |
| true_transition_rate | 0.052617 | 0.052617 |
| predicted_transition_rate | 0.058020 | 0.073093 |
| transition_rate_error | 0.017948 | 0.030408 |

Transition Macro-F1은 소폭 상승했지만 stable Macro-F1은 하락했고, 전환율 오차도 0.017948→0.030408로 증가했다. 경계 구간의 일부 개선만으로 전체 성능 하락을 상쇄하지 못했다. 전환율 오차는 subject별 절대 오차의 평균이다.

### 학습 및 loss 추세

| 모델 | 완료 epochs | Best epoch (1-based) | 총 updates | 첫 train loss → 마지막 | 첫 val loss → 마지막 |
| --- | --- | --- | --- | --- | --- |
| B0 | 13 | 3 | 90155 | 1.0682 → 0.6354 | 1.1802 → 1.9506 |
| B1 | 18 | 8 | 3186 | 0.8219 → 0.3554 | 1.1861 → 1.8027 |

B0는 3번째 epoch, B1은 8번째 epoch가 best였고 각각 이후 10 epochs 동안 개선되지 않아 종료됐다. 두 모델 모두 train loss는 낮아지지만 validation loss는 전반적으로 높아지는 양상이다. 이는 **일반화 격차 확대·과적합과 일치하는 신호**이며, 낮은 LR만으로 문제가 해결되지 않았음을 보여준다.

Train/eval 모드 차이가 있으므로 loss 차이만으로 원인을 확정하지 않는다. Weighted CE와 Macro-F1은 목적이 달라 최소 validation loss epoch와 최고 Macro-F1 epoch가 달라도 모순은 아니다. Parent에는 loss 기록이 없어 이전 LR보다 과적합이 더 심해졌다고 직접 비교할 수는 없다.

### 판단과 다음 방향

- 기존 B0 LR 3e-4 + B1 40→20을 parent로 유지한다. 이번 run은 정상 완료된 성능 개선 미확인 결과로 보존한다.
- LR를 더 낮추거나 context를 계속 늘리기보다, 기존 LR에서 augmentation 또는 regularization을 한 축으로 검증하는 방향을 고려한다.
- 새 가설 채택 전 subject별 오류와 REM/Deep precision·recall을 함께 확인한다. 개선 후보는 추가 seed로 검증한다.
- 본 문서 갱신에서는 추가 학습을 실행하지 않았다. 단일 seed validation 결과이며 독립 test 결론은 아니다.

### 근거 파일

- [완료 상태](../../artifacts/experiments/b0_lr1e4_20260914T154714/experiment.json)
- [Parent 비교](../../artifacts/experiments/b0_lr1e4_20260914T154714/parent_comparison.json)
- [B0 loss 이력](../../artifacts/experiments/b0_lr1e4_20260914T154714/b0/history.jsonl)
- [B1 loss 이력](../../artifacts/experiments/b0_lr1e4_20260914T154714/b1/history.jsonl)
