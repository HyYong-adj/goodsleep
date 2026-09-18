#!/usr/bin/env bash
# Phase 2 완료를 기다렸다가 결과를 집계하고, 이어서 Phase 3(end-to-end)를 돌린다.
# Phase 3 가 실패해도 Phase 2 집계는 이미 저장돼 있다.
set -u
cd /home/sleep/researchers/choihy
export OMP_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=/venv/main/bin/python
P2=artifacts/experiments/phase2_20260916
P3=artifacts/experiments/phase3_20260916
LOG=artifacts/logs

echo "[chain $(date -u +%FT%TZ)] Phase 2 완료 대기"
while true; do
  status=$($PY -c "import json;print(json.load(open('$P2/phase2.json'))['status'])" 2>/dev/null || echo PENDING)
  [ "$status" = "RUNNING" ] || { echo "[chain] Phase 2 status=$status"; break; }
  sleep 120
done

echo "[chain $(date -u +%FT%TZ)] Phase 2 집계"
$PY scripts/analyze_phase2.py "$P2" > docs/PSG_PHASE2_REPORT_20260916.md 2>>"$LOG/chain.log" \
  && echo "[chain] wrote docs/PSG_PHASE2_REPORT_20260916.md" \
  || echo "[chain] WARNING: Phase 2 집계 실패"

echo "[chain $(date -u +%FT%TZ)] Phase 3 시작 (end-to-end, night_norm 없음 — C0 대비 단일 변수)"
CUDA_VISIBLE_DEVICES=2 $PY -u scripts/run_phase3.py \
  --seed 20260910 --epochs 25 --output "$P3" > "$LOG/phase3_20260916.log" 2>&1
code=$?
echo "[chain $(date -u +%FT%TZ)] Phase 3 exit=$code"

if [ -f "$P3/evaluation/results.json" ]; then
  $PY - <<'PY' >> docs/PSG_PHASE2_REPORT_20260916.md
import json
from pathlib import Path
d = json.loads(Path('artifacts/experiments/phase3_20260916/evaluation/results.json').read_text())
m = d['metrics']
print('\n---\n')
print('# Phase 3 결과 — end-to-end 공동 학습\n')
print('구조는 parent 와 같고 **학습 방식만** 바꿨다(동결 2단계 -> 공동 학습).')
print('비교 대상은 같은 GPU 의 Phase 2 C0 대조군이다.\n')
print(f"- Macro-F1 **{m['macro_f1']:.4f}** / κ {m['cohen_kappa']:.4f} / Acc {m['accuracy']:.4f}")
print(f"- parent B1(0.483261) 대비 {d['delta_vs_parent_b1']:+.4f}  *(참고용 — GPU 가 다르므로 C0 대비로 판단할 것)*")
print(f"- 파라미터 {d['trainable_parameters']:,} / {d['epochs_completed']} epochs / "
      f"{d['training_seconds']/3600:.2f} h / peak {d['peak_cuda_memory_allocated_bytes']/2**20:.0f} MiB")
print('\n| 클래스 | F1 | recall |')
print('| --- | ---: | ---: |')
for name, v in m['per_class'].items():
    print(f"| {name} | {v['f1']:.4f} | {v['recall']:.4f} |")
try:
    c0 = next(r for r in json.loads(Path('artifacts/experiments/phase2_20260916/phase2.json').read_text())['records'] if r['variant']=='C0')
    print(f"\n**C0 대비 ΔMacro-F1 = {m['macro_f1']-c0['b1_macro_f1']:+.4f}** (C0 {c0['b1_macro_f1']:.4f})")
except Exception as e:
    print(f"\n(C0 대조군을 읽지 못함: {e})")
print('\n단일 seed, validation-only. 채택 판단 전 3 seed 확장 필요.')
PY
  echo "[chain] Phase 3 요약을 보고서에 추가"
else
  echo "[chain] Phase 3 결과 없음 — $LOG/phase3_20260916.log 확인" 
fi
echo "[chain $(date -u +%FT%TZ)] 완료"
