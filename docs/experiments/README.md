# 실험 계획 및 실행 가이드

실험 설정과 실행 명령을 모은 폴더다. 결과 보고서는 상위 `docs/` 폴더에 유지한다.
각 문서의 실행 명령은 저장소 루트(`/home/sleep/researchers/choihy`) 기준이다.

| 문서 | 내용 |
| --- | --- |
| [PSG 실행 가이드](PSG_EXPERIMENT_QUICKSTART.md) | 환경 준비 및 B0 → embedding → B1 실행 |
| [B0 학습률 실험](B0_LR1E4_EXPERIMENT.md) | B0 learning rate 3e-4 → 1e-4 |
| [B1 80→20 실험](B1_80TO20_EXPERIMENT.md) | BiLSTM context 확장 |
| [Transformer context 실험](TRANSFORMER_CONTEXT_EXPERIMENT.md) | T40 / T80 비교 |
| [Conformer epoch 실험](CONFORMER_EPOCH_EXPERIMENT.md) | Conformer encoder + BiLSTM40 |
| [Phase 1 부가 요소 이식](PHASE1_EMBEDDING_OPTIONS.md) | 동결 embedding 위 A0–A5 |

## 결과 보고서

- [PSG 기본 실험](../PSG_EXPERIMENT_REPORT_20260914.md)
- [B1 80→20 결과](../PSG_B1_80TO20_REPORT_20260914.md)
- [Transformer 결과 및 후속 계획](../PSG_TRANSFORMER_REPORT_20260915.md)
- [Conformer 결과 및 후속 검증 방향](../PSG_CONFORMER_REPORT_20260916.md)
- [Phase 1 결과 — 동결 embedding 부가 요소 (음성)](../PSG_PHASE1_REPORT_20260916.md)
- [Phase 2–5 누적 결과](../PSG_PHASE3_5_REPORT_20260918.md) — **현재 최고 0.5310 (end-to-end)**
- [다음 실험 방향 제언 (rev.2)](../PSG_RESEARCH_DIRECTION_20260916.md) — **현재 방향 기준 문서.** 트랙 위치, byoungjun 트랙 대조 및 성능 격차 분석, REM 데이터 제약 감사, 재실행 금지 목록
