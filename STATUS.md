# choihy Status

**Updated:** 2026-09-18
**Stage / lane:** Foundation / Conformer epoch encoder — 음향 표현 축 (pbj 분담안 기준 "Conformer/T40" 레인)
**Current parent:** **choihy-e012** end-to-end Conformer+BiLSTM (SpecAugment + cosine), validation Macro-F1 **.5310** — 단일 seed, 재현 미완

> 전부 validation 44명 / 23,621 유효 epoch, best epoch 기준. `test_locked` 49명 미접근. 공용 evaluator 미연동.
> 본 트랙은 3 seed 까지만 돌았다. cloud9sm 이 쓰는 10-seed 게이트(+.010 & 8/10)에는 미달하며, 승격 주장 아님.

## Current conclusion

- **입력은 PSG 중 동기 녹음된 환경 마이크(EDF 채널 18) 오디오뿐이고, PSG 신호는 라벨만 제공한다.** 30초 epoch 마다 Wake/REM/Light(N1+N2)/Deep(N3) 4-class 를 예측한다. Light 가 81% 라 전부 Light 로 찍어도 Accuracy .814 (Macro-F1 .224) 가 나오므로 Macro-F1 으로만 판단한다.
- **현재 최고 .5310** (부풀림 보정 .5128). 같은 GPU 대조군 .4716 대비 **+.059**. 단 **단일 seed** 이고 cloud9sm 의 Mic-only 최고(E033 .5230 ± .0098, 10 seed)와 사실상 같은 구간이다. **본 트랙이 앞선다고 주장할 근거는 없다.**
- **효과가 확인된 레버는 셋뿐이다:** end-to-end 공동 학습 **+.024**, SpecAugment **+.024**, gated KD **+.010**.
- **동결 2단계가 본 트랙의 자기 발목이었다.** `encoder 학습 → 동결 → embedding 캐시 → temporal 학습` 구조에서 encoder 는 "고립된 30초 분류"에 유용한 특징만 배우고 멈춘다. 공동 학습으로 바꾸자 동결 경로 최고(.5071)를 +.024 앞섰고 Deep F1 이 .265 → **.399** 로 올랐다. **이는 팀 차원의 새 발견이 아니다** — byoungjun 파이프라인은 처음부터 end-to-end 였다. 본 트랙만 뒤늦게 따라잡은 것이다.
- **temporal 축은 본 트랙에서도 소진됐다.** 문맥 40→80 epoch(−.002), BiLSTM→Transformer(−.027/−.023) 모두 기각. bj-e037(full-night GRU 3 seed, causal .517 = bidirectional .517)·cloud9sm 의 독립 결론과 일치한다.
- **무효 또는 유해로 판정한 것:** night_norm(부풀림 보정 후 소멸), 시각 특징(−.009, 3 seed 전부 음수), label smoothing, REM class weight ×2, cosine LR 단독, 문맥 확장, Transformer, B0 학습률 1e-4.
- **목표 대비:** Macro-F1 .531 / κ .395 / Acc .799 vs 목표 .60 / .50 / .60. 남은 격차는 데이터(REM 커버리지) 문제로 판단한다.

## New evidence

전체 계보. `test_locked` 미사용, 모두 validation-only.

| Experiment | Result | Comparator | Verdict |
|---|---|---|---|
| choihy-e001 B0 CNN 단일 epoch | .3631 | majority(Light) .2243 | COMPLETE. 파이프라인·라벨 정렬 검증 |
| choihy-e002 B1 CNN + BiLSTM40 (20분 문맥) | **.4340**, κ .3008 | e001 .3631 | COMPLETE. 문맥 효과 확인 +.071 |
| choihy-e003 B0 학습률 3e-4 → 1e-4 | B0 .3662 이나 downstream B1 **.3960** | e002 .4340 | **기각.** encoder↑ / downstream↓ 탈동조 첫 사례 |
| choihy-e004 문맥 40 → 80 epoch (40분) | .4319, REM/Deep 모두 하락 | e002 .4340 | **기각.** 문맥 2배로 개선 없음 |
| choihy-e005 BiLSTM → Transformer 40 / 80 | .4069 / .4086, Transformer80 Deep recall 6.6% | e002 .4340 | **기각.** temporal 교체 무효 |
| choihy-e006 CNN → **Conformer** encoder 교체 | 단독 .4127, downstream **.4833** | e002 .4340 | COMPLETE. +.049, 당시 최고. 병목이 음향 표현임을 지지 |
| choihy-e007 원본 RML 라벨 감사 (CPU) | V3 287명: REM **4.77%**, **102명(35.5%) 밤 전체 REM 0**, 판독 중앙값 4.45 h. 매핑 코드는 정상, MachineStaging 교차확인 | — | COMPLETE. **데이터 특성 확정.** byoungjun 라벨 감사와 독립 일치 |
| choihy-e008 동결 embedding 위 시각특징/cosine/smoothing/REM가중 (A1–A5, 3 seed) | 결합 A5 **−.0098** (3 seed 전부 음수), 시각특징 단독 −.0089 | A0 .4817 | **전부 기각.** 단 참조 combo 와 겹치는 요소가 하나뿐이라 combo 검증이 아니었음 |
| choihy-e009 encoder 단계 (C0–C4, 1 seed) | **C0 대조군 .4716** / C2 night_norm .4952(보정 .4652) / **C3 SpecAugment .5052**(보정 .4939) / C4 결합 .4912 (REM .197→**.147**) | C0 | C3 채택. **C2·C4 기각.** C4 는 사전 등록 기준(REM 미하락) 미달 |
| choihy-e010 **gated KD** + cosine (D0–D3, 3 seed) | D0 .4952 ± .0089 / D2 KD .5053 ± .0027 / **D3 .5071 ± .0094**, paired **+.0119, 3/3 seed 양수**, 네 클래스 모두 미하락 | D0 (= C3 재측정) | 채택. KD 가 이득 대부분(+.0101). cosine 단독 무효이나 부호 안정화 |
| choihy-e011 end-to-end 1차 (예산 부족) | .4765, 부풀림 **+.0868** (ep6 .400 → ep7 .477 → ep8 .379) | D3 .5071 | **미학습·스파이크.** 폐기 근거로 쓰지 않음 |
| **choihy-e012 end-to-end + SpecAugment + cosine** | **.5310**, 부풀림 +.0182, **Deep F1 .399**, κ .3951, 47 epoch / 1.8 h | D3 .5071, C0 .4716 | **현재 parent.** 단일 seed, 재현 필요 |
| choihy-e013 end-to-end + gated KD (3 seed) | — | — | **BLOCKED** — CUDA 노드 장애 |

실행 디렉터리: `artifacts/experiments/{phase1_20260916, phase2_20260916, phase3_20260916, phase3b_20260917, phase4_20260917}`.
보고서: `docs/PSG_PHASE3_5_REPORT_20260918.md`(최신), `docs/PSG_PHASE1_REPORT_20260916.md`, `docs/PSG_RESEARCH_DIRECTION_20260916.md`(e007 감사 재현 스크립트 포함).

## Building on / conflicting with others

- **Related work:** byoungjun 의 frozen split(`psg_audio_subject_split_v1`)·mel 캐시·PSG teacher 캐시(`cache/teacher_targets`)·SpecAugment·gated KD 를 그대로 재사용했다. epoch 격자가 같아 다른 트랙과 epoch 단위로 비교 가능하다. **조율은 아직 미실시** — 단독 진행 중이며, 아래 상충 결과와 방법론 발견을 공유해야 한다.
- **Reproduction:** SpecAugment 와 gated KD 가 **Conformer encoder 에서도** 작동한다(byoungjun 은 CNN encoder 에서 확인). gated KD 가 REM 을 보존한다는 관찰도 재현(REM +.008). e007 라벨 감사는 byoungjun 라벨 감사와 독립적으로 같은 결론(REM 희소는 코호트 특성).
- **Contradiction — 공유 필요:** byoungjun 의 **"네 요소는 단독 무효, 결합해야 효과(+.054)"가 본 트랙에서는 성립하지 않는다.** **C4(결합) < C3(단독)** 이고 결합 시 REM 이 .197 → .147 로 붕괴한다. encoder 가 다르면 combo 필요성도 다른 것으로 보인다.
- **선점된 레인 — 중복 금지:** 본 트랙이 `docs/PSG_RESEARCH_DIRECTION_20260916.md` 에서 제안했던 항목 중 다음은 이미 타 트랙 소유다. **재실행하지 않는다.**
  - **Tracheal 채널** → cloud9sm E027(ceiling +.0409, 10 seed) / E029(`Mic⊕Tracheal` **.5357**, 프로젝트 최고) / E031(tracheal→Mic KD)
  - **AudioMAE / foundation encoder** → cloud9sm E018 frozen, E028·E033 last-4 fine-tune(.5230 ± .0098)
  - **실측 소음 강건성·2단계 어댑터·GRL** → pbj
  - **RML event auxiliary supervision** → sjh
- **teacher 용어 주의:** 이 프로젝트에 "teacher" 가 둘이다. ① 전략문서 Stage 1 teacher = 최고 성능 **오디오** 모델 ② byoungjun `bj-e001` PSG teacher = **생체신호** 입력 KD 소스(.672). 본 트랙 모델은 오디오 입력이므로 KD 의 **타깃(student)** 이지 소스가 아니다.
- **Planned coordination:** ① 위 Contradiction ② 아래 방법론 발견 2건 ③ Conformer encoder 를 byoungjun/cloud9sm 파이프라인에 모듈로 제공 가능 여부.

### 팀에 공유할 방법론 발견

1. **선택 부풀림은 regime 마다 다르다.** 동결 embedding +.011~.016 / encoder 재학습 +.010~.030 / end-to-end 미학습 **+.087** / end-to-end 충분 학습 +.018. 팀이 쓰는 상수 +.049 를 전 regime 에 적용할 수 없다. **부풀림 크기 자체가 실행 안정성 지표** — 큰 값은 그 결과가 단일 epoch 스파이크라는 신호다(e011 이 실제 사례).
2. **GPU 간 비트 재현이 성립하지 않는다.** 같은 GPU 재실행은 해시 완전 일치, 다른 GPU 는 요소 약 0.05% 가 float16 2 ULP 이내로 달라진다. 그 미세 차이가 downstream 학습을 발산시켜 **같은 encoder 인데 B1 이 .0117 차이** 났다. **기존 보고서의 checkpoint 해시는 GPU 종속이며 다른 GPU 에서의 재현 검증 기준으로 쓸 수 없다.** e009 에 C0 대조군을 넣은 이유다.
3. 단일 seed 값은 대표값이 아니다 — C3 의 .5052 는 3 seed 중 운 좋은 하나, 평균 .4952.
4. **encoder 단독 성능으로 downstream 을 예측할 수 없다.** C4 는 단일 epoch 성능이 가장 좋은데(.4458) downstream 은 C3 보다 낮다. e003 에서도 같은 패턴이었다.

## Blockers and risks

- **Data/evaluation:** 공용 evaluator 미연동 → registry 등록 불가. 최고 후보의 **5-fold CV 평가 미실시**(고정 val-44 는 CV 평균보다 약 .027 낙관적). 본 트랙 최대치가 3 seed 라 팀이 쓰는 10-seed 게이트에 미달. **[HIGH] 라벨 커버리지** — REM 4.77%, 287명 중 102명 REM 0, 중증 OSA(AHI 중앙값 54)·split-night 57명. `test_locked` 49명도 REM 4.8% 이므로 test 에서도 개선되지 않는다. **4-stage .60 은 구조적으로 어렵다 — 3-stage 주 지표 협의 필요.**
- **Compute/artifact:** **[HIGH] CUDA 노드 장애(진행형).** `nvidia-smi` 가 `Unable to determine the device handle for GPU2: 0000:61:00.0: Unknown Error` 를 보고하고 **GPU 0/1/3 도 torch 에서 `cuda.is_available()=False`**. GPU2 하드웨어 오류가 드라이버를 오염시킨 것으로 추정, 전례 있음(byoungjun: "GPU 2 복구(호스트 재부팅)"). **관리자 개입 필요.** choihy-e013 이 여기서 막혀 있다. 부작용으로 실행이 **조용히 CPU 폴백**해 31분 헛돌았고, 산출물 삭제 후 `run_phase3.py` 에 CPU 폴백 즉시 실패 가드를 추가했다. 인스턴스 볼륨 없음 → 외부 백업 필요.
- **Integration/device/privacy:** 본 트랙 모델은 **비인과**(미래 epoch 사용)다. 2026-09-15 제품 정의 변경(아침 밤 전체 그래프, 비인과 허용, 단말 내 처리)에서는 허용된다. 실시간 경로 후보가 아니다. 실제 가정 스마트폰 held-out 데이터가 없어 Stage 1 exit gate 가 원리적으로 닫히지 않는다.

## Next experiments

1. **choihy-e013 — CUDA 복구 후 재시작.** end-to-end + gated KD, 3 seed. 확인된 두 최대 레버의 결합이며 기대 이득이 가장 크다. 구현·smoke 완료 상태.
2. **choihy-e014 — e012 재현.** .5310 이 단일 seed 이므로 최소 3 seed. 판정은 seed 쌍 비교로만.
3. **choihy-e015 — 승자의 5-fold CV 평가** (`byoungjun/manifests/cv5/`). 팀 채택 기준(CV 평균 Δ ≥ .02 또는 4/5 fold 동일 부호)에 맞춘다.
4. **choihy-e016 — end-to-end 위 night_norm 재시험.** 동결 경로(C2)에서는 부풀림에 가려 소멸했으나 공동 학습에서는 다를 수 있다. 저비용.
5. **choihy-e017 — KD α/T 스윕** (현재 α .3 / T 2.0 고정).
6. **조율.** 위 Contradiction 과 방법론 발견을 공유하고, Conformer encoder 를 타 트랙 파이프라인에 모듈로 넣을지 합의.

## Code / reproduction

`src/psg_only/endtoend.py` 신규. `models.py`·`data.py`·`train.py`·`evaluate.py`·`cache_embeddings.py` 에 옵션 추가 — **전부 기본값 off** 라 기존 체크포인트가 그대로 로드된다.
`scripts/run_phase{1,2,3,4}.py`, `scripts/build_teacher_cache.py`(PSG teacher logits 를 canonical 순서·epoch 축으로 변환, argmax 일치율 .8951 로 검증).
테스트 44 → **72건 통과**. 무결성 게이트 2종 통과: ① 옵션 off 시 parent Macro-F1 비트 단위 재현 ② embedding 재생성 2 ULP 이내.
구현 중 잡은 버그 2건: `evaluate_b0` 가 night_norm 을 무시(학습/평가 정규화 불일치), end-to-end 루프에서 지역변수 `batch` 가 dataloader 튜플을 가림. 둘 다 정규 실행 전 smoke 에서 검출.
