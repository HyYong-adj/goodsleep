# choihy Status

**Updated:** 2026-10-14 (상단 갱신 블록) · 본문 2026-09-23
**Stage / lane:** Foundation / Conformer epoch encoder — 음향 표현 축 (pbj 분담안 "Conformer/T40" 레인)
**Current parent:** **choihy-e014** end-to-end Conformer+BiLSTM (SpecAugment + cosine) — fixed val-44 .5351 ± .0040 (3 seed), **5-fold CV .4954 ± .0217** (choihy-e018)

> 전부 validation. `test_locked` 49명 미접근. 공용 evaluator 미연동. 판단은 같은 seed / 같은 fold 의 paired 차이로만 한다.
> 고정 val-44 는 CV 보다 **+.040 낙관적**임을 이 트랙에서 실측했다(e018). 절대값은 참고용이다.

## 2026-10-14 갱신 — SNUBH 내부 데이터 교차 실험 (상세: `docs/HANDOFF.md`, `docs/SNUBH_CROSS_DATASET_PLAN.md`)

- e023 PCEN −.0025 / e024 logmel+delta −.0014 (같은 seed parent .5310 대비) → 기각. front-end 축 종료.
- e025 PSG-Audio 모델의 SNUBH 제로샷: **.260 ± .054**, REM ≈ 0, 정규화로 +.02 만 회복, AUROC 붕괴 → 코덱·장비 차이가 표현을 깨뜨린다.
- e026 REM-Light 는 경증 코호트에서도 (시간대 대조 시) 소리로 갈리지 않는다 → e022 의 "중증 OSA 때문" 해석 철회, 병목은 실내 마이크 표현.
- e027 PSG-Audio+SNUBH 동시 학습: SNUBH **.526**(+.257, 5/5), PSG-Audio .484(−.011, 편차 안). e028 SNUBH 단독: SNUBH .521 / PSG-Audio .332. 데이터셋끼리 서로 돕지 않는다 — 평가 환경과 같은 녹음 조건의 데이터가 핵심.
- SNUBH REM F1 .35–.39 중 .308 은 밤 중 위치만으로도 나온다(소리 기여 +.04–.08).
- SNUBH 학습은 2026-10-11 사용자 전달("연구가설 확인용 학습 가능")에 따름. 승인 주체 미기록, 릴리스 사용 불가.

## Current conclusion

- **팀 공통 기준(5-fold CV)에서 이 레인의 모델은 기존 CNN 파이프라인과 동등하다.** CV .4954 ± .0217 vs byoungjun CV 기준선 .498 ± .019 → **−.003**. 고정 val 에서 보였던 우위(.5351)는 모델이 아니라 프로토콜이었다. 승격 주장 없음.
- **효과가 확인된 레버는 둘뿐이다:** SpecAugment(+.024), end-to-end 공동 학습(+.024). 후자는 팀 차원의 발견이 아니라 이 레인이 스스로 택했던 동결 2단계 설계를 걷어낸 것이다(byoungjun 은 처음부터 end-to-end).
- **parent 확정(09-20) 이후 9개 실험이 전부 parent 를 넘지 못했다** — KD ×2, encoder 용량, 앙상블, focal, SupCon, BiLSTM 뒤 Transformer ×2, front-end 특징. 닫힌 축: 시간 문맥(교체·적층), encoder 용량, KD-on-end-to-end, 손실함수(재가중·focal·contrastive), 앙상블, 호흡 포락선 front-end.
- **REM 이 모든 축의 공통 병목이고, 원인은 두 겹의 데이터 제약이다.** ① V3 287명 중 102명(35.5%)이 밤 전체 REM 0 (e007). ② **REM 이 있는 피험자에서도 환경 마이크 표현으로는 REM 과 Light 가 갈라지지 않는다** (e022: 호흡 리듬·포락선 12특징 모두 피험자 내 효과 < .08 sd). 중증 OSA(AHI 중앙값 54) 에서 Light 도 무호흡으로 조각나 호흡이 REM 만큼 불규칙하기 때문으로 읽힌다. 같은 특징이 Deep 은 강하게 가른다(변조 피크비 +.69, 54/68명, p 1e-6) — 그리고 Deep 은 모델이 이미 잘 맞힌다.
- **목표 대비 (CV):** Macro-F1 .495 / κ .337 / Acc .774 vs .60 / .50 / .60. REM F1 .278. 남은 격차는 아키텍처가 아니라 데이터·지표 정의의 문제다.

## New evidence

| Experiment | Result (val, best epoch) | Comparator | Verdict |
|---|---|---|---|
| choihy-e001–e006 (09-14~15) | B0 CNN .3631 → B1 BiLSTM .4340 → Conformer encoder **.4833**; 학습률 1e-4·문맥 80·Transformer 교체는 기각 | majority .2243 | 병목은 시간 모델이 아니라 음향 표현 |
| choihy-e007 RML 라벨 감사 (CPU) | REM 4.77%, **102/287명 REM 0**, 판독 중앙값 4.45 h; 매핑 코드 정상 | — | 데이터 특성 확정. byoungjun 감사와 독립 일치 |
| choihy-e008 동결 embedding 위 시각특징/cosine/smoothing/REM×2 (3 seed) | 전부 기각, 결합 −.0098 (3/3 음수) | A0 .4817 | 기각 |
| choihy-e009 encoder 단계 (1 seed, C0 대조군) | night_norm .4952(보정 .4652) / **SpecAugment .5052** / 결합 .4912 (REM .197→.147) | C0 .4716 | SpecAugment 채택, 결합 기각. GPU 간 비트 재현 불성립 확인 |
| choihy-e010 gated KD + cosine (3 seed, 동결) | **.5071 ± .0094**, paired +.0119, 3/3 양수 | D0 .4952 | 채택(동결 경로 한정) |
| choihy-e011/e012/e014 end-to-end | 1차 미학습(.4765, 부풀림 +.087) → 예산·정규화 후 **.5310** → 3 seed **.5351 ± .0040** | D3 .5071 | **현재 parent** |
| choihy-e013 / e016 end-to-end + KD α.3 / α.15 (3 seed) | +.0043 (1/3 양수, sd 6배) / **−.0085 (0/3)**; α 줄여도 악화 | e014 | **KD-on-e2e 라인 종료.** 가설: teacher 는 깨끗한 오디오, student 는 SpecAugment 입력 |
| choihy-e015 Conformer 2→4 blocks (+65% params) | +.0059 | e012 .5310 | 기각 — 용량은 병목 아님 |
| choihy-e017 3-seed 확률 앙상블 (CPU) | +.0012; κ +.018 인데 **REM −.042** | e014 평균 | 기각 — 평균화가 희소 클래스를 지움 |
| **choihy-e018 5-fold CV** | **.4954 ± .0217** (fold .469–.526); fold sd 가 seed sd 의 5.4배 | 고정 val .5351 | **고정 val 낙관 +.040 실측. byoungjun CV 와 동등** |
| choihy-e019 focal γ2 | −.0287, **REM .268→.175** | e012 | 기각 |
| choihy-e020 SupCon w.1 (pre-temporal, class-weighted) | −.0121, 부풀림 +.045; 배치당 REM anchor 3.8개·사용 .998 → 희소성 탓 아님 | e012 | 기각 — **손실함수 라인 종료** |
| choihy-e021 BiLSTM 뒤 Transformer 2/4층 | −.0242 (REM →.181) / −.0016 (부풀림 **+.110**, 최대치) | e012 | 기각 — **시간 구조 축 종료** (교체 e005 + 적층 e021) |
| choihy-e022 front-end 특징 진단 (CPU) | REM-Light 분리 없음; Deep-Light 강한 분리; 12특징 선형 probe REM F1 .083 | 2.6M 모델 REM .268 | **호흡 포락선 front-end 라인 종료, GPU 미사용** |

원장: `/home/sleep/registry/EXPERIMENTS.md` (choihy-e001~e022 전건). 해설: `docs/RESEARCH_WALKTHROUGH.md`.

## Building on / conflicting with others

- **Related work:** byoungjun 의 frozen split·mel 캐시·teacher 캐시·5-fold manifest(`manifests/cv5`)·SpecAugment·gated KD 재사용. epoch 격자가 같아 다른 트랙과 직접 비교 가능. cloud9sm E027/E029(Tracheal) 결과를 e022 가설의 동기로 사용.
- **Reproduction:** SpecAugment 와 gated KD 가 Conformer encoder 에서도 작동(동결 경로). REM 희소는 코호트 특성(byoungjun·choihy 독립 일치). 시간 문맥 포화(bj-e037 과 e004/e005/e021 일치).
- **Contradiction — 공유 필요:** ① byoungjun 의 "네 요소는 결합해야 효과" 가 Conformer 에서는 반대(결합 < 단독, REM 붕괴). ② **증강 + KD 결합이 공동 학습에서 불안정**(e013/e016, 용량 줄여도 악화) — byoungjun 현재 최고 설정 `homeaug + KD` 가 단일 seed 이므로 seed 확장 시 같은 현상이 나올 수 있다.
- **선점 레인 — 재실행 금지:** Tracheal(cloud9sm E027/E029/E031), AudioMAE(cloud9sm E018/E028/E033), 실측 소음 강건성·어댑터·GRL(pbj), RML event aux(sjh).
- **Planned coordination:** 위 Contradiction 2건 + 아래 방법론 발견 + e022 의 REM-Light 비분리 결과를 지표 재협의(`bj-e033`)에 제공.

### 팀에 공유할 방법론 발견
1. **선택 부풀림은 regime 마다 다르다:** 동결 +.011~.016 / encoder 재학습 +.010~.030 / e2e 미학습 +.087 / e2e 충분 학습 +.018 / **BiLSTM+Transformer +.110**. 부풀림 크기 자체가 결과 신뢰도 지표다.
2. **GPU 간 비트 재현 불성립** (같은 GPU 는 해시 일치, 다른 GPU 는 float16 2 ULP 이내 차이 → downstream .0117 차이). 기존 보고서의 checkpoint 해시는 GPU 종속.
3. **고정 val-44 의 낙관 편향은 +.040** (이 트랙 실측; byoungjun 실측 +.027 보다 큼). fold 간 sd(.022) ≫ seed 간 sd(.004).
4. 단일 seed 값·encoder 단독 지표·앙상블 κ 상승은 각각 대표값이 아니다.
5. **지표 주의:** 통상 3-stage(Wake/NREM/REM)도 REM 을 독립 클래스로 두므로 이 천장을 물려받는다. byoungjun 3-stage .587 이 높은 건 NREM 병합 때문이지 REM 이 나아진 게 아니다.

## Blockers and risks

- **Data/evaluation:** [HIGH] REM 데이터 제약 두 겹(e007 부재, e022 비분리) → REM 을 독립 클래스로 두는 어떤 지표든 .60 미달. 공용 evaluator 미연동. 팀 10-seed 게이트 대비 이 트랙은 3 seed / 5 fold 까지만.
- **Compute/artifact:** **GPU 0/1 은 타인 배정 — 사용 금지.** 09-23 에 e021 이 0/1 로 옮겨져 54분 점유 후 중단·재실행했다(재발 방지: 규칙 메모리 저장, chain 스크립트 주석, GPU 여유 <4 GiB preflight). GPU 2/3 만 사용. `pkill -f` 는 자기 셸과 매칭되므로 PID 로 종료. chain 로그의 `exit=$?` 가 `$(date)` 에 덮이던 버그 수정(14곳).
- **Integration/device/privacy:** 비인과 모델(제품 정의상 허용). 실제 가정 스마트폰 held-out 데이터 없음 → Stage 1 exit gate 원리적으로 미충족.

## Next experiments

1. **성능 실험 중단 권고.** 9연속 음성이고 남은 축이 없다. 이 레인의 결론은 "REM 은 데이터·표현에서 막혀 있다"이며, 이를 근거로 넘기는 것이 다음 일이다.
2. byoungjun 에 Contradiction 2건 공유(특히 증강+KD 불안정 — 그쪽 단일 seed 최고 설정에 직접 관련).
3. 지표 재협의에 e007 + e022 증거 제공. REM 독립 클래스 유지 시 천장 명시.
4. Conformer encoder 를 모듈로 제공할지는 CV 동등 결과상 "더 낫다" 주장 불가 — 앙상블 다양성 관점에서만 검토 가능.
5. (조건부) 대역폭 >3.9 kHz 는 985 GB 재처리 비용 대비 cloud9sm 16 kHz 라인의 REM ≤ .232 가 부정적 간접 증거 → 보류.

## Code / reproduction

`src/psg_only/{endtoend,objectives}.py` 신규(end-to-end 모델, focal/supcon, BiLSTM 뒤 Transformer 옵션). 기존 파일 옵션은 전부 기본값 off. `scripts/run_phase3.py` 가 현재 주력 러너(`--kd --kd-alpha --layers --manifest --focal --supcon --post-transformer`). `scripts/frontend_probe.py`(e022). 테스트 **101건 통과**. 무결성 게이트 2종 통과.
