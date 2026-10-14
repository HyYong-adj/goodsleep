# 연구실 내부 데이터(SNUBH PSG) 교차 데이터셋 실험 계획

**작성:** 2026-10-11 · choihy
**요청:** "training: PSG-Audio → evaluation: 우리 데이터" 형태의 교차 검증(및 그 반대)이 가능한지, 그리고 추가 데이터가 모델에 어떤 영향을 주는지 확인.
**참고 문서(pbj):** `researchers/pbj/reports/snubh_psg_dataset_analysis_20261006.md`(구조·동기·품질 전수 분석), `reports/snubh_psg_integration_framework_plan_20261007.md`, `reports/DATA_CARD_snubh_psg.md`, `external_manifests/snubh_psg/README.md`, `experiments/pbj-e040-snubh-psg-data-prep/`
**관련 기록:** `choihy-e025`(외부 평가), `choihy-e026`(코호트 특징 진단) — `/home/sleep/registry/EXPERIMENTS.md`

> 환자 식별 정보는 이 문서에 쓰지 않는다. 원본 폴더명에는 실명이 들어 있으므로 녹화는 비식별 ID(`SNUBH-R-xxxxxxxx`)로만 부른다.

---

## 0. 한 줄 요약

- **PSG-Audio 학습 → SNUBH 평가(방향 A)는 지금 가능하고 실행했다.** 결과: **전이가 거의 안 된다.** Macro-F1 CV .495 → SNUBH .260 ± .054(5 fold 모두 하락), REM F1 ≈ 0. 레벨 정규화(밤별·도메인)로 +.02 만 회복. 원인은 문턱값이 아니라 표현 붕괴다(REM AUROC .80 → .56). 상세 §5.
- **REM 병목의 원인:** SNUBH(REM 14 %, 경증 다수)에서도 같은 시간대 기준으로 REM-Light 가 소리 특징으로 갈리지 않았다 → "중증 OSA 때문"(e022 해석)은 지지되지 않음, 병목은 실내 마이크 소리 표현 자체.
- **SNUBH 를 학습에 넣은 실험(방향 B·C)도 실행했다(10-11 학습 허용 전달 후).** PSG-Audio + SNUBH 동시 학습(e027)은 SNUBH val .526(제로샷 .269 대비 +.257, 5/5), REM F1 .389 로 올랐고 PSG-Audio val 은 .484(−.011, fold sd 안)로 사실상 그대로다. SNUBH 단독(e028)은 SNUBH .521 / PSG-Audio .332. **두 데이터셋은 서로를 돕지도 해치지도 않는다** — 성능은 "평가 도메인의 데이터가 학습에 있느냐" 로 거의 전부 결정된다. 상세 §5.4.
- **SNUBH 의 REM 상승은 대부분 소리가 아니다:** 밤 중 위치만 쓰는 기준선이 이미 REM F1 .308 이다. 모델은 그 위에 +.04~.08 만 더한다.

---

## 1. 데이터 구조 (`/home/sleep/data/marg_psg_audio/`)

### 1.1 계층

| 층 | 위치 | 내용 | 비고 |
|---|---|---|---|
| L0 원본 | `SNUBH_PSG/{1월,2월}/<날짜>/<환자명+검사유형>/` | Embla N7000 신호(`.ebm` 채널별), RemLogic 스코어(`.esedb`), 야간 영상(`.wmv`), 이벤트(`.esev`) | **실명 포함, 반출 금지.** 166 GB, 2014-01-02 ~ 02-21 |
| 비식별 키 | `_deid/` | salt, `recording_map_private.csv` (원본경로↔해시ID) | QC 외 열람 금지 |
| L1 오디오 | `_derived/snubh_psg/audio/<rid>/clipNN.flac` | 영상 음성 트랙을 16 kHz mono FLAC 로 디코딩, `clips.json`(sha256) | 102 녹화 / 120 클립 / 83 GB |
| L2 매니페스트 | `researchers/pbj/external_manifests/snubh_psg/` | `recordings / participants / epochs / events / sync / audio_quality.csv` | 비식별, SHA256SUMS |
| L3 캐시 | `_derived/snubh_psg/cache/audio_compact_v1/` | **log-mel float16 [N,48,1499] — PSG-Audio 캐시와 같은 계약**(8 kHz, fmax 3.9 kHz), labels(W/L/D/R), valid | 기본 뷰 66 녹화, 7.9 GB |
| | `cache/audio_wave8k_v1/` | 8 kHz 파형 [N,240000] float16 | wave 입력 모델용, 27 GB |
| L4 번들 | `bundles/snubh_psg_external_v1/` | `epochs.csv`, `roles.csv`(전부 `external`), `protocol.json`, `audit.json` | **평가 전용 프로토콜** |

데이터 처리 전 과정은 pbj 가 `pbj-e040` 으로 끝내 두었다(파서 역공학, 동기, 품질 QC, 누수 감사). 이 레인은 그 산출물을 **읽기만** 한다.

### 1.2 PSG-Audio 와 무엇이 같고 다른가

| 항목 | PSG-Audio (학습 데이터) | SNUBH 기본 뷰 (66 녹화 / 58명) | 실험에 주는 의미 |
|---|---|---|---|
| 마이크 | 실내 마이크 48 kHz PCM | **영상 음성 트랙**, WMA Voice 16 kbps, 동적범위 중앙값 11 dB | 코덱·장비가 다른 "열화 도메인". 성능 차이에 코덱이 섞인다 |
| log-mel 전역 평균 | −40.9 dB (std 11.8) | **−25.4 dB** (std 10.0) | 15 dB 레벨 차이. 정규화 조건을 분리해 봐야 함(§3) |
| 라벨 | 4-class (W/REM/Light/Deep), RML | 기술자 수동 30 s, S0–S3/REM → 같은 4-class 매핑 | 매핑은 동일. 채점자·시대(2014)·AASM 버전 차이 가능 |
| REM 비율 | 4.77 %, 287명 중 102명 REM 0 | **14.2 %**, 66 중 3건만 REM 0 | REM 이 정상 범위. 이 레인의 병목(REM)을 다른 조건에서 볼 수 있다 |
| 중증도 | AHI 중앙값 54, AHI<30 은 32명뿐 | AHI 추정 중앙값 15, **normal 22 / mild 11 / moderate 12 / severe 21** | PSG-Audio 에 거의 없는 경증·정상 층을 제공 |
| 기타 | 그리스 성인 OSA 의심군 | 한국, 소아(<18) 8, CPAP·split-night 14 | 층화 보고 필수. **번들 `ahi_class` 는 녹화(그 밤) 기준이라 CPAP 밤은 치료 중 값**("normal" 5건이 실제 정상 아님). 환자 진단 AHI 는 `participants.csv`(`ahi_est_diag`) — 층화·분할 키는 이것을 쓴다 |
| 밤 길이 | 중앙값 4.5 h | 중앙값 8.0 h | 이 레인 모델은 40-epoch 창이라 경과시간 피처가 없어 영향 없음. 경과시간 피처를 쓰는 팀 모델은 1.0 초과 주의(pbj §5.1) |
| 라벨 품질 | RML | 자동 스코어 레이어(21건, 수동 대비 κ .18)는 **사용 금지**; 재스코어 사본 1건 κ .957 | 라벨은 수동 레이어만. 자동-수동 불일치는 난이도 지표로만 |
| 시간축 | 녹화 시작 anchor | `ANALYSIS-START` anchor, 영상 클록 오프셋 보정(중앙값 +5.8 s) | 내 로더는 `subject_epoch_index` 로 읽게 파생 매니페스트 사용 |

제외된 36 녹화: 8 kHz 코덱 15(사실상 불용), 동기 low 19, 주간 MSLT 4(중복 포함). 정식 평가에 넣지 않는다.

---

## 2. 무엇을 알고 싶은가

1. **외부 일반화:** PSG-Audio 에서 CV .495 인 모델이 다른 병원·장비·인구에서 얼마나 유지되는가.
2. **REM 병목의 원인:** 이 레인의 결론(e007/e022)은 "REM 은 데이터·표현에서 막혀 있다"였다. 그 원인이 **코호트(중증 OSA)** 인지 **표현(실내 마이크 소리)** 인지 SNUBH 의 경증 층으로 가를 수 있다.
3. **추가 데이터의 효과:** SNUBH 를 학습에 더하면(양·중증도 구성) PSG-Audio 와 SNUBH 양쪽 성능이 어떻게 변하는가. 위 2 의 답이 이 실험의 기대치를 정한다.

---

## 3. 방향 A — PSG-Audio 학습 → SNUBH 평가 (지금 실행 가능, A 등급)

### A1. 동결 체크포인트 외부 평가 — `choihy-e025` (실행함)

- **모델:** e018 의 5-fold CV 체크포인트 5개(PSG-Audio 4/5 로 학습, fold val 로 선택). SNUBH 로 아무것도 고르지 않는다.
- **입력:** `audio_compact_v1` + 파생 매니페스트(`recording_start_seconds` 제거, `subject_epoch_index` 를 시간축으로; 녹화 내 연속 확인).
- **정규화 3조건 (모두 라벨 미사용):**
  - N0 `train` — PSG-Audio train 전역 통계. 학습 그대로. **주 결과.**
  - N1 `night` — 밤별 z-score. 아침에 밤 전체를 추론하는 제품 정의상 허용되는 test-time 정규화.
  - N2 `domain` — SNUBH 전역 통계. 표적 도메인 unlabeled 통계를 쓰는 transductive 참고값.
- **보고:** 전체 + AHI 등급 / 연령대 / 검사유형 / 동기품질 / 성별 코드 층별. fold 5개 평균 ± sd, 같은 fold 의 in-domain(PSG-Audio val) 점수와 paired 비교.
- **코드:** `scripts/eval_external_snubh.py`, `scripts/chain_e025.sh`. 산출물 `artifacts/experiments/e025_snubh_external/<fold>__<norm>/{results.json, predictions.csv, per_subject.csv}`.

### A2. 코호트 특징 진단 — `choihy-e026` (실행함, CPU)

- e022 의 12개 호흡 포락선 특징으로 피험자 내 REM-Light / Deep-Light paired 효과를 **SNUBH AHI 층별**(성인)과 **PSG-Audio AHI<30 / ≥30**(byoungjun severity 목록)으로 다시 낸다. 모델·probe 적합 없음.
- 판정: e022 의 "중증 OSA 때문" 읽기가 맞다면 경증·정상 층에서 REM-Light 효과가 커져야 한다. 경증에서도 0 이면 병목은 표현 쪽이다.
- **코드:** `scripts/snubh_feature_probe.py`. 산출물 `artifacts/experiments/e026_snubh_feature_probe/`.

### A3. 후속 (A 등급 안에서, 결과 보고 결정)

- **오류 분해:** A1 에서 성능이 무너지면 원인을 (a) 레벨/정규화 (N0 vs N1 차이), (b) 코덱, (c) 인구로 나눈다. (b) 는 PSG-Audio val 을 WMA 계열 코덱으로 왕복시킨 대조군이 필요한데, **이는 pbj 레인(M5 코덱 시뮬레이션·실측 소음 강건성, 선점)** 이므로 직접 하지 않고 결과를 넘겨 요청한다.
- **다른 레인 모델과의 비교:** pbj 가 같은 번들로 `homeaugkd` 등 팀 모델 외부 평가를 GPU 대기 중. 겹치지 않게 이 레인은 Conformer 체크포인트만 평가하고 같은 채점 층으로 맞춘다.
- **민감도:** 소아 8 녹화·CPAP 9 녹화 제외 시 수치, 동기 medium 만 / high 만, `low_dynamic_range` 플래그 녹화 유무(pbj §7.2 가 이 플래그를 제외 기준으로 쓸지 판단 자료를 요청). 전부 e025 예측 CSV 의 사후 층화로 추가 GPU 없이 가능.
- **코골이 의존도 (pbj §8-E):** 같은 환자의 진단 PSG 밤 ↔ CPAP 밤(≈8명), split-night 6밤에서 코골이가 사라졌을 때 단계별 성능 변화. e022 가 본 "호흡 소리가 Deep 은 가르고 REM 은 못 가른다" 를 치료 전후 쌍으로 확인하는 소규모 ablation. 사후 분석, GPU 불필요.
- **라벨 난이도:** 자동 스코어 레이어가 있는 21건에서 자동-수동 불일치 epoch 와 모델 오류의 겹침(라벨로 쓰지 않고 난이도 지표로만).
- **보고 원칙:** locked-test·target-held-out(가정 스마트폰)과 분리된 "external validation" 블록으로만 보고한다. 이 결과를 스마트폰 일반화의 대체 증거로 쓰지 않는다(pbj §8-A).

---

## 4. 방향 B·C — SNUBH 를 학습에 쓰는 실험

> **2026-10-11 갱신:** choihy 가 "연구가설 확인용으로는 학습에 사용할 수 있다" 는 확인을 받았다고 전달함(승인 주체·문서 번호는 이 문서에 미기재 — 데이터 카드에 기록 필요). 범위는 **연구 가설 확인용 학습**에 한정하며, 상업 릴리스·팀 release candidate 편입은 여전히 불가. 이에 따라 B1·C1 을 시작함(`choihy-e027` C1, `choihy-e028` B1).

### 4.0 실행 조건 (원래 계획 — 1 은 위 갱신으로 충족 처리, 2 는 공유 번들을 바꾸지 않고 내 디렉터리에 별도 분할을 두는 것으로 대응)

1. SNUBH 연구용 학습 사용이 데이터 카드에 기록됨(IRB·이용 범위 "확인 필요" 항목 해소). 전략 문서 4.1 의 "Internal PSG = research-only sensitivity and expected-improvement analysis" 범위로 한정, 상업 릴리스 편입 없음.
2. pbj(데이터 담당)와 evaluation owner 동의 — SNUBH 학습용 분할은 공유 번들 `snubh_psg_external_v1`(평가 전용)을 바꾸지 않고 **별도 번들**로 만든다(pbj `multisrc/splits.py` 에 `snubh_psg_dev_v1` 옵션이 설계돼 있음).
3. 레지스트리 사전 등록(판정 규칙 포함).

### 4.1 분할 설계

- **SNUBH 5-fold:** 참가자 단위(58명; 2녹화 이상 8명 — split-night PSG/CPAP 는 같은 fold), `participants.csv` 진단 AHI 등급으로 층화. 기본 뷰 66 녹화. 소아·CPAP 포함 여부는 사전 등록(기본: 포함하되 층화 보고).
- **PSG-Audio:** 기존 `manifests/cv5`(e018 과 동일 fold). `test_locked` 49명은 계속 미접근.
- 두 데이터셋의 fold 를 짝지어(k ↔ k) 같은 k 에서 학습/평가 집합이 어느 쪽에서도 겹치지 않게 한다.

### 4.2 실험 행렬

| ID(가칭) | 학습 | 평가 | 질문 |
|---|---|---|---|
| A1 (=e025) | PSG-Audio fold-train | SNUBH 전체 | 외부 일반화 (실행함) |
| B1 | SNUBH fold-train (~46명) | SNUBH fold-val, **PSG-Audio val fold** | 역방향 일반화. 작은 표본 기준선 |
| B2 | SNUBH 학습량 25/50/100 % | 동일 | B1 의 학습 곡선 — 표본 수 효과와 도메인 효과 분리 |
| C1 | PSG-Audio + SNUBH fold-train (동시) | 양쪽 val fold | **추가 데이터 효과** (A1·B1 대비) |
| C2 | C1 에서 SNUBH 비율 0 / 25 / 50 / 100 % | 양쪽 val fold, AHI 층별 | 추가량-성능 곡선, 어느 층이 이득을 보는가 |
| C3 | PSG-Audio 학습 후 SNUBH fine-tune | 양쪽 val fold | 순차 적응 vs 동시 학습, PSG-Audio 망각 정도 |
| C4 | C1 + 데이터셋 조건(코덱 임베딩/도메인 토큰) | 양쪽 | 코덱 차이를 모델이 "지름길"로 쓰는지 |

- **모델·학습 고정:** e014/e018 설정(Conformer+BiLSTM end-to-end, SpecAugment, cosine). 변수는 데이터 구성 하나.
- **정규화:** 데이터셋별 train 통계(각 fold train 에서만 계산) 또는 밤별 z-score — A1 의 N0/N1 결과로 정한다.
- **판정:** fold 5개 × seed 1 의 paired 차이(같은 fold 기준). 팀 기준에 따라 효과가 fold sd(~.02) 를 넘을 때만 주장. 지표는 Macro-F1·κ·REM F1 을 함께 보고, **양쪽 데이터셋 모두**의 결과를 낸다(한쪽만 좋아지는 것을 이득으로 부르지 않는다).
- **실측(10-11):** C1 은 epoch 당 3~4 분(학습 epoch 15.4만 = PSG-Audio 10.6만 + SNUBH 4.7만, REM 표본 2.4배), fold 당 2~3 h. 실행: `scripts/build_snubh_cv5.py`(분할), `scripts/run_multisource.py`(러너), `scripts/chain_e027_e028.sh <gpu>`(GPU 2·3 작업 큐), `scripts/summarize_e027_e028.py`(같은 fold 기준선과 paired 집계).
- **비용 추정(원안):** 1 run ≈ 1~1.3 h(GPU 3, e018 기준). B1–B2 ≈ 15 run, C1–C3 ≈ 30 run → 약 2~3 GPU-일. C2 는 C1 결과가 양(+)일 때만.

### 4.3 해석상 주의

- SNUBH 는 58명이다. B1 이 PSG-Audio 에서 낮게 나오는 것은 도메인보다 표본 수 탓일 수 있어 B2(학습 곡선)와 같이 읽는다. 같은 맥락에서 C1 의 이득은 "데이터 양" 과 "중증도 다양성" 이 섞이므로, PSG-Audio 쪽 구성 실험인 **byoungjun bj-e058(중증도별 학습 구성·학습 곡선)** 과 설계를 맞춘다.
- 코덱이 데이터셋과 완전히 겹친다(SNUBH = 전부 WMA). 데이터셋 효과와 코덱 효과는 C4 와 pbj 코덱 시뮬레이션 없이는 분리되지 않는다.
- 라벨 기준(채점자·AASM 2007/2012·hypopnea 3 %/4 % 규칙)이 미확인이다(pbj 부록 C 병원 질의). AHI 등급 경계도 이 규칙에 따라 달라진다.
- 라벨 기준이 다를 수 있다. 데이터셋 간 혼동행렬 차이를 라벨 정의 차이로 읽을 가능성을 열어 둔다.

---

## 5. 결과 (2026-10-11 ~ 10-14)

모든 수치는 평가 전용. SNUBH 로 학습·선택·보정한 것 없음. 원장: `registry/EXPERIMENTS.md` choihy-e025 / e026.

### 5.1 A1 외부 평가 (`choihy-e025`, e018 5-fold 체크포인트)

| 정규화 | Macro-F1 | κ | Acc | REM F1 | Wake / Light / Deep F1 | 같은 fold in-domain 대비 |
|---|---:|---:|---:|---:|---|---:|
| (참고) PSG-Audio CV in-domain | .495 ± .022 | .337 | .774 | .278 | — | — |
| **N0 train 통계 (주)** | **.260 ± .054** | .102 | .383 | .000 | .41 / .45 / .18 | **−.236 ± .049 (5/5 하락)** |
| N1 밤별 z-score | .282 ± .095 | .141 | .378 | .003 | .42 / .45 / .25 | −.213 ± .103 |
| N2 SNUBH 전역 통계 | .284 ± .090 | .140 | .380 | .003 | .43 / .45 / .25 | −.212 ± .098 |

- fold 별 N0: .270 / .233 / .183 / .323 / .289. N1·N2 는 fold 0·1·3 을 +.01~.12 올리고 fold 2·4 를 떨어뜨림(fold 2 는 80 % 를 Deep 으로 예측). **정규화는 평균을 조금 올리지만 fold 간 편차를 키운다** — 학습 분포 밖에서 모델의 결정 경계가 fold 마다 제멋대로라는 신호.
- **REM 을 사실상 예측하지 않는다**: 예측 REM 비율 0.01–0.04 % (실제 14.2 %).
- **문턱값 문제가 아니다** — 클래스별 AUROC(N1): Wake .68–.76 (in-domain .91–.95), **REM .53–.60 (in-domain .79–.84)**, Light .55–.62, Deep .64–.78. 순위 정보 자체가 사라졌으므로 사후 보정(temperature, prior 교정)으로는 회복되지 않는다.
- 버그 점검: 예측을 ±1~4 epoch 밀면 Wake/Deep AUROC 가 단조 감소, 0 에서 최대 → 시간 정렬 정상.
- 층별(N0): AHI 기울기 없음(normal .263 / mild .253 / moderate .291 / severe .230). CPAP·split-CPAP 밤이 최저(.14–.19), `low_dynamic_range` 녹화 −.04~−.07, 동기 high/medium 차이 없음, 연령·성별 차이 작음.
- **해석:** 레벨(15 dB) 차이만이 문제였다면 N1/N2 로 대부분 회복됐어야 한다. 회복이 +.02 에 그치고 AUROC 가 무너진 것은 **코덱·장비 차이(WMA Voice 16 kbps 영상 음성 vs 48 kHz PCM)가 스펙트럼-시간 패턴 자체를 바꿨다**는 뜻이다. 인구(AHI) 축에서는 차이가 안 보이므로 1차 원인은 인구보다 신호 쪽으로 본다. 단 코덱과 데이터셋이 완전히 겹쳐 있어 이 분리는 pbj 코덱 시뮬레이션 대조군으로 확정해야 한다.

### 5.2 A2 코호트 특징 진단 (`choihy-e026`, CPU)

피험자 내 REM vs Light 효과(robust sd 단위, 양수 피험자 수/전체):

| 데이터 | 비교 | 변조 호흡대역 비율 | 변조 피크비 | 포락선 ACF 피크 |
|---|---|---|---|---|
| SNUBH 전체 | 비대조 | −.27 (12/63, p 7e-7) | −.19 (13/63) | −.16 (22/63) |
| SNUBH 전체 | **시간대 대조(±10 epoch)** | **−.12 (21/60, p .03)** | −.05 (29/60) | −.04 (16/60) |
| PSG-Audio 전체 | 비대조 / 대조 | −.08 / −.01 | −.07 / .00 | −.05 / −.01 |
| PSG-Audio AHI<30 (19명) | 대조 | +.05 | −.12 | .00 |

- 비대조 SNUBH 에서는 모든 AHI 층(중증 포함)에서 REM-Light 차이가 보였지만, **같은 시간대 Light 와 비교하면 대부분 사라진다**. REM 이 8 시간 밤의 후반에 몰려 있어 생긴 교란이었다.
- Deep-Light 는 두 데이터셋·모든 층에서 강하게 갈린다(SNUBH 피크비 +.59, 35/43; PSG-Audio +.70, 68/85).
- **결론:** 경증 코호트에서도 REM 은 호흡 포락선으로 Light 와 갈리지 않는다. e022 의 "중증 OSA 가 Light 를 조각내서" 해석은 지지되지 않고, 병목은 실내 마이크 소리 표현이다.

### 5.3 결과가 계획에 주는 함의

1. **방향 A 는 "결론 남"**: PSG-Audio 모델을 이 도메인에 그대로 쓸 수 없다. 추가 GPU 평가(다른 체크포인트·seed)는 결론을 바꾸지 않으므로 하지 않는다. 남은 A 등급 작업은 사후 층화(소아·CPAP 제외 민감도, PSG↔CPAP 쌍)와 pbj 코덱 대조군 요청뿐.
2. **방향 C(합쳐서 학습)가 핵심 질문이 됐다.** 제로샷이 안 되므로 "추가 데이터의 효과"는 SNUBH 를 학습에 넣어야만 볼 수 있다. 승인 시 우선순위: **C1(동시 학습) → B1(SNUBH 단독, 기준선) → C3(fine-tune) → C2(양 곡선)**.
3. **기대치:** e026 에 따르면 SNUBH 를 넣어도 REM 이 소리로 더 잘 갈릴 근거는 없다. SNUBH 학습으로 REM 이 오른다면 그것은 REM 사전확률(14 % vs 4.8 %)과 시간 문맥(밤 후반·주기) 덕일 가능성이 크다 → C 실험에서 **SNUBH val 의 REM 이득을 시간 문맥 ablation(창 길이 축소)과 함께 읽는다**.
4. **코덱 증강:** 5.1 의 해석이 맞다면 PSG-Audio 에 코덱 열화 증강(WMA/저비트율 왕복)을 넣는 것만으로 SNUBH 성능이 오를 수 있다. 이는 SNUBH 학습 없이(A 등급) 검증 가능한 유일한 "개선" 실험이지만 pbj 선점 레인이므로 pbj 에 제안하고 결과 비교만 맡는다.

---

### 5.4 B·C 학습 실험 (`choihy-e027` 동시 학습, `choihy-e028` SNUBH 단독)

5-fold × seed 1, 같은 fold 기준선과 paired. 선택 기준은 사전 등록한 대로 학습에 쓴 데이터셋 val 평균(e028 은 SNUBH val; PSG-Audio val 로는 선택 안 함).

| 학습 데이터 | SNUBH val Macro-F1 | SNUBH κ | SNUBH REM F1 | PSG-Audio val Macro-F1 | PSG-Audio REM F1 |
|---|---:|---:|---:|---:|---:|
| PSG-Audio 만 (e018 / e025 제로샷) | .269 ± .048 | .121 | .002 | .495 ± .022 | .278 |
| **PSG-Audio + SNUBH (e027)** | **.526 ± .042** | .366 | **.389** | .484 ± .016 | .254 |
| SNUBH 만 (e028) | .521 ± .031 | .363 | .352 | .332 ± .021 | .127 |
| (참고) 밤 중 위치만 쓰는 기준선 | .263 | — | .308 | — | — |

paired 차이:
- **SNUBH 추가 → SNUBH 성능:** e027 − 제로샷 = **+.257 (5/5)**. 학습 데이터에 같은 도메인이 들어가는 것이 결정적이다.
- **SNUBH 추가 → PSG-Audio 성능:** e027 − e018 = −.011 (1/5 +). PSG-Audio val 로 고른 checkpoint 는 +.011 (4/5 +). 둘 다 fold sd(.022) 안이다 → **이득도 손해도 없음**. 코덱이 다른 데이터를 섞어도 PSG-Audio 쪽이 망가지지 않는다.
- **PSG-Audio 추가 → SNUBH 성능:** e027 − e028 = +.005 (fold 별 +.013 / +.027 / −.001 / −.001 / −.013). PSG-Audio 10만 epoch 를 더해도 SNUBH 에는 보탬이 없다.
- **역방향 전이:** SNUBH 단독 모델의 PSG-Audio 성능 .332 는 정방향 제로샷(.260)보다 조금 낫지만 둘 다 실용 수준이 아니다.
- **REM:** SNUBH 에서 REM F1 .35~.39 가 나오지만 소리 없이 밤 중 위치(10분위)만으로 예측해도 .308 이다(SNUBH train fold 로 적합, 라벨 비율 보정). 모델의 순수 기여는 +.04~.08 이고, REM AUROC 는 .73~.75 다. e026(시간대 대조 시 REM-Light 소리 특징 비분리)과 일관된다 — **REM 은 여전히 소리로 잘 갈리지 않고, SNUBH 에서 나아 보이는 것은 REM 비율(14 %)과 수면 주기 위치 덕이 크다.**

### 5.5 요청 질문에 대한 답

1. **"PSG-Audio 학습 → 우리 데이터 평가, 그 반대도 가능한가":** 둘 다 실행했다. 어느 방향이든 제로샷 전이는 안 된다(.260 / .332). 원인은 코덱·장비 차이로 인한 표현 붕괴다(§5.1).
2. **"추가 데이터가 모델에 어떤 영향을 주는가":** 다른 도메인(코덱) 데이터를 더하는 것은 **그 도메인 성능을 크게 올리고(+.257), 기존 도메인은 해치지 않는다**. 하지만 서로의 도메인에는 보탬이 없다. 따라서 "데이터 양" 이 아니라 "**배포 환경과 같은 녹음 조건의 데이터**" 가 핵심이다. 스마트폰 배포를 생각하면 같은 조건의 라벨 데이터, 또는 코덱·장비 열화 증강이 필요하다는 근거가 된다.

### 5.6 다음 단계 후보

- **C3 (PSG-Audio 사전학습 → SNUBH fine-tune):** e027 과 e028 이 거의 같아서 기대 이득은 작다. 데이터가 적은 새 도메인(스마트폰)에 대비한 "소량 적응" 곡선을 보려면 B2(SNUBH 10/25/50 %) 와 묶어서 하는 것이 낫다 — **SNUBH 몇 밤이면 제로샷 갭이 닫히는가**가 실제 의사결정 질문.
- **코덱 증강(pbj 레인):** PSG-Audio 에 WMA/저비트율 왕복 증강만 넣고 SNUBH 로 평가 → SNUBH 라벨 없이 갭을 얼마나 닫는지. pbj 에 제안.
- **REM:** 시간 문맥을 줄인 ablation 은 우선순위 낮음 — 위치 기준선으로 이미 답이 나왔다.

## 6. 결정 필요 / 확인 요청

0. **pbj 에 공유:** e025 결과(제로샷 붕괴, REM AUROC .56, 정규화 효과 미미) — pbj 의 팀 모델 외부 평가와 같은 채점 층으로 비교, 코덱 시뮬레이션 대조군·코덱 증강 실험의 근거로 제공.
1. **SNUBH 학습 사용 승인** — 누가(데이터 담당 pbj / PI / IRB) 무엇을 확인해야 하는지. 승인 전까지 §4 는 실행하지 않는다.
2. 소아·CPAP 밤을 학습·평가 기본 뷰에 넣을지.
3. pbj 와의 역할 분담 — 팀 모델 외부 평가·코덱 시뮬레이션은 pbj, Conformer 외부 평가와 REM 원인 분석은 이 레인.
