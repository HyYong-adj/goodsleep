# PSG B0 → embedding → B1 실행 가이드

2026-09-14 설정 및 CPU 검증 완료. 원본 위치는 `/home/sleep/data/psg_audio/V3`다.
기존 train 191 / val 44 split을 유지한다. V1/V2 또는 새 subject를 자동 추가하지 않는다.
현재 manifest에 test가 없으므로 모든 결과는 validation-only 개발 결과다.
B1은 미래 context를 쓰는 비인과적/post-wake 모델이다.

## 지금 바로: 기존 Mel cache로 실행

학습 Python은 `/venv/main/bin/python`이다. 원본 복사가 끝나기 전에도 이 경로는 실행할 수 있다.
아래 `GPU_INDEX`를 실제 배정받은 GPU 번호로 바꾼다. 실행기는 GPU를 임의 선택하지 않는다.

```bash
cd /home/sleep/researchers/choihy
/venv/main/bin/python scripts/run_experiment.py --source cached --run --gpu GPU_INDEX
```

자동 순서: 입력 검사 → B0 best checkpoint → B0 validation → frozen embedding → B1 best checkpoint → B1 validation → 동일 epoch/target 비교.
기본값은 seed 20260910, B0 최대 30 epochs, B1 최대 50 epochs, patience 10, CUDA AMP 사용이다.
기본 설정은 192차원 embedding 및 40→20/stride20이다. 80→20 context 비교는 [전용 실험 가이드](B1_80TO20_EXPERIMENT.md)를 따른다.

GPU 없이 실행기 검증만 하려면:

```bash
/venv/main/bin/python scripts/run_experiment.py --source cached --smoke --run --cpu
```

smoke는 train 8 / val 2 subjects, 모델별 최대 2 optimizer steps다.
smoke 결과는 모델 성능 주장에 사용하지 않는다.

## 원본 복사 완료 후: 새 Mel cache부터 실행

필요한 구조:

```text
/home/sleep/data/psg_audio/V3/
  APNEA_EDF/<subject>/<subject>[NNN].edf
  APNEA_RML/<subject>.rml
```

원본 준비 상태 검사와 설정 생성만 수행:

```bash
/venv/main/bin/python scripts/run_experiment.py --source raw
```

EDF 길이/헤더/시간 정렬/Mic 채널/샘플 읽기와 RML UserStaging label 일치를 검사한다.
파일 부재, 복사 중 변경, 잘린 EDF 또는 정렬 불일치가 있으면 non-zero exit와 `BLOCKED_RAW_COPY_OR_ALIGNMENT`를 남긴다.
이 검사는 원본 공급처의 checksum 대조를 대체하지 않는다.
`APNEA_RML_clean`이나 MachineStaging으로 임의 대체하지 않는다.

검사 통과 후 원본부터 full 실행:

```bash
/venv/main/bin/python scripts/run_experiment.py --source raw --run --gpu GPU_INDEX
```

원본 Mel 생성은 프로젝트의 `.venv-raw/bin/python`에서 CPU로 실행한다.
B0/B1 학습과 embedding은 지정된 GPU에서 실행한다. 원본 전체 변환은 I/O와 CPU 시간이 상당히 필요하므로,
기존 baseline을 먼저 실행하려면 `--source cached`를 사용한다.

원본 전처리는 기존 cache 생성기의 48 kHz Mic → epoch mean 제거 → 8 kHz resample →
48-bin log-Mel, n_fft=256, win=200, hop=160, center=False, 30–3900 Hz를 유지한다.
학습 split의 valid epoch만으로 global mean/std를 계산한다.
새 cache는 실행 폴더에 저장하며 공용 원본과 기존 cache는 수정하지 않는다.

## 산출물 및 재실행

각 실행은 `artifacts/experiments/<UTC timestamp>/` 아래 새 폴더를 생성한다.
`--output /absolute/new/run_directory`로 별도 위치를 지정할 수 있으며 기존 폴더 덮어쓰기는 거부한다.

- `experiment.json`: seed, 실행 예산, 단계, 성공/실패 상태
- `manifest.csv`, `b0.yaml`, `b1.yaml`: 기존 split을 보존한 로컬 manifest와 resolved config
- `raw_readiness.json`, `cache_preflight.json`: 실제 입력 검사 결과
- `train_b0.log`, `embeddings.log`, `train_b1.log`, 평가 로그
- `b0/best.pt`, `b1/best.pt`, 각 모델의 `history.jsonl`
- `embeddings/`: canonical label, 실제 30초 시간축, 파일 hash와 B0 provenance
- `b0/evaluation/`, `b1/evaluation/`: predictions.csv, results.json
- `comparison.json`: 동일 validation epoch/target 검사와 metric 비교

중단된 full 학습을 optimizer state에서 이어가는 기능은 없다. 새 실행 폴더로 다시 시작한다.
개별 생성 스크립트는 완료 metadata와 hash가 일치하는 subject cache를 재사용할 수 있다.
이전 smoke checkpoint/embedding은 새 provenance 계약을 만족하지 않으므로 재생성한다.

## 검증 결과

- CPU pytest: 24 passed.
- 실제 cache preflight: 135,010 rows, valid 132,164 epochs, train 191 / val 44 / test 0.
- 통합 CPU smoke: `artifacts/setup_smoke_20260914/experiment.json`의 `SMOKE_COMPLETE`.
- B0/B1 모두 동일한 validation 926 valid epochs 평가 및 4-class support 확인.
- checkpoint logits round-trip, 동일 seed B1 재학습, 빈 target batch 건너뛰기, provenance 변조,
  manifest/cache label 불일치, split leakage, 실제 시간축 gap, 평가 누락 검증.
- 합성 EDF/RML: 정상 입력 통과, truncated EDF와 annotation mismatch 검출.
- 합성 원본 → Mel 생성 → B0 → embedding → B1 통합 CPU smoke 완료:
  `artifacts/raw_fixture_smoke_20260914/` (실제 원본 full 학습과 구분).
- 실제 원본 Mic 1 epoch probe: 1,440,000 samples → `[48,1499]`, finite.
  기존 float16 Mel과 mean absolute error 0.00682, max 0.01593.
  전체 원본 동기화 검증을 완료했다는 의미는 아니다.

초기 전체 원본 검사 시점에는 EDF 1,243개 중 141개 검사 통과, 1개 검증 실패, 1,101개 미복사,
RML 235개 미복사였다. 복사는 진행 중이므로 최신 상태는 실행기로 다시 검사한다.

Full 준비 설정은 `artifacts/ready_cached_20260914/`, 원본 준비 설정은 `artifacts/ready_raw_20260914/`에 있다.
GPU/full 학습은 이번 설정 작업에서 실행하지 않았다.

## 환경 재구성

현재 주 환경: `/venv/main/bin/python` (torch 2.14.0+cu130, numpy/pandas/PyYAML/scikit-learn/pyedflib/pytest).
원본 변환 환경: `.venv-raw` (torch 2.8.0+cpu, torchaudio 2.8.0+cpu; 기타 의존성은 주 환경에서 상속).
시스템 torch를 덮어쓰지 않는다. 원본 환경을 다시 만들 때:

```bash
bash scripts/setup_raw_env.sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 /venv/main/bin/python -m pytest -q
```

모든 생성 데이터, 식별자가 포함된 manifest, checkpoint, 가상환경은 Git 제외 대상이다.

## requirements 설치 명령

`requirements.txt`는 학습·평가·EDF 검사에 필요한 직접 의존성,
`requirements-dev.txt`는 여기에 pytest를 추가한 목록이다.
`requirements-raw.txt`는 공통 의존성과 검증된 CPU torch/torchaudio 조합을 포함한다.
일반 패키지는 호환 버전 범위이며 정확한 환경 lock 파일은 아니다.

현재 서버에서 기존 CUDA torch를 활용하면서 프로젝트 전용 환경을 만들려면:

```bash
cd /home/sleep/researchers/choihy
/venv/main/bin/python -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
bash scripts/setup_raw_env.sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 python -m pytest -q
```

테스트가 필요하지 않으면 `requirements-dev.txt` 대신 `requirements.txt`를 사용한다.
원본용 환경은 위의 setup 스크립트가 `.venv-raw`에 설치한다.
직접 설치 명령은 `.venv-raw/bin/python -m pip install -r requirements-raw.txt`다.
CPU torch가 고정된 원본용 requirements는 GPU 학습 환경과 별도의 환경에 설치한다.
이후 활성화된 `.venv`에서 `python scripts/run_experiment.py ...`로 실행할 수 있다.
