# 심정지 조기경보 시스템

[![CI](https://github.com/BioCode67/hf-risk-prediction-sw/actions/workflows/ci.yml/badge.svg)](https://github.com/BioCode67/hf-risk-prediction-sw/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **활력징후 시계열로 원내 심정지를 미리 경고하되, 오경보를 줄이고 "왜 위험한지"까지 설명하는 시스템**
>
> 2026 K-Health 미개방 의료데이터 활용 경진대회 출품작 (경북대학교병원 활력징후 데이터)

---

## 한 문단 요약

병동에서 쓰는 조기경보점수(NEWS)는 **모든 환자에게 같은 기준**을 적용해서 오경보가 많습니다.
거짓 알람이 잦으면 의료진이 알람을 무시하게 되고(**alarm fatigue**), 그러면 성능 좋은 모델도
현장에서 안 쓰입니다. 이 프로젝트는 정확도 경쟁 대신 **① 환자 개인의 기저선에서 얼마나
벗어났는지**로 판단하고, **② 모든 경보에 SHAP 근거를 붙이며**, **③ 같은 검출률에서 알람을 몇 번
울리는지(알람 부담)**를 핵심 지표로 평가합니다.

---

## 처음 오셨나요? — 읽는 순서

| 순서 | 무엇을 | 어디서 |
|:--:|---|---|
| 1 | **ML 흐름 익히기** (EDA→FE→모델→튜닝→앙상블) | [`notebooks/02_learning_project.ipynb`](notebooks/02_learning_project.ipynb) ← 42초면 끝, 여기부터 |
| 2 | 지표 읽는 법 | 아래 [핵심 개념](#핵심-개념--지표-읽는-법) 절 |
| 3 | 이 프로젝트 파이프라인 체험 | [`notebooks/01_baseline_pipeline.ipynb`](notebooks/01_baseline_pipeline.ipynb) |
| 4 | 실데이터로 돌려보기 | [`notebooks/03_challenge2019_realdata.ipynb`](notebooks/03_challenge2019_realdata.ipynb) |
| 5 | 대회 전략·제안서 | [`docs/`](docs/README.md) |

> 처음이라면 **2번 노트북부터** 여세요. 본 과제 데이터는 양성이 1% 남짓이라 지표가 늘 바닥에
> 붙어 있어서 배우기가 어렵습니다. 2번은 신호가 뚜렷한 데이터로 흐름만 먼저 익히는 용도입니다.

---

## 폴더 구조

```
hf-risk-prediction-sw/
│
├── 📓 notebooks/                    ← 여기부터 보세요 (실행하며 배우는 곳)
│   ├── 01_baseline_pipeline.ipynb       프로젝트 파이프라인 4단계 체험
│   ├── 02_learning_project.ipynb        ML 표준 흐름 학습용 (권장 시작점)
│   └── 03_challenge2019_realdata.ipynb  실제 공개 데이터로 검증
│
├── 🧠 src/                          ← 실제 코드 (아래 3개 트랙으로 나뉨)
│   │
│   │  ── 시계열 트랙 ── 이 프로젝트의 본체. 활력징후 → 심정지 예측
│   ├── vitals_data.py          데이터 로드·정제·슬라이딩 윈도우·환자단위 분할
│   ├── vitals_train.py         XGBoost 학습 + NEWS 비교 + 오경보 지표
│   ├── vitals_explain.py       SHAP — "왜 위험한가" 설명
│   ├── vitals_report.py        그림 생성 (PR곡선·궤적·lead-time·알람부담)
│   ├── vitals_phenotype.py     심정지 표현형 군집화 (호흡성/순환성/혼합)
│   │
│   │  ── 데이터 어댑터 ── 여러 데이터를 같은 파이프라인에 연결
│   ├── mimic_explore.py        MIMIC-IV (실제 중환자 데이터)
│   ├── sepsis_explore.py       PhysioNet Challenge 2019 (공개, 인증 불필요)
│   ├── omop_explore.py         OMOP CDM 표준 포맷
│   │
│   │  ── 딥러닝 벤치마크 ── XGBoost와 비교용 (선택)
│   ├── utils.py                전처리 (전진대치·정규화)
│   ├── dataset.py              가변길이 시퀀스 배칭 (padding/masking)
│   ├── model.py                LSTM / GRU 분류기
│   │
│   │  ── 정적 트랙 ── 초기 버전. 한 시점 검사값 → 심부전 위험
│   ├── data_loader.py          데이터 로드 + OMOP CDM 변환
│   ├── train.py                LightGBM/XGBoost 학습
│   ├── explainability.py       SHAP
│   └── main.py                 FastAPI 서버 (/predict, /convert-omop)
│
├── 🧪 tests/                        pytest — 데이터 없어도 통과 (자동 skip)
├── 📄 docs/                         대회 전략·제안서·런북 → docs/README.md 참고
├── 📊 data/                         데이터셋 (git에 없음)
├── 💾 models/                       학습된 모델·그림 (git에 없음)
├── 🚀 train_dl.py                   딥러닝 학습 실행 스크립트
└── ⚙️  requirements.txt              의존성
```

> **왜 `src/`가 폴더로 안 나뉘어 있나요?**
> `src/`는 설치되는 패키지가 아니라 **import 경로**입니다. 모듈끼리 `from vitals_data import ...`
> 처럼 이름만으로 서로를 부르기 때문에, 하위 폴더로 나누면 모든 import와 실행 경로가 깨집니다.
> 대신 파일명 앞부분(`vitals_*`, `*_explore`)으로 트랙을 구분합니다.

---

## 두 개의 트랙

이 저장소에는 **목적이 다른 두 파이프라인**이 함께 있습니다.

|  | ⏱ 시계열 트랙 (본체) | 📋 정적 트랙 (초기 버전) |
|---|---|---|
| **입력** | 시간에 따른 활력징후 (맥박·혈압·체온·SpO₂·호흡수) | 한 시점의 검사 결과 |
| **예측** | 앞으로 N시간 내 심정지 | 심부전 사망 위험 |
| **파일** | `vitals_*.py` | `data_loader.py`, `train.py`, `main.py` |
| **상태** | **대회 출품작 — 현재 개발 중** | 완성, 유지만 |

아래 설명은 대부분 **시계열 트랙** 기준입니다.

---

## 설치

```bash
pip install -r requirements.txt
```

딥러닝 벤치마크(LSTM/GRU)까지 쓸 경우에만 추가로:

```bash
pip install -r requirements-torch.txt
```

> GPU 서버 환경 구축은 [`docs/server-runbook.md`](docs/server-runbook.md)에 단계별로 정리돼 있습니다.

---

## 빠른 시작

**데이터가 없어도 바로 돌아갑니다.** 합성 데이터 생성기가 내장돼 있습니다.

```bash
# 1. 학습 + NEWS와 비교 (합성 데이터, 1분 이내)
python src/vitals_train.py

# 2. SHAP으로 "왜 위험한지" 확인
python src/vitals_explain.py

# 3. 그림 생성 (models/ 폴더에 저장)
python src/vitals_report.py

# 4. 심정지 표현형 발견
python src/vitals_phenotype.py
```

### 실제 공개 데이터로 돌려보기

PhysioNet Challenge 2019(패혈증)은 **인증 없이 다운로드**할 수 있어서, 실데이터 검증용 프록시로
씁니다. 심정지는 아니지만 "활력징후 → 임박한 악화 사건"이라는 **문제 구조가 같습니다**.

```bash
# 데이터 다운로드
wget -r -N -c -np -nH --cut-dirs=4 \
  https://physionet.org/files/challenge-2019/1.0.0/training/training_setA/

# 실행 (--horizon: 예측 지평, 아래 주의사항 참고)
python src/sepsis_explore.py ./training_setA --horizon=6

# 전체 데이터 + 튜닝 + GPU
python src/sepsis_explore.py ./training_setA --horizon=6 --tune --trials=50 --gpu
```

> ⚠️ **`--horizon` 기본값 1시간은 너무 좁습니다.** 환자당 이벤트가 1개뿐이라 양성 윈도우가
> 사실상 1개(전체의 0.2%)가 되고, AUPRC가 기준선까지 붕괴합니다. 조기경보의 표준인
> **6시간(`--horizon=6`)** 정도로 넓히세요.

---

## 주요 명령어

| 하고 싶은 것 | 명령어 |
|---|---|
| 합성 데이터로 학습 | `python src/vitals_train.py` |
| Optuna 하이퍼파라미터 튜닝 | `python src/vitals_train.py --tune --trials 40` |
| GPU로 학습 | `python src/vitals_train.py --gpu` |
| SHAP 설명 | `python src/vitals_explain.py` |
| 그림 전부 생성 | `python src/vitals_report.py` |
| 심정지 표현형 군집 | `python src/vitals_phenotype.py` |
| Challenge 2019 실행 | `python src/sepsis_explore.py <폴더> --horizon=6` |
| MIMIC-IV 구조 확인 | `python src/mimic_explore.py <폴더>` |
| MIMIC-IV 전체 실행 | `python src/mimic_explore.py <폴더> --model --gpu` |
| 딥러닝(LSTM) 학습 | `python train_dl.py --rnn lstm --epochs 20` |
| API 서버 (정적 트랙) | `uvicorn main:app --app-dir src --reload` |
| 테스트 | `pytest -q` |

`--gpu`와 `--tune`은 함께 쓸 수 있습니다(각 Optuna 시도가 CUDA에서 실행). 둘 다 **선택 사항**이며,
기본 경로는 CPU·고정 하이퍼파라미터입니다 — 본선 안심존이 오프라인 폐쇄망이라 기본 경로를
가볍게 유지합니다.

---

## 핵심 개념 — 지표 읽는 법

> 결과 숫자가 좋은 건지 나쁜 건지 판단이 안 될 때 이 절을 보세요.

### AUPRC — **기준선과 비교해서** 읽어야 합니다

가장 중요한 지표입니다. 그런데 **절대값만 보면 반드시 오판합니다.**

> **AUPRC의 기준선 = 양성 비율**

- 양성이 35%인 데이터 → 아무 정보 없는 모델도 AUPRC ≈ 0.35
- 양성이 1.2%인 데이터 → 아무 정보 없는 모델은 AUPRC ≈ 0.012

즉 **AUPRC 0.03은 상황에 따라 훌륭할 수도, 형편없을 수도** 있습니다. 기준선이 0.012면 2.5배로
좋은 것이고, 기준선이 0.35면 재앙입니다. **항상 양성 비율을 함께 확인하세요.**

### ROC-AUC — 희귀 사건에서는 믿지 마세요

심정지처럼 드문 사건은 음성이 압도적으로 많아서 ROC-AUC가 쉽게 0.9를 넘습니다.
**ROC는 높은데 AUPRC는 바닥인 상황이 흔합니다.** 이 프로젝트가 ROC를 주 지표로 쓰지 않는 이유입니다.

### 알람 부담 (alarm burden) — 이 프로젝트의 핵심

> **"같은 검출률을 낼 때 알람을 몇 번 울리는가"** — 낮을수록 좋음

간호사가 실제로 체감하는 숫자입니다. 민감도를 동일하게 맞춘 뒤 알람 횟수를 비교해야
공정합니다. NEWS보다 이 값이 낮아야 우리 방법이 이긴 겁니다.

### 조기경보 확보시간 (lead-time)

사건 몇 시간 전에 경보했는지. 길수록 의료진이 개입할 여유가 생깁니다.

### 민감도 @ 95% 특이도

"오경보를 5%로 묶었을 때 실제 사건을 몇 % 잡는가" — 임상 운영 관점의 실전 지표입니다.

### ⚠️ 정확도(Accuracy)는 쓰지 마세요

양성이 1%면 **전부 "음성"이라고 찍어도 정확도 99%**입니다. 불균형 데이터에서 정확도는 무의미합니다.

### ⚠️ 교차검증 표준편차를 반드시 함께 보세요

모델 간 차이가 CV 표준편차보다 작으면 **"차이 없음"** 입니다. 경북대 데이터는 573명 소표본이라
특히 중요합니다. 없는 우위를 주장하지 않으려면 항상 평균±표준편차로 보고하세요.

---

## 방법론

### 데이터 흐름

```
시간별 활력징후 (맥박·수축기/이완기혈압·체온·SpO₂·호흡수)
   ↓  sanitize_vitals()      생리학적으로 불가능한 값 제거, 화씨→섭씨 변환
   ↓  build_windows()        슬라이딩 윈도우 + "N시간 내 심정지" 라벨링
   ↓  피처 생성              vital별 평균·표준편차·최소·최대·최근값·기울기·변화량 + shock index
   ↓  add_personalized_features()   ★ 개인 기저선 대비 편차 (차별점)
   ↓  add_static_features()  연령대·성별
   ↓  patient_level_split()  환자 단위 분할 (같은 환자가 train/test에 동시에 없도록)
   ↓  train_xgboost()        cost-sensitive XGBoost
   ↓  평가                   AUPRC · 민감도@특이도 · 알람부담 · lead-time
   ↓  SHAP                   경보 근거 제시
```

### ★ 개인 기저선 이탈 — 이 프로젝트의 차별점

`add_personalized_features()`는 각 vital에 대해 **그 환자 자신의 초기 안정기 대비 편차**를
추가합니다.

병동 전체 기준으로는 정상인 값이 **특정 환자에게는 큰 이탈**일 수 있습니다. 이 설계는
- 개인차에서 오는 오경보를 구조적으로 줄이고
- 병원이 달라져도 잘 옮겨가며 (전이성↑)
- **대조군이 없는 경북대 데이터의 약점을 방법론으로 전환**합니다 ("환자가 곧 자신의 대조군")

합성 데모 기준 AUPRC가 **0.76 → 0.84**로 올랐습니다.

### 하나의 파이프라인, 여러 데이터

같은 윈도우·라벨링 로직이 얇은 어댑터를 통해 네 가지 데이터에서 동작합니다.

| 어댑터 | 데이터 | 역할 |
|---|---|---|
| `generate_synthetic_cohort` | 합성 | 데이터 없이 CI·데모 실행 |
| `cohort_from_challenge2019` | PhysioNet Challenge 2019 | **인증 불필요** 공개 실데이터 검증 |
| `cohort_from_mimic` | MIMIC-IV | 경북대에 없는 **대조군** 공급 → 정직한 오경보 측정 |
| `cohort_from_khth` | 경북대 (안심존) | 본선 실제 데이터, `CARDT` 기준 정확한 라벨 |

---

## 데이터

⚠️ **데이터와 학습된 모델은 git에 없습니다** (PHI·용량 문제로 `.gitignore` 처리).
새로 클론하면 `data/`와 `models/`가 비어 있는 것이 정상입니다.

그래서 이렇게 동작합니다:
- `python src/vitals_train.py` → **합성 데이터로 정상 작동** ✅
- `python src/train.py` (정적 트랙) → 데이터 필요, 없으면 `FileNotFoundError`
- `/predict` API → `models/best_model.pkl` 없으면 503 반환
- `pytest` → **통과** ✅ (데이터가 필요한 테스트는 자동으로 skip)

정적 트랙까지 돌리려면 `data/`에 아래 파일을 넣으세요 (로더가 자동 압축 해제):

- `Heart Failure Clinical Records Dataset.zip` → 299행, 타깃 `DEATH_EVENT`
- `Cardiovascular Disease dataset.zip` → 70,000행, `;` 구분

---

## 테스트

```bash
pytest -q
```

합성 데이터를 쓰는 시계열 테스트는 **항상 실행**되고, 외부 데이터가 필요한 정적 트랙 테스트는
데이터가 없으면 자동으로 skip됩니다. 그래서 **새로 클론한 상태에서도 CI가 초록색**입니다.

---

## 문서

| 문서 | 내용 |
|---|---|
| [`docs/README.md`](docs/README.md) | 📑 문서 전체 지도 (여기부터) |
| [`docs/STATUS.md`](docs/STATUS.md) | 현재 진행 상황 요약 |
| [`docs/competition-strategy.md`](docs/competition-strategy.md) | 대회 전략 및 심사 기준 정렬 |
| [`docs/proposal-draft.md`](docs/proposal-draft.md) | 예선 제안서 초안 (30장) |
| [`docs/differentiation.md`](docs/differentiation.md) | 본선 발표·질의응답 대비 |
| [`docs/server-runbook.md`](docs/server-runbook.md) | GPU 서버 환경 구축 절차 |
| [`CLAUDE.md`](CLAUDE.md) | 개발 규칙 (Claude Code / 개발자용) |

---

## 정적 트랙 API

초기 버전인 정적 트랙은 FastAPI 서버를 제공합니다.

```bash
uvicorn main:app --app-dir src --reload   # 문서: http://localhost:8000/docs
```

| 메서드 | 엔드포인트 | 설명 |
|---|---|---|
| `GET` | `/` | 서비스 정보 |
| `GET` | `/health` | 상태 + 모델 로드 여부 |
| `POST` | `/predict` | 위험 확률 + 라벨 + SHAP 상위 3개 요인 |
| `POST` | `/convert-omop` | OMOP CDM v5.4 테이블로 변환 |

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 65, "anaemia": 0, "creatinine_phosphokinase": 146,
       "diabetes": 0, "ejection_fraction": 20, "high_blood_pressure": 1,
       "platelets": 162000, "serum_creatinine": 1.3, "serum_sodium": 129,
       "sex": 1, "smoking": 1}'
```

---

> **⚠️ 면책**: 연구·교육용 소프트웨어입니다. 의료기기가 아니며 임상 의사결정에 사용할 수 없습니다.

## 라이선스

[MIT License](LICENSE)
