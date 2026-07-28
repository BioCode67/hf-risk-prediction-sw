# `legacy/` — 정적 심부전 위험 예측 (초기 버전)

> 이 폴더는 **대회 출품작이 아닙니다.** 학습하실 때는 넘어가셔도 됩니다.

## 이게 뭔가요

프로젝트 초기에 만든 파이프라인입니다. **한 시점의 검사값**(나이·박출률·혈청 크레아티닌 등)으로
심부전 환자의 사망 위험을 예측합니다.

이후 주제가 **활력징후 시계열 → 심정지 조기경보**로 바뀌면서 본체가 `src/`로 옮겨갔고,
이 코드는 여기에 보관합니다.

| | 정적 트랙 (여기) | 시계열 트랙 (`src/`) |
|---|---|---|
| 입력 | 한 시점 검사값 | 시간에 따른 활력징후 |
| 예측 | 심부전 사망 위험 | N시간 내 심정지 |
| 상태 | 보관 | **개발 중 (대회 출품작)** |

## 왜 지우지 않았나요

1. **FastAPI 서버** — 완성된 REST API가 있습니다. SW저작권 등록 시 "완결된 소프트웨어"로서
   가치가 있습니다.
2. **OMOP CDM v5.4 변환** — `to_omop_cdm()`이 환자 데이터를 국제 표준 포맷으로 바꿉니다.
   대회 본선에서 OMOP 데이터가 제공되면 재사용할 수 있습니다.
3. 동작하는 코드이고, 테스트도 통과합니다.

## 파일

| 파일 | 역할 |
|---|---|
| `data_loader.py` | 로드·전처리·층화분할 + `to_omop_cdm()` |
| `train.py` | LightGBM(Optuna, AUPRC) + XGBoost → `models/best_model.pkl` |
| `explainability.py` | SHAP 전역 top-5 + 환자별 top-3 |
| `main.py` | FastAPI 서버 |
| `tests/test_pipeline.py` | 위 4개의 테스트 |

## 실행

⚠️ `data/`에 원본 데이터가 있어야 합니다. 없으면 `FileNotFoundError`가 납니다
(새로 클론한 상태에서는 정상적인 동작입니다).

```bash
python legacy/data_loader.py     # 데이터 확인·전처리
python legacy/train.py           # 학습 → models/best_model.pkl
python legacy/explainability.py  # SHAP 리포트

uvicorn main:app --app-dir legacy --reload   # API 서버 (http://localhost:8000/docs)
```

필요한 데이터:
- `Heart Failure Clinical Records Dataset.zip` → 299행, 타깃 `DEATH_EVENT`
- `Cardiovascular Disease dataset.zip` → 70,000행, `;` 구분

## API

| 메서드 | 엔드포인트 | 설명 |
|---|---|---|
| `GET` | `/` | 서비스 정보 |
| `GET` | `/health` | 상태 + 모델 로드 여부 |
| `POST` | `/predict` | 위험 확률 + SHAP 상위 3개 요인 |
| `POST` | `/convert-omop` | OMOP CDM v5.4 테이블 변환 |

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 65, "anaemia": 0, "creatinine_phosphokinase": 146,
       "diabetes": 0, "ejection_fraction": 20, "high_blood_pressure": 1,
       "platelets": 162000, "serum_creatinine": 1.3, "serum_sodium": 129,
       "sex": 1, "smoking": 1}'
```

## 구조상 참고

`legacy/`도 `src/`처럼 **설치되는 패키지가 아니라 import 경로**입니다. 내부 모듈끼리
`from data_loader import ...`처럼 이름으로 서로를 부르고, 저장소 루트의 `conftest.py`가
테스트 실행 시 이 폴더를 `sys.path`에 넣어줍니다.
