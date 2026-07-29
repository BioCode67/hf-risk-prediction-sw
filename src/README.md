# `src/` 모듈 지도

어떤 파일이 무슨 일을 하는지, 어떤 순서로 불리는지 정리했습니다.

---

## 시계열 트랙 — 이 프로젝트의 본체

활력징후 시계열로 심정지를 조기경보합니다. 실행 순서대로:

### `vitals_data.py` — 데이터 준비 (813줄, 가장 큼)

모든 데이터가 여기를 거쳐 같은 형태가 됩니다.

| 함수 | 역할 |
|---|---|
| `generate_synthetic_cohort()` | 합성 환자 생성 (데이터 없이 실행 가능) |
| `cohort_from_khth()` | 경북대 데이터 어댑터 (본선) |
| `cohort_from_mimic()` | MIMIC-IV 어댑터 |
| `cohort_from_challenge2019()` | PhysioNet Challenge 2019 어댑터 (패혈증) |
| `cohort_from_challenge2012()` | PhysioNet Challenge 2012 어댑터 (ICU 사망, 4,000명) |
| `sanitize_vitals()` | 불가능한 값 제거, 화씨→섭씨 변환 |
| `build_windows()` | 슬라이딩 윈도우 + "N시간 내 심정지" 라벨링 |
| `add_personalized_features()` | 개인 기저선 대비 편차 — 차별점 |
| `add_static_features()` | 연령대·성별 추가 |
| `patient_level_split()` | 환자 단위 분할 (누수 방지) |

주요 상수: `OBSERVATION_WINDOW_HOURS=8`, `PREDICTION_HORIZON_HOURS=1`, `VITALS`, `PLAUSIBLE_RANGE`

### `vitals_train.py` — 학습과 평가

| 함수 | 역할 |
|---|---|
| `train_xgboost()` | cost-sensitive XGBoost 학습 |
| `tune_xgboost()` | Optuna 튜닝 (환자 그룹 CV, AUPRC 최적화) |
| `compute_news_scores()` | NEWS 임상 점수 (비교군) |
| `evaluate()` | AUPRC·ROC·민감도@특이도 계산 |
| `alarm_burden()` | 동일 민감도에서의 알람 횟수 — 핵심 지표 |
| `lead_time_summary()` | 사건 몇 시간 전에 경보했는지 |

### `vitals_explain.py` — 설명

SHAP으로 전역 기여도와 개별 윈도우 근거를 뽑습니다.

### `vitals_narrate.py` — 경보를 문장으로

SHAP 숫자를 병동에서 읽을 한국어 문장으로 옮깁니다. 두 단계로 분리돼 있습니다.

| 함수 | 성격 |
|---|---|
| `build_evidence()` | 결정론적. SHAP 상위 요인 + 현재값 + 개인 기저선 + NEWS를 dict로 |
| `format_evidence()` | 그 dict를 텍스트 블록으로. LLM 없이도 이것만으로 설명이 됩니다 |
| `narrate()` | Groq API로 문장 생성. LLM은 판단하지 않고 표현만 담당합니다 |

`narrate()`만 외부 API를 씁니다. 폐쇄망(안심존)에서는 앞의 두 개만 쓰세요.
MIMIC·경북대 데이터는 DUA상 외부 전송이 금지라 `source="mimic"`/`"khth"` 이면
`narrate()`가 `PermissionError`를 냅니다. 키는 `.env`의 `GROQ_API_KEY`에서
읽으며 `.env`는 git에 올라가지 않습니다. 의존성은 `requirements-llm.txt`.

### `vitals_report.py` — 그림

`render_report()` 하나로 PR곡선·알람부담·악화궤적·lead-time 그림을 전부 생성합니다.

### `vitals_phenotype.py` — 표현형 발견

심정지 직전 궤적을 KMeans로 군집화 → 호흡성/순환성/혼합형 아형 발견 + 히트맵.

---

## 데이터 탐색·실행 도구

각 데이터를 한 번의 명령으로 끝까지 돌리는 진입점입니다.

| 파일 | 대상 | 주요 옵션 |
|---|---|---|
| `sepsis_explore.py` | PhysioNet Challenge 2019 (패혈증) | `--horizon` `--tune` `--gpu` `--max-files` |
| `mortality_explore.py` | PhysioNet Challenge 2012 (ICU 사망) | `--horizon` `--outcomes` `--tune` `--gpu` |
| `mimic_explore.py` | MIMIC-IV | `--model` `--scan-arrest` `--arrest-counts` `--gpu` |
| `omop_explore.py` | OMOP CDM 폴더 | — |

---

## 딥러닝 벤치마크 (선택)

멘토 요청으로 만든 XGBoost 대조군입니다. torch가 필요하며,
`requirements-torch.txt`로 따로 설치합니다 (CI는 torch 없이 돕니다).

| 파일 | 역할 |
|---|---|
| `utils.py` | 전진대치 → 평균/0 대치 → z-score 정규화 |
| `dataset.py` | 가변길이 시퀀스 배칭 (padding + masking) |
| `model.py` | LSTM/GRU 분류기 |

실행은 저장소 루트의 `train_dl.py`입니다.

---

## 구조에 관한 중요한 제약

`src/`는 설치되는 패키지가 아니라 import 경로입니다.

모듈끼리 이렇게 부릅니다:

```python
from vitals_data import build_windows      # 현재 방식
from src.vitals_data import build_windows  # 작동하지 않음
```

이게 성립하는 이유는 각 진입점이 `src/`를 경로에 넣기 때문입니다:

| 진입점 | 방식 |
|---|---|
| 스크립트 | `python src/vitals_train.py` — 실행 파일의 폴더가 자동으로 경로에 포함 |
| 테스트 | 저장소 루트 `conftest.py`가 `sys.path`에 `src/` 추가 |
| CI | `sys.path.insert(0, 'src')` |
| 노트북 | `sys.path.insert(0, str(REPO / "src"))` |

그래서 파일을 하위 폴더로 옮기면 위 네 곳이 전부 깨집니다. 폴더로 나누는 대신
파일명 접두사(`vitals_*`, `*_explore`)로 트랙을 구분하고 있습니다.

바꿔야 한다면 네 진입점을 모두 함께 수정하고 `pytest -q`로 검증하세요.
(본선 안심존이 오프라인 폐쇄망이라 `pip install -e .` 같은 설치 단계를 요구하지 않는 편이
안전합니다 — 이것도 평평한 구조를 유지하는 이유입니다.)

> 초기 버전인 정적 심부전 예측 트랙은 [`../legacy/`](../legacy/README.md)로 옮겼습니다.
> 같은 이유로 그쪽도 별도의 import 경로입니다.
