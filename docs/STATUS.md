# 프로젝트 현황 정리 — 활력징후 기반 심정지 조기경보

> 최종 갱신: 2026-07-29. 두 갈래가 동시에 굴러갑니다.
>
> | 트랙 | 대상 | 상태 |
> |---|---|---|
> | **K-Health 경진대회** | 경북대 활력징후 → 심정지 | 예선 제안서 준비 중, **마감 8/14 16:00 (D-16)** |
> | **KMEDIhub 인턴** | Challenge-2019 → 패혈증 | **완료·배포됨** → [데모](https://web-ebon-rho-81.vercel.app) · [`sepsis-dashboard.md`](sepsis-dashboard.md) |
>
> 인턴 트랙은 같은 파이프라인을 공개 실데이터로 끝까지 검증한 것입니다. 대회 제출물이
> 아니지만, 제안서의 실현성 근거로 그대로 쓸 수 있습니다(§5-1).

## 1. 대회 개요
- 대회: 2026 K-Health 미개방 의료데이터 활용 경진대회 (운영: 데이콘)
- 일정: 예선(아이디어 제안서 PDF 30장) 마감 8/14 16:00 → 본선 진출 발표 8/28 → 본선 제출 11/6
- 평가: [예선] 시의성20·실현성30·참신성30·파급성20 (상위 15팀) / [본선] 수행30+발표70
- 가점: 예선 기간 안심존 1회 방문 +5점
- 제약: 본선은 안심존(오프라인 폐쇄망), 사전신고 패키지만, 결과물 반출 심의 1~2주

## 2. 주제 & 핵심 논지 (확정)
- 주제: 활력징후 시계열 → 원내 심정지 조기경보
- 논지: "개인 기저선 이탈 기반, 설명가능한 심정지 조기경보로 alarm fatigue 해결"
- 차별점(참신성): ①개인화 기저선 이탈(환자=자기 대조군) ②설명가능성(XAI)
  ③case-only 데이터를 within-patient 설계로 전환 ④심정지 표현형 발견
- 정체성: 정확도 경쟁이 아니라 "오경보 저감 + 설명가능성"

## 3. 데이터
- 선택 데이터: [대구] 경북대병원 입원환자 활력징후 (KHTH_PINFO, KHTH_VITAL)
  - N=573명 (2023~25, 20~80세), 전원 심정지 후 사망 = 대조군 없음(case-only)
  - 입원 24h 이내 심정지 제외 → 전 환자 ≥24h 관찰 확보
  - 스키마: KHTH_PINFO(PATID, AGE(연령대), SEX, INDD, OUDD, CARDT(심정지시각), DEATHDT)
           KHTH_VITAL(PATID, INDD, VSDT, VS_GBN=HR/SBP/DBP/BT/SPO2/RR, VS_RSLT)
           조인키 PATID+INDD, 라벨은 CARDT(정확한 심정지 시각) 기준
- 개발·검증 데이터: MIMIC-IV
  - Demo(100명): 인증 불필요, 다운로드 완료 → 구조/품질 검증 완료(대조군만)
  - 전체 MIMIC-IV: 무료 CITI 인증 필요(2~7일) → 실제 심정지 표본 확보용
  - 심정지 정의: procedureevents itemid 225466 "Cardiac Arrest"(+225475/225464)
  - 역할: 경북대에 없는 대조군 제공 → 오경보 정직 측정
- 참고: 실 OMOP 심부전 데이터는 본선에만 제공(예선 샘플=Synthea 합성)

## 4. 구축한 파이프라인 (한 흐름)
데이터(합성/MIMIC/KHTH) → 정제(센서 아티팩트·화씨 보정)
  → 슬라이딩 윈도우 통계 피처(평균/표준편차/최소/최대/최근/추세/변화량 + shock index)
  → 개인 기저선 이탈 피처 + AGE/SEX 정적 피처
  → 라벨(within-patient, 향후 1h 내 심정지)
  → cost-sensitive XGBoost (선택: Optuna 튜닝 / GPU) vs NEWS(임상표준 baseline)
  → 평가: AUPRC·AUROC·민감도@95%특이도·알람부담@민감도·lead-time
  → 설명(SHAP) + 발견(심정지 표현형 군집)
  → 발표용 그림 5종 자동 생성

## 5. 예비 검증 결과 (합성/MIMIC 개발 데이터 — 실데이터 아님)
- AUPRC: XGBoost 0.77~0.97 vs NEWS 0.42~0.61 (ROC-AUC는 둘 다 ~0.99로 유사)
  → "ROC만 보면 안 보이는 오경보 격차" 실증
- 개인화 기저선 피처 추가: AUPRC 0.76 → 0.84
- 알람부담(동일 90% 민감도): XGBoost 오경보 0.45·알람 2.0/100 vs NEWS 0.61·2.9/100
  → 96~99% 고민감도에서 NEWS 알람 16~36/100 폭증, XGBoost는 ~3/100 유지
- lead-time: 심정지 median 2~3시간 전 경보
- SHAP 상위 요인: 호흡수·SpO2·맥박의 추세(slope)
- 표현형: 호흡성형 / 순환성형 / 혼합 3아형 자동 발견

## 5-1. 공개 실데이터 검증 (Challenge-2019, 패혈증) — 완료

인턴 트랙에서 같은 파이프라인을 실데이터로 끝까지 돌린 결과입니다. 상세는
[`sepsis-dashboard.md`](sepsis-dashboard.md).

- 규모: 20,336명 로드 → train 15,886 / **test 3,972명 (126,558 윈도우)**, 양성률 1.19%
- AUPRC 0.027 (기준선 0.0119 대비 2.3배), ROC-AUC 0.679
- 알람 부담 (같은 검출률에서): 50% → **40% 감소**(24.8 vs 41.3/100),
  70% → 23%, 90%는 NEWS 특이도 0이라 비교 불성립
- lead-time: 첫 경보 기준 중앙값 33h, 검출 131/262명

**제안서에 쓸 때 반드시 조건을 붙여야 하는 것 3가지:**

1. **"알람 40% 감소"는 전체 데이터 + 50% 검출률에서만.** 4,000명 부분집합에서는 10%로
   떨어집니다. 데이터 규모 의존성이 큽니다.
2. **개인 기저선 피처는 주역이 아닙니다.** 전역 기여도 4위·6위이고 1~3위는 호흡수·체온
   절대값입니다. "기여한다"는 맞고 "핵심"은 과장입니다. (합성 데이터의 0.76→0.84는
   실데이터에서 그대로 재현되지 않습니다.)
3. **극단 상태에서 점수가 역전됩니다.** SpO2 70% 미만 윈도우 232개 중 6시간 내 발병
   0건 — 이미 붕괴한 환자는 발병 시점을 지났거나 다른 원인으로 악화 중이라 패혈증
   확률이 실제로 낮습니다. 모델은 맞지만 화면이 "안정"으로 읽히면 위험합니다.
   대시보드는 모델과 무관한 생리학적 채널로 이 경우 경고를 띄웁니다.

3번은 숨기지 말고 방법론으로 다루는 편이 참신성·실현성 점수에 유리합니다 — 라벨 정의의
한계를 발견하고 대응한 것 자체가 서사가 됩니다.

## 6. 코드 자산 (repo)
- 브랜치: claude/cardiac-arrest-early-warning-07fq9e (github.com/BioCode67/hf-risk-prediction-sw)
- 모듈:
  - vitals_data.py     : 합성/KHTH/MIMIC 어댑터, 정제, 윈도우, 개인화·정적 피처, 분할, lead-time
  - vitals_train.py    : XGBoost vs NEWS, AUPRC·민감도@특이도·알람부담·lead-time, Optuna 튜닝·GPU(CUDA)
                         + 모델 레지스트리 --compare (XGBoost/LightGBM/CatBoost/RandomForest/Logistic)
                         ※ 합성 데이터에서는 Logistic이 1위 — 합성 신호가 단조 드리프트라 선형에 유리한
                           탓일 가능성이 큼. 실데이터(Challenge-2019) 재현 전까지 인용 금지
  - vitals_explain.py  : SHAP(전역/윈도우별)
  - vitals_report.py   : 그림 5종(PR-curve·궤적·lead-time·알람부담) render_report()
  - vitals_phenotype.py: 심정지 표현형 군집 + 히트맵
  - mimic_explore.py   : 실 MIMIC-IV 탐색 + --scan-arrest/--arrest-counts/--model(원커맨드)
  - omop_explore.py    : OMOP CDM 탐색(참고)
  - vitals_narrate.py  : SHAP 근거 → 한국어 문장. build_evidence(결정론적·오프라인)와
                         narrate(Groq LLM)로 분리. LLM은 판단하지 않고 표현만 담당하며,
                         MIMIC/KHTH는 DUA 위반이라 narrate가 거부함
  - vitals_api.py      : 대시보드용 FastAPI. 사전계산 아티팩트를 읽어 0.1초에 기동
- 대시보드(web/): Next.js. 근거를 전부 미리 계산해 정적 JSON으로 굽기 때문에 배포에는
  파이썬도 ML 라이브러리도 필요 없음(706MB → 0). scripts/export_dashboard.py로 생성
- 품질: 테스트 76개 green, CI(3.11/3.12) 반영, 전부 합성/네트워크 불필요로 항상 실행
- 원커맨드: python src/mimic_explore.py <MIMIC경로> --model [--gpu] [--tune] [--trials=N]
  → (Optuna 튜닝) + 지표 + 알람부담 + lead-time + 그림 5종 + 표현형 일괄 생성
- 협업: 선생님(로컬)이 GPU/CUDA·Optuna·UTF-8 CLI 커밋 → 원커맨드에 통합 완료

## 7. 문서
- docs/competition-strategy.md : 우승 전략 & 제안서 설계(배점 정렬)
- docs/proposal-draft.md       : 예선 제안서 초안(전 절 확장 + 결과표 + 그림)
- docs/differentiation.md      : 본선 발표·질의응답(Q&A) 대비 + 경쟁우위
- docs/sepsis-dashboard.md     : 공개 실데이터 검증 결과·한계·배포 구조 (인턴 트랙)
- docs/deploy.md               : 대시보드 배포 절차(아티팩트 → Vercel)
- web/README.md                : 대시보드 구조·실행
- docs/STATUS.md               : (본 문서) 프로젝트 현황 정리
- 제안서 Word(docx) 초안 v3     : 채팅으로 전달(그림 5종·결과표 임베드)

## 8. 남은 일 (To-Do)

기한이 있는 것부터. 오늘 2026-07-29 기준 예선까지 **16일**.

- [ ] **오늘: PhysioNet CITI 인증 신청** — 무료지만 2~7일 걸려 임계경로.
      완료되면 전체 MIMIC-IV → `--model`로 심정지 실학습 수치 산출
- [ ] **오늘: 안심존 방문 예약** — 예선 기간에만 유효한 **+5점**. 상위 15팀
      커트라인에서 결정적일 수 있음. 실제 값 범위·결측도 함께 확인
- [ ] 제안서 최종화: 팀 정보·문구·30장 분량 확장, PDF 변환, 데이터셋명 명기, 출처 표기
      — §5-1의 조건 3가지를 반드시 반영할 것
- [ ] 안심존 사전신고 패키지 목록 확정(numpy pandas scikit-learn xgboost shap matplotlib)
      — LLM(narrate)은 폐쇄망에서 못 쓰므로 목록에 넣지 않음. build_evidence만 사용
- [ ] (선택) 표현형별 대응 전략, 2D PCA 산점도 등 시각화 보강

### 인턴 트랙에서 이어서 할 수 있는 것 (급하지 않음)
- 라벨을 "6시간 내 발병" 대신 "악화"로 재정의해 §5-1의 3번 거동이 사라지는지 확인
- 실시간 추론이 필요해지면 XGBoost → ONNX (실측: 226MB → 0.5MB 모델 + 56MB 런타임,
  추론 4.9ms → 0.12ms, 오차 3.6e-07)
