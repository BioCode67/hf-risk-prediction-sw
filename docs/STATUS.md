# 프로젝트 현황 정리 — 활력징후 기반 심정지 조기경보

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
  → cost-sensitive XGBoost  vs  NEWS(임상표준 baseline)
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

## 6. 코드 자산 (repo)
- 브랜치: claude/cardiac-arrest-early-warning-07fq9e (github.com/BioCode67/hf-risk-prediction-sw)
- 모듈:
  - vitals_data.py     : 합성/KHTH/MIMIC 어댑터, 정제, 윈도우, 개인화·정적 피처, 분할, lead-time
  - vitals_train.py    : XGBoost vs NEWS, AUPRC·민감도@특이도·알람부담·lead-time
  - vitals_explain.py  : SHAP(전역/윈도우별)
  - vitals_report.py   : 그림 5종(PR-curve·궤적·lead-time·알람부담) render_report()
  - vitals_phenotype.py: 심정지 표현형 군집 + 히트맵
  - mimic_explore.py   : 실 MIMIC-IV 탐색 + --scan-arrest/--arrest-counts/--model(원커맨드)
  - omop_explore.py    : OMOP CDM 탐색(참고)
- 품질: 테스트 40개 green, CI(3.11/3.12) 반영, 전부 합성/네트워크 불필요로 항상 실행
- 원커맨드: python src/mimic_explore.py <MIMIC경로> --model
  → 지표 + 알람부담 + lead-time + 그림 5종 + 표현형 일괄 생성

## 7. 문서
- docs/competition-strategy.md : 우승 전략 & 제안서 설계(배점 정렬)
- docs/proposal-draft.md       : 예선 제안서 30장 골격 초안(예비수치·그림 반영)
- docs/STATUS.md               : (본 문서) 프로젝트 현황 정리

## 8. 남은 일 (To-Do)
- [ ] 전체 MIMIC-IV: CITI 인증 → 다운로드 → --model로 실학습 수치 산출
- [ ] 안심존 방문 예약(예선 +5 가점 + 실제 값 범위·결측 확인)
- [ ] 제안서 최종화: 팀 정보·문구·30장 분량 확장, PDF 변환, 데이터셋명 명기, 출처 표기
- [ ] 안심존 사전신고 패키지 목록 확정(numpy pandas scikit-learn xgboost shap matplotlib)
- [ ] (선택) 표현형별 대응 전략, 2D PCA 산점도 등 시각화 보강
