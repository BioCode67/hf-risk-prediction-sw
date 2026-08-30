# 프로젝트 현황 정리 — 활력징후 기반 심정지 조기경보

> 최종 갱신: 2026-08-31.
>
> | 트랙 | 대상 | 상태 |
> |---|---|---|
> | **K-Health 경진대회** | 경북대 활력징후 → 심정지 | **본선 진출 확정 (8/27 통보).** 본선 제출 11/6 16:00 |
> | ~~KMEDIhub 인턴~~ | Challenge-2019 → 패혈증 | **종료.** 산출물 유지보수만, 신규 작업 없음 |
>
> **인턴 트랙은 종료되었습니다.** 대시보드([데모](https://web-ebon-rho-81.vercel.app))·발표자료·
> 문서가 모두 완료되어 배포된 상태이며, **더 이상 작업하지 않습니다.** 이 트랙의 검증 결과는
> 제안서의 실현성 근거로만 인용합니다(§5-1). `web/`·`docs/sepsis-dashboard.md`·
> `docs/intern-track-briefing.md`는 완료된 산출물이므로 손대지 마세요.
>
> ## 지금 확정된 것 (2026-08-13)
>
> | 항목 | 상태 |
> |---|---|
> | 팀명 | **PRODROME (전조)** — 등록 완료, 1인 팀 |
> | 선택 데이터 | 경북대병원 활력징후 (`KHTH_PINFO`/`KHTH_VITAL`) — 확정 |
> | 안심존 방문 | **완료** (건양대 데이터안심의료존) → 예선 가점 +5 확보 |
> | MIMIC-IV 접근 | **자격 심사 통과** (CITI 이수 + 소속기관 승인) → 대조군 포함 전체 데이터 활용 가능 |
> | 연산 자원 | KOREN AI Cloud (Cheetah 플랫폼) 접속 확보 — H200 141GB × 2, 3개월, 스토리지 15TB |
> | 제안서 | **29장 완성** (제출 PDF 기준) — 빌드는 아래 §7 참조 |
> | 반입 | 외부 데이터·학습 모델·가중치 모두 승인 시 가능 |
> | 반출 | 분석 결과는 가능, **모델은 불가 전제** |
> | 안심존 이용 개시 | 사전신고 후 약 1주 |
>
> ## 본선 확정 사실 (DACON 8/27·8/28 메일)
>
> | 항목 | 내용 |
> |---|---|
> | **예선 결과** | **통과 — 본선 진출** (상위 15팀) |
> | 분석 장소 | 대구 계명대 동산의료원 안심활용센터 — **GPU 불가**, Python 버전 지정 가능, 10석 |
> | 패키지 사전신청 | **8/31(월) 10:00 회신 마감** → 전 팀 취합 후 일괄 설치 **2~3주** 소요 |
> | 분석 시작 | 설치 완료 후 (9월 중순 예상). 본선 기간 9/1~11/6 |
> | 방문 증빙 | **본선 신설** — 신청·승인 메일 + 외부 사진, 방문마다. 미제출 = 미방문 처리 |
> | 방문 점수 | 1회 2점 · 2회 4점 · 3회 6점 · 5회 8점 · **7회+ 10점** → 7회 이상 목표 |
> | 제출 3종 | 결과보고서 PDF · 발표자료 PDF · 방문증빙 ZIP — 11/6(금) 16:00 |
> | 제안서 연속성 | 예선 제안서 일부 수정 가능, **데이터 변경·대폭 변경 불가** |
> | 발표평가 | 11/13(금) — 팀원 1인 이상 필참. 시상식 11/27 |
>
> 사전신청 회신·방문 메일 템플릿·체크리스트: scratchpad `finals/` (세션 산출물)

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
- 개인화 기저선 피처 (통제 실험, 시드 5회): AUPRC 0.784 → 0.834 (상대 +6.6%, 4/5 개선)
  → 더 중요한 것은 분산 — 개인화를 빼면 시드 간 표준편차가 4.7배(0.008 → 0.037).
  기여는 평균 성능보다 **평가 환자군이 바뀌어도 성능이 유지되는가** 쪽에 있음.
  재현: `python scripts/ablate_personalized.py --seeds 42,1,7,13,2024`
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
   절대값입니다. "기여한다"는 맞고 "핵심"은 과장입니다. (합성 심정지 데이터의 통제
   실험 결과는 실데이터에서 그대로 재현되지 않습니다.)
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
- 품질: 테스트 88개 (76 pass / 12 skip — 데이터 의존분만 skip), CI(3.11/3.12) 반영.
  합성 데이터를 내장해 데이터 없이도 실행됨
  주의: `pytest` 실행 파일이 uv 격리 환경일 수 있음 → `python -m pytest` 사용
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
- scripts/build_proposal.js    : 예선 제출본 생성. 두 벌을 빌드합니다
    · Word 편집용: `node scripts/build_proposal.js` → models/PRODROME_제안서.docx
    · 제출 PDF용 : `PROPOSAL_FONT=NanumGothic PROPOSAL_OUT=models/_pdfsrc.docx node scripts/build_proposal.js`
      (맑은 고딕이 없는 환경에서 LibreOffice가 한글을 중국어 폰트로 치환하는 것을 피합니다)
    · 폰트가 바뀌면 페이지가 밀리므로 목차는 `PROPOSAL_PAGES='{"키":쪽}'` 로 빌드별 지정
- scripts/make_pipeline_figure.py / make_ward_figure.py : 파이프라인 개념도 · 병동 알람 그래프
- scripts/make_baseline_figure.py / make_labeling_figure.py : 개념도 2종
- scripts/ablate_personalized.py : 개인화 피처 통제 실험 (--mimic / --challenge2019 지원)

## 8. 남은 일 (To-Do)

### ~~예선~~ — 완료. 제출(8/14) → **본선 진출 확정(8/27)**

### 지금 즉시 — 8/31(월) 10:00 마감 ⚠️
- [ ] **분석환경 사전신청 회신** — dacon@dacon.io + 602626@kyuh.ac.kr 둘 다.
      Python 3.11 + 패키지 138종(버전 고정, 이 컨테이너의 테스트 green 구성과 동일)
      + 사전학습 모델(XGBoost JSON) + 코드·나눔고딕 폰트 반입 선언. GPU 불필요

**완료된 것:** §3 상용 제품(VUNO Med-DeepCARS) 실명·근거 논문 반영 · 참고문헌 10건
실재 확인 · §7-1 실측치 교체 · 개념도 2종(개인 기저선·within-patient) 작성 ·
§6.2 개인화 피처 통제 실험 · §8 병동 규모 환산 · §9 안심존 운영 조건·성공 기준 ·
§10 윤리 신설

### 9월 초 — 패키지 설치 대기 2~3주 동안 (안심존 밖)

- [ ] **MIMIC-IV 사전학습 모델 생성** — KOREN GPU에서.
      `--arrest-counts` 로 표본 확인 → `--model --gpu` → **XGBoost JSON 포맷 저장**
      (사전신청서에 JSON으로 선언 — pickle 아님)
- [ ] **사건 유형별 개인화 유효성 비교** — `scripts/ablate_personalized.py` 를
      `--mimic`(심정지)과 `--challenge2019`(패혈증) 양쪽에 돌려 비교.
      "개인 기저선 이탈이 어떤 사건에서 유효한가"는 §9-5의 3번 검증 항목이자
      본 과제 고유의 기여가 될 수 있음

### 안심존 분석 기간 (9월 중순~10월)

- [ ] 첫 방문에서 환경 검증: `python -m pytest -q` (데이터 없이 green이어야 정상)
- [ ] 경북대 데이터로 제안서 §9.5의 5단계 수행 + gap 상향 비교(누수 검증)
- [ ] **방문 7회 이상** — 매회 신청 메일(CC 602626@kyuh.ac.kr) + 외부 사진 보관
- [ ] **반출 신청 늦어도 10/16** (심의 1~2주). 반출물 = 수치·그림·표. MIMIC 원자료 반입 금지(DUA)
- [ ] 11/6 16:00: 결과보고서 PDF + 발표자료 PDF + 방문증빙 ZIP · 11/13 발표 필참

### 하지 않을 일 (인턴 트랙 — 종료됨)

아래는 완료된 산출물입니다. **작업하지 마세요.**

- `web/` 대시보드, `docs/sepsis-dashboard.md`, `docs/intern-track-briefing.md`,
  `docs/deploy.md` — 배포 완료 상태 유지
- 라벨 재정의("악화" 실험), ONNX 전환 — 인턴 트랙의 후속 아이디어였으며 대회와 무관
