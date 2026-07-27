# GPU 서버 실행 런북 (개발·인턴 트랙)

> 회사 GPU 서버(RTX A6000 ×2)에서 파이프라인을 GPU로 돌리는 절차.
> **주의**: 이 서버는 개발·검증용이다. 공모전 **본선 안심존은 별개**(오프라인·CPU·사전신고
> 패키지)이므로 개인 GPU를 쓸 수 없고, 안심존엔 코드만 반입한다.

## 0. 접속 (회사망에서)
```bash
ssh jhkim@222.103.107.7 -p 4132   # 방화벽 IP 제한 → 회사망에서만
nvidia-smi                        # A6000 ×2 보이는지 확인
```

## 1. 환경 구축 (최초 1회)
```bash
cd /workspace
git clone https://github.com/BioCode67/hf-risk-prediction-sw.git
cd hf-risk-prediction-sw
git checkout claude/cardiac-arrest-early-warning-07fq9e

python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt          # xgboost>=2.0 = CUDA(device="cuda") 지원
python -c "import xgboost; print('xgboost', xgboost.__version__)"
```

## 2. 지금 바로 — GPU + Optuna 스모크 테스트 (실데이터 불필요)
합성 데이터로 GPU·Optuna 스택이 A6000에서 도는지 검증한다.
```bash
python src/vitals_train.py --gpu --tune          # Optuna 튜닝(GPU) + XGBoost vs NEWS
# 다른 터미널에서: watch -n1 nvidia-smi  → 학습 중 GPU 사용률 확인
```
- 정상이면 Optuna best CV AUPRC 로그 + XGBoost/NEWS 지표가 출력된다.
- `[gpu] CUDA unavailable ...` 가 뜨면 CPU로 자동 폴백된다(코드가 처리) → 드라이버/설치 점검.

전체 리포트(그림 포함)까지 합성으로 보려면:
```bash
python src/vitals_report.py                       # PR·알람부담·궤적·lead-time 그림 생성(models/)
python src/vitals_phenotype.py                    # 표현형 히트맵
```

## 2b. 지금 바로 — 실제 오픈데이터로 (양성 표본 포함, 인증 불필요) ★추천
심정지 특화 공개데이터(인증 없는)는 없다. 가장 가까운 **개방·즉시 사용 + 양성/대조군 모두** 있는
대안은 **PhysioNet/CinC Challenge 2019(sepsis)** — 40k 환자, 시간별 활력징후 + 악화 라벨.
심정지는 아니지만 "활력징후 → 임박 악화사건" 구조가 동일해 우리 파이프라인이 그대로 돈다.
```bash
# (a) 인증 없이 다운로드 (training_setA/B: p*.psv, 환자 1인당 1파일)
wget -r -N -c -np https://physionet.org/files/challenge-2019/1.0.0/
#    -> physionet.org/files/challenge-2019/1.0.0/training/training_setA
# (b) 우리 파이프라인 그대로 (먼저 일부만 빠르게, 그다음 전체)
python src/sepsis_explore.py <.../training_setA> --max-files=4000 --tune --gpu
python src/sepsis_explore.py <.../training_setA> --tune --gpu           # 전체
```
출력: XGBoost vs NEWS(AUPRC·민감도@특이도·오경보) + 알람부담 + lead-time + 그림 5종 + 표현형.
**대조군이 있어 오경보(특이도)를 진짜로 측정**할 수 있다 → 방법 검증에 이상적.

## 2c. MIMIC-IV 접근 신청 (CITI + PhysioNet) — 며칠 소요
MIMIC-IV("미믹 포")는 다운로드에 **credentialed 인증**이 필요하다(무료, 승인까지 수일~2주).
1. **PhysioNet 계정** 생성: https://physionet.org/register/
2. **CITI 교육 이수**: "Data or Specimens Only Research" 과정 수료 → 완료 리포트(PDF) 확보.
   PhysioNet 프로필의 *Training* 항목에서 어떤 CITI 모듈이 필요한지 안내를 따른다.
3. **Credentialing 신청**: PhysioNet 프로필 작성 + **레퍼런스**(본인 아닌 지도교수/멘토 등)를
   기재 → 심사 승인 대기.
4. 승인 후 **MIMIC-IV 프로젝트 페이지에서 DUA 서명**: https://physionet.org/content/mimiciv/
5. **다운로드**(ICU 모듈 필수: chartevents, procedureevents, d_items, icustays):
```bash
wget -r -N -c -np --user <PhysioNet-ID> --ask-password \
  https://physionet.org/files/mimiciv/3.1/     # 최신 v3.1 (icu/, hosp/)
# 우리 어댑터는 icu 모듈을 사용 → .../mimiciv/3.1/icu 가 있는 경로를 --model 에 지정
```
> 인증 대기 동안 §2b(Challenge 2019)로 파이프라인·수치를 먼저 확보한다.

## 3. 실학습 — 전체 MIMIC-IV (CITI 인증 후, 실제 심정지)
Demo(100명)는 심정지가 거의 없어 학습 불가. **전체 MIMIC-IV**가 필요하다.
```bash
# (a) PhysioNet CITI 인증 완료 후 다운로드 (개인 계정)
#     https://physionet.org/content/mimiciv/  (icu 모듈: chartevents, procedureevents, d_items, icustays)
# (b) icu/ 폴더가 있는 경로를 지정해 원커맨드 실행
python src/mimic_explore.py /workspace/mimic-iv/2.2 --model --tune --gpu --trials=50
```
출력: Optuna 튜닝 → 튜닝 XGBoost vs NEWS 지표 → 알람부담@민감도 → lead-time
→ 그림 5종(models/) → 심정지 표현형. 한 줄로 본선 리포트 재료가 나온다.

- GPU 확인: 학습 중 `nvidia-smi`에 python 프로세스 GPU 점유가 보여야 한다.
- 대용량이면 `--trials`를 늘려도 A6000에서 빠르다(GPU의 실이득 구간).

## 4. 산출물 회수
```bash
ls models/*.png models/*.json     # 그림·지표. git-ignore 대상이라 커밋 안 됨
scp -P 4132 jhkim@222.103.107.7:/workspace/hf-risk-prediction-sw/models/*.png ./   # 로컬로 회수
```

## 체크리스트
- [ ] `nvidia-smi` A6000 ×2 확인
- [ ] `--gpu --tune` 합성 스모크 테스트 통과 (GPU 점유 확인)
- [ ] (CITI 인증) 전체 MIMIC-IV 확보
- [ ] `--model --tune --gpu` 실학습 → 그림·지표 회수
- [ ] 결과 해석 → 제안서/발표 자료 반영

> 안심존(본선)에는 이 코드를 **CPU·사전신고 패키지**(numpy·pandas·scikit-learn·xgboost·shap·
> matplotlib)로 그대로 실행한다. GPU 옵션(`--gpu`)만 빼면 동일 파이프라인이다.
