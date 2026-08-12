# 주간 보고서용 스크린샷

주차별 2장씩, 보고서 본문에 붙일 이미지입니다. 전부
`scripts/make_report_screenshots.py`가 저장소에서 직접 만들어냅니다 — 코드는 실제
소스 줄을 하이라이트한 것이고, 콘솔 화면은 그 자리에서 스크립트를 실행한 진짜
stdout이며, 그래프는 `vitals_report` / `vitals_explain`의 산출물입니다.

```bash
pip install playwright pygments        # requirements.txt에는 없는 개발용 의존성

python scripts/make_report_screenshots.py            # 코드·콘솔·그래프 8장
python scripts/make_report_screenshots.py --dashboard # 대시보드까지 (아래 주의)
```

## 목록

| 파일 | 주차 | 내용 |
|---|---|---|
| `week1_1_preprocessing_code.png` | 1주차 | 생체신호 이상치 정제 — 센서 단선(맥박 0), 화씨 오기록, 생리학적 범위 밖 값을 NaN 처리 |
| `week1_2_data_exploration.png` | 1주차 | 코호트 규모·변수별 결측률·슬라이딩 윈도우 생성 결과·클래스 불균형(1:81)·환자 단위 분할 누출 검증 |
| `week2_1_rnn_model_code.png` | 2주차 | LSTM/GRU 시퀀스 분류기 — 가변 길이 패딩 마스킹, `rnn_type`으로 두 구조 교체 |
| `week2_2_xgboost_code.png` | 2주차 | 비용민감 XGBoost 베이스라인 — `scale_pos_weight`, `eval_metric="aucpr"`, GPU 폴백 |
| `week3_1_evaluation_output.png` | 3주차 | XGBoost vs NEWS: AUPRC·ROC-AUC·특이도 95%에서의 민감도·혼동행렬, 동일 민감도 알람 부담, 리드타임 |
| `week3_2_pr_curve.png` | 3주차 | Precision–Recall 곡선 (AUPRC 0.84 vs NEWS 0.63, chance 0.012) |
| `week4_1_shap_summary.png` | 4주차 | SHAP 요약 플롯 — 어떤 활력징후 추세가 위험도를 올리는지 |
| `week4_2_dashboard.png` | 4주차 | 대시보드 — 환자 타임라인, 개인 기저선 대비 편차, SHAP 근거, 경보 임계값 조절 |

참고용 여분 2장: `extra_shap_console.png`(SHAP 콘솔 출력),
`extra_alarm_burden.png`(동일 민감도 알람 부담 막대그래프).

## 주의 — 이 숫자들은 합성 코호트 결과입니다

이미지 전부 저장소 내장 **합성 코호트**에서 나온 값입니다. 실제 환자 데이터가
아니므로 성능 근거로 인용하면 안 되고, 파이프라인이 무엇을 하는지 보여주는
예시로만 쓰십시오. 보고서 본문에 실측 성능(예: Transformer 0.94)을 함께 적는다면
이 그림들의 수치와 다른 출처라는 점을 캡션에 밝혀 두는 편이 안전합니다.

`data/`에 실제 데이터가 있으면 같은 스크립트가 그대로 돌아가고 숫자만 바뀝니다.

## 대시보드 스크린샷을 다시 찍으려면

정적 JSON을 먼저 굽고 Next.js를 띄운 다음 `--only-dashboard`로 찍습니다.

```bash
EWS_ARTIFACT=models/dashboard_synth.pkl \
  python -c "import sys; sys.path.insert(0,'src'); import vitals_api; vitals_api.service()"
python scripts/export_dashboard.py --artifact models/dashboard_synth.pkl --out web/public/data
cd web && npm run dev &                       # :3000
python scripts/make_report_screenshots.py --only-dashboard
```
