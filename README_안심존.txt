PRODROME(전조) — 안심존 USB 사용법
=====================================
반입 신청 대상: 이 폴더 전체 (run_khth.py, src/*.py 5개, fonts/Pretendard 2종)
필요 패키지: pandas·numpy·scikit-learn·xgboost·shap·matplotlib — 9/2 설치 결과(179개)에 모두 있음. 추가 설치 없음.

[1회차 — 서류 작성 후]
1) 폴더를 분석 PC에 복사
2) python run_khth.py --selftest        ← 합성 데이터로 30초 안에 끝나면 환경 OK
3) python run_khth.py <KHTH 데이터 폴더>  ← PINFO/VITAL 파일 자동 탐색
   - 화면에 열 이름·VS_GBN 값이 찍힘 → HR/SBP/DBP/BT/SPO2/RR 이 아니면 run_khth.py 의 normalize() 매핑 한 줄 수정
   - 결과: results/ 폴더 (집계표 7종 + 그림 6종 + metrics.json)
4) 결과 숫자를 손으로 메모(연습장) — 보고서 초안용

[2~3회차]
- 성능 올리기: python run_khth.py <폴더> --tune --trials=30
- 예측 지평 바꾸기: --horizons=1,2,4,6
- ⚠ 10/28(수)까지 results/ 반출 신청 (심의 1~2주, 11/6 16:00 마감)
  반출 대상 = results/ 폴더만. 00_README.txt 가 반출 심의용 설명(집계값만·n<5 마스킹·식별자 없음).

[방문 증빙 — 매회 필수, 없으면 미방문 처리]
- 사전 신청 메일(10/25 발송분 + 센터 회신) 보관
- 출입 전·후 건물 외부에서 사진 1장씩 (내부·화면 촬영 금지 = 실격)
- 제출: PRODROME(전조)_안심존방문증빙.zip — 방문 회차별 폴더(메일 캡처 + 사진 2장), 11/6 16:00까지
