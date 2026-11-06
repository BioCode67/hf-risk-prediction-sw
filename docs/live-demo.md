# 실시간 병상 시연 (`/live`)

본선 발표(11/13)에서 실제 측정기기를 연결해 보여 주기 위한 화면입니다. 측정값이 들어오면
분석과 같은 파이프라인(슬라이딩 윈도우 → 개인 기저선 편차 → XGBoost)으로 위험도를 계산하고,
NEWS와 나란히 비교하며, 왜 그 점수가 나왔는지 기여 요인을 보여 줍니다.

## 무엇이 어디서 돌아가나

| 구성 | 파일 | 비고 |
|---|---|---|
| 화면 | `web/src/app/live/page.tsx` | 대시보드 상단 "실시간 병상 시연" 버튼 |
| 특징 계산 | `web/src/lib/live-features.ts` | `src/vitals_live.py`와 **소수 4째 자리까지 일치** (검사: 아래) |
| 모델 실행 | `web/src/lib/live-model.ts` | `public/live/model.json`을 브라우저에서 직접 실행. 서버·인터넷 불필요 |
| 기기 연결 | `web/src/lib/ble.ts` | Web Bluetooth (Chrome·Edge) |
| 시나리오 | `web/src/lib/live-scenarios.ts` | 기기 없이도 같은 흐름 재현 |
| 서버판(선택) | `POST /api/live/score` (`src/vitals_api.py`) | 같은 계산을 FastAPI로. 테스트 `tests/test_live.py` |

모델은 **합성 데이터로 학습한 데모 모델**입니다(`scripts/export_live_model.py`). 경북대 데이터로 학습한
모델은 안심존 밖으로 나올 수 없으므로 시연에 쓰지 않습니다. 화면에도 그렇게 적혀 있습니다.

시연을 위해 **10초 = 1단계**로 시간을 압축했습니다. 처음 6단계(60초)는 개인 기저선을 재는 구간이라
그동안은 점수를 내지 않습니다.

## 검사

```bash
python scripts/export_live_model.py                              # 모델 다시 만들 때만
node --experimental-strip-types web/scripts/check_live_parity.mjs # 파이썬 == 브라우저
node --experimental-strip-types web/scripts/check_ble_parsers.mjs # 기기 패킷 해석
pytest -q tests/test_live.py
```

## 실행

```bash
cd web && npm install && npm run build && npm start   # http://localhost:3000/live
```

Web Bluetooth는 **https 또는 localhost**에서만 동작합니다. 배포 주소(https)나 노트북의 localhost로
여세요. 휴대폰 브라우저·Safari는 지원하지 않습니다.

## 지원 기기

| 종류 | 방식 | 예시 |
|---|---|---|
| 심박 | 표준 Heart Rate 0x180D | Polar H10, COOSPO H808S 등 가슴 스트랩 |
| 산소포화도 | BerryMed 공개 프로토콜 | BM1000C |
| 산소포화도 | Viatom 공개 디코더 기반(실험적) | PC-60FW |
| 산소포화도 | 표준 Pulse Oximeter 0x1822 | 지원 기기 드묾 |
| 혈압 | 표준 Blood Pressure 0x1810 | A&D UA-651BLE 등. 측정 끝날 때 1회 전송 |
| 체온 | 표준 Health Thermometer 0x1809 | A&D UT-201BLE 등. 측정 끝날 때 1회 전송 |

호흡수는 소비자 기기로 재기 어렵습니다. "간호 기록 입력"에 직접 넣습니다(실제 병동도 간호사가 셉니다).
혈압계·체온계는 **OS 블루투스 설정에서 먼저 페어링**해 두고, 연결한 뒤 측정 버튼을 누르세요.

## 시연 대본 (약 4분)

1. **안정 환자** 시나리오 → 60초 기저선 측정 → 모델·NEWS 모두 조용함.
2. **평소 수치가 나쁜 만성 환자** → NEWS 8점 경보, 모델은 0%대.
   "맥박 112, SpO2 92%가 이 분의 평소 값입니다. 기존 점수는 매번 울리지만, 개인 기저선을 보는 모델은
   울리지 않습니다. 저희가 줄이려는 오경보가 이것입니다."
3. **서서히 악화** → 80초부터 악화, 경보가 뜨면 "왜 이 점수인가"에서 SpO2 추세·기저선 이탈을 짚기.
4. **실제 기기**: 심박 스트랩(+산소포화도계)을 착용하고 "처음부터" → 60초 동안 가만히 앉아 기저선
   측정 → 제자리 스쿼트 20회나 숨 참기 → 맥박이 평소 대비 오르는 것이 실시간으로 반영됨.
   (데모 모델이라 경보까지 가지 않을 수 있습니다. "평소 대비 편차"가 근거 목록에 올라오는 것을 보여 주면 충분합니다.)

## 리허설 체크

- 발표 노트북에서 `npm start`로 로컬 실행해 `/live`가 뜨는지 확인(배포 주소는 인터넷이 필요합니다).
- 기기 배터리, 블루투스 켜기, 혈압계·체온계 OS 페어링.
- 발표장 와이파이가 불안하면 로컬 실행으로 시연(모델이 브라우저에서 돌아 인터넷 불필요).
