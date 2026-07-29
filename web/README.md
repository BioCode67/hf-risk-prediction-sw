# 조기경보 대시보드 (web/)

`src/vitals_api.py`(FastAPI)가 내주는 위험도·경보 판단·SHAP 근거를 임상의가 보는
화면으로 옮긴 Next.js 앱입니다. KMEDIhub 인턴 트랙용이며, K-Health 안심존(폐쇄망)
에서 돌리는 것이 아닙니다.

## 왜 `src/`가 아니라 `web/`인가

저장소 루트의 `src/`는 파이썬 import 루트입니다(`from vitals_data import ...`).
거기에 Next.js 앱을 넣으면 두 모듈 시스템이 같은 이름 공간을 놓고 충돌합니다.
그래서 앱 루트를 `web/`으로 두고, 그 안에서 통상적인 경로를 씁니다 —
`web/src/app/page.tsx`, `web/src/components/Dashboard.tsx`.

## 실행

터미널 두 개가 필요합니다.

```bash
# 1) 백엔드 — 합성 코호트 (데이터 없이 바로 뜸)
uvicorn vitals_api:app --app-dir src --reload

# 실데이터로 띄우려면
EWS_DATA_DIR=data/challenge2019/training_setA EWS_MAX_FILES=2000 \
  uvicorn vitals_api:app --app-dir src --reload
```

```bash
# 2) 프론트엔드
cd web
npm install
npm run dev                          # http://localhost:3000
```

브라우저가 여는 포트는 **3000 하나뿐**입니다. `/api/*`는 Next가 서버 사이드에서
FastAPI로 프록시하므로(`next.config.mjs`의 rewrites) 8000번을 따로 포워딩할 필요가
없고, 크로스 오리진 요청이 아예 일어나지 않아 CORS도 없습니다. `/docs`(FastAPI
문서)도 같은 포트로 열립니다.

원격 서버/Dev Container라면 VS Code의 PORTS 탭에서 3000만 포워딩하면 됩니다.

첫 요청에서 코호트를 만들고 모델을 학습하므로 실데이터는 1~2분 걸립니다.
`EWS_MAX_FILES`로 줄이세요. 학습은 train 환자로만 하고 화면에는 test 환자만
노출합니다 — 자기가 학습한 환자의 점수를 보여주면 안 되기 때문입니다.

백엔드 환경변수: `EWS_DATA_DIR`, `EWS_MAX_FILES`(기본 2000), `EWS_HORIZON`(기본 6),
`GROQ_API_KEY`(자연어 설명용). `EWS_CORS_ORIGINS`는 위 프록시를 쓰는 한 필요 없고,
브라우저가 FastAPI를 직접 호출하도록 바꿀 때만 씁니다(`*`로 전부 허용 가능).

## 구조

```
src/
  app/
    layout.tsx        # data-theme를 첫 페인트 전에 찍는 인라인 스크립트
    page.tsx          # Dashboard 하나만 렌더
    globals.css       # 팔레트 전체 (라이트/다크) — 색은 여기서만 정의
  components/
    Dashboard.tsx     # 레이아웃. 상태는 갖지 않고 배치만
    PatientList.tsx   # 위험도순 목록 (표 — 세로로 숫자를 비교하는 일이므로)
    RiskTimeline.tsx  # 모델 위험도 / NEWS, 축이 달라 별도 플롯 2개
    VitalsGrid.tsx    # 활력징후 6종 small multiples + 개인 기저선 가로선
    EvidenceCard.tsx  # 경보 판단 + SHAP 상위 3 + 계산값 + (선택) LLM 문장
    AlarmBurden.tsx   # 같은 검출률에서의 알람 수 비교 + 표 보기
    RiskBadge.tsx     # 상태 색 + 아이콘 + 라벨 (색만으로 의미를 전달하지 않음)
    ui/               # shadcn/ui 호환 최소 구현 (card, button)
  hooks/
    useDashboard.ts   # 서버 상태 전부. reducer 하나
  lib/
    types.ts          # FastAPI 응답 모델과 1:1 — 여기가 계약
    api.ts            # HTTP만 담당
    risk.ts           # 상태 밴드 / 시리즈 색 / 숫자 포맷
```

## v0 코드를 얹을 때

`components/ui/`는 shadcn/ui와 같은 이름·props로 맞춰 뒀습니다. v0가 뽑아준
컴포넌트를 그대로 붙여도 `@/components/ui/card`, `@/lib/utils`의 `cn` 임포트가
해결됩니다. 정식 shadcn을 쓰려면 `npx shadcn@latest add card button`으로 이
파일들을 덮어쓰면 됩니다.

색만 주의하세요. v0 기본 테마는 자체 CSS 변수를 쓰는데, 이 앱은 `globals.css`의
팔레트를 단일 출처로 씁니다. 차트 색은 `--series-1`(본 모델) / `--series-2`(NEWS)
두 슬롯에 고정돼 있고, 상태 색 4종은 시리즈 색으로 재사용하지 않습니다.

## 차트에서 지킨 규칙

- 이중 축 없음. 모델 확률(0~1)과 NEWS(0~15)는 공통 축이 없어 겹쳐 그리면 없는
  상관을 만들어 냅니다. 그래서 x축만 공유하는 플롯 2개입니다.
- 활력징후 6종도 같은 이유로 small multiples입니다.
- 색은 개체를 따라갑니다. 본 모델은 언제나 슬롯 1, NEWS는 언제나 슬롯 2.
- 툴팁은 보조 수단입니다. 모든 값은 툴팁 없이도 화면에 적혀 있고, 알람 부담은
  표 보기를 함께 제공합니다.
- 팔레트는 눈으로 고르지 않고 검증기를 돌렸습니다(라이트/다크 모두 통과).

## 한계

- 읽기 전용입니다. 쓰기 경로도, 인증도, 데이터베이스도 없습니다.
- 실시간이 아닙니다. 백엔드는 기동 시 만든 코호트를 메모리에 들고 있습니다.
- 연구·교육용 데모입니다. 의료기기가 아니며 임상 의사결정에 쓸 수 없습니다.
