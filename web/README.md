# 패혈증 조기경보 대시보드 (web/)

향후 6시간 내 패혈증 발병 확률을 시점마다 계산하고, 임계값을 넘으면 경보하며,
그 판단 근거(SHAP)를 임상의가 읽을 형태로 보여주는 Next.js 앱입니다.
KMEDIhub 인턴 트랙이며, K-Health 안심존(폐쇄망)용이 아닙니다.

배포: https://web-ebon-rho-81.vercel.app

## 왜 `src/`가 아니라 `web/`인가

저장소 루트의 `src/`는 파이썬 import 루트입니다(`from vitals_data import ...`).
거기에 Next.js 앱을 넣으면 두 모듈 시스템이 같은 이름 공간을 놓고 충돌합니다.
그래서 앱 루트를 `web/`으로 두고, 그 안에서 통상적인 경로를 씁니다.

## 백엔드가 없습니다

화면이 보여주는 건 고정된 후향 코호트의 분석 결과입니다. 요청할 때마다 달라질 게
없으므로 전부 미리 계산해 `public/data`에 JSON으로 굽습니다.

이게 무게를 없앤 지점입니다. 윈도우를 채점하려면 xgboost(226MB, 대부분 CUDA 커널)가,
근거를 뽑으려면 shap(numba + LLVM 171MB)이 필요합니다. 둘이 합쳐 706MB 중 396MB인데,
**이미 계산된 숫자를 돌려주는 데는 아무것도 필요 없습니다.**

서버 코드는 `/api/explain` 라우트 하나뿐입니다(Groq 호출).

## 실행

```bash
# 1) 스냅샷 굽기 — 저장소 루트에서, 한 번만
python scripts/export_dashboard.py     # 약 3분, 40MB

# 2) 실행
cd web
npm install
npm run dev                            # http://localhost:3000
```

원격 서버/Dev Container라면 VS Code PORTS 탭에서 3000만 포워딩하면 됩니다.

`GROQ_API_KEY`를 환경변수로 주면 자연어 설명 버튼이 켜집니다. 없어도 나머지는
전부 동작하며, 계산된 근거는 그대로 보입니다.

스냅샷을 만들려면 아티팩트가 먼저 필요합니다 — `docs/deploy.md` 참고.

## 배포

Vercel 프로젝트 하나로 끝납니다.

```bash
cd web
npx vercel --prod
```

`.vercelignore`가 있어야 합니다. 없으면 CLI가 `.gitignore`를 따라가서 `public/data`가
통째로 빠지고, 사이트는 뜨는데 모든 데이터가 404 납니다.

## 구조

```
src/
  app/
    layout.tsx                  # data-theme를 첫 페인트 전에 찍는 인라인 스크립트
    page.tsx                    # 좌우 2단 배치. 상태는 갖지 않음
    globals.css                 # 팔레트 전체 (라이트/다크) — 색은 여기서만 정의
    api/explain/route.ts        # 유일한 서버 코드. Groq으로 문장 생성
  components/
    patient-list.tsx            # 병동 목록 + 검색 + 필터(발병/불일치)
    patient-header.tsx          # 인적사항 + 실제 결과(정답 라벨)
    vitals-kpi-cards.tsx        # 활력징후 6종 현재값 + 개인 기저선 대비
    vitals-chart.tsx            # 활력징후 추이 (small multiples)
    sepsis-prediction.tsx       # 위험도 + SHAP 상위 3 + 자연어 설명
    alarm-news.tsx              # 모델 vs NEWS 타임라인 + 알람 부담
    model-scope-warning.tsx     # 활력징후 붕괴 + 낮은 점수일 때의 경고
    RiskBadge.tsx               # 상태 색 + 아이콘 + 라벨
    ui/                         # shadcn/ui 호환 최소 구현 (card, button)
  hooks/
    useDashboard.ts             # 서버 상태 전부. reducer 하나
  lib/
    types.ts                    # 백엔드 응답 모델과 1:1 — 여기가 계약
    api.ts                      # 정적 JSON 로드 + /api/explain 호출
    risk.ts                     # 상태 밴드 / 시리즈 색 / 숫자 포맷
    severity.ts                 # 모델과 무관한 생리학적 위험 신호
```

## 화면이 하지 않는 것

참고한 목업에 있었지만, 우리가 계산하지 않아서 뺀 것들입니다.

- **model confidence** — 모델은 확률 하나를 낼 뿐, 그 확률에 대한 확신도를 계산하지
  않습니다. 화면에서 가장 인용되기 쉬운 거짓말이 됩니다.
- **처치 권고** — 처방을 지시하는 순간 연구 데모가 아니라 의료기기로 읽힙니다.
- **활력징후 예측 곡선** — 모델은 위험도를 예측하지 미래 측정값을 만들지 않습니다.
- **"syncing every 5s"** — 갱신되는 게 없습니다. 후향 코호트가 고정돼 있습니다.

## 차트 규칙

- 이중 축 없음. 모델 확률(0~1)과 NEWS(0~15)는 공통 축이 없어 겹쳐 그리면 없는
  상관을 만들어 냅니다. x축만 공유하는 플롯 2개입니다. 활력징후 6종도 같은 이유로
  small multiples입니다.
- 색은 개체를 따라갑니다. 본 모델은 언제나 슬롯 1, NEWS는 언제나 슬롯 2.
- 상태 색은 아이콘 + 라벨과 항상 함께 나갑니다. 색만으로 의미를 전달하지 않습니다.
- 툴팁은 보조 수단입니다. 모든 값이 화면에 적혀 있고, 알람 부담은 표 보기를 제공합니다.
- 팔레트는 눈으로 고르지 않고 검증기를 돌렸습니다(라이트/다크 모두 통과).

## 한계

- 읽기 전용 스냅샷입니다. 모델·코호트·임계값이 바뀌면 다시 구워야 합니다.
- 실시간이 아닙니다. 새로 들어오는 데이터를 추론하지 않습니다.
- 연구·교육용 데모입니다. 의료기기가 아니며 임상 의사결정에 쓸 수 없습니다.
