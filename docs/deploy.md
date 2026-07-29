# 배포 — 대시보드 (KMEDIhub 인턴 트랙)

FastAPI(`src/vitals_api.py`) + Next.js(`web/`) 두 조각을 올립니다. K-Health
안심존과는 무관합니다(폐쇄망은 배포 자체가 불가).

## 0. 배포 전에 반드시

- **Groq 키를 새로 발급하세요.** 지금 `.env`에 있는 키는 대화 로그에 평문으로
  남았습니다. 배포하면 인터넷에 노출된 서비스가 그 키를 씁니다. Groq 콘솔에서
  기존 키를 폐기하고 새 키를 Render 환경변수에만 넣으세요. 저장소에는 넣지 않습니다.
- 공개 URL이 생기므로 화면 하단 면책 문구(연구·교육용, 의료기기 아님, 처치 권고
  없음, 데이터는 패혈증이지 심정지가 아님)를 지우지 마세요.
- 데이터는 PhysioNet/CinC Challenge 2019 공개셋입니다. 인증이 필요 없는 자료라
  공개 배포에 문제가 없습니다. **MIMIC-IV나 경북대 데이터로 만든 아티팩트는
  절대 배포하지 마세요** — DUA 위반입니다.

## 1. 아티팩트를 먼저 만든다

원본 20,336명을 읽고 학습하는 데 **약 5분 / 최대 4.1GB**가 듭니다. 배포 인스턴스가
이걸 할 수는 없습니다. 로컬에서 한 번 굽고 결과만 올립니다.

```bash
EWS_DATA_DIR=data/challenge2019/training_setA \
EWS_MAX_FILES=all \
EWS_HORIZON=6 \
EWS_MAX_PATIENTS=400 \
EWS_ARTIFACT=models/dashboard_challenge2019.pkl \
python -c "import sys; sys.path.insert(0,'src'); import vitals_api; vitals_api.service()"
```

산출물은 13MB입니다. 코호트 지표(알람 부담 등)는 **잘라내기 전 전체 test set
3,972명**에서 측정하고, 열람 가능한 400명만 행을 남깁니다. 즉 화면에서 못 여는
환자가 있어도 숫자는 전수 기준입니다.

`models/*.pkl`은 `.gitignore` 대상이라 강제로 추가해야 합니다.

```bash
git add -f models/dashboard_challenge2019.pkl
```

> pickle은 역직렬화 시 코드를 실행할 수 있습니다. 이 파일은 우리가 만든 것만
> 쓰고, 외부에서 받은 pkl을 `EWS_ARTIFACT`로 지정하지 마세요.

## 2. 백엔드 — Render

`render.yaml`이 저장소 루트에 있습니다. Render 대시보드에서 New → Blueprint로
저장소를 연결하면 그대로 읽습니다.

측정값 (이 아티팩트 기준):

| 항목 | 값 |
|---|---|
| 아티팩트 로드 | 0.1초 |
| import 직후 RSS | 186 MB |
| 로드 후 RSS | 305 MB |
| 첫 상세 요청 | 929ms (SHAP explainer 생성) |
| 이후 상세 요청 | 20~30ms |
| **정상 상태 RSS** | **461 MB** |

**메모리가 문제입니다.** Render 무료와 Starter는 둘 다 512MB라 461MB면 여유가
50MB뿐입니다. 단일 사용자 데모는 돌아가지만 동시 요청이 겹치면 OOM으로 죽습니다.
선택지는 셋입니다.

1. **Standard(2GB)로 올린다** — 가장 확실. 유료.
2. **무료로 감수한다** — 발표 중 죽을 수 있음. 무료 티어는 15분 무접속 시
   슬립되고, 깨어날 때 콜드스타트가 붙습니다.
3. **SHAP을 런타임에서 걷어낸다** — 아티팩트를 구울 때 윈도우별 근거(top-3)를
   미리 계산해 넣으면, 서비스는 xgboost·shap 없이 데이터만 내주면 됩니다.
   RSS가 150MB 수준으로 떨어져 무료 티어에 넉넉히 들어갑니다. 아직 구현 안 됨.

3번이 제대로 된 답입니다. 발표가 급하면 1번.

## 3. 프론트엔드 — Vercel

Next.js라 Vercel이 자연스럽습니다.

- Import 시 **Root Directory를 `web`으로** 지정 (저장소 루트가 아닙니다).
- 환경변수 `EWS_API_ORIGIN` = Render 서비스 URL (예: `https://ews-api.onrender.com`).
  `NEXT_PUBLIC_`이 아닙니다 — 프록시는 서버 사이드에서 돕니다.
- 그 외 설정 없음. 빌드는 `next build` 기본값.

브라우저는 Vercel 도메인만 호출하고, Vercel의 Next 서버가 Render로 프록시합니다.
그래서 CORS 설정이 필요 없습니다.

## 4. 확인

```bash
curl https://<vercel-도메인>/api/health      # {"status":"ok"}
curl https://<vercel-도메인>/api/overview    # 코호트 지표
```

`/docs`로 FastAPI 문서도 같은 도메인에서 열립니다.

## 5. 알아둘 한계

- 실시간이 아닙니다. 후향 코호트를 고정해 들고 있습니다. 화면의 "syncing" 류
  표현을 다시 넣지 마세요 — 갱신되는 게 없습니다.
- 쓰기 경로도 인증도 없습니다. 읽기 전용 데모입니다.
- Render 무료는 슬립합니다. 발표 직전에 한 번 깨워두세요.
