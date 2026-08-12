# 배포 — 대시보드 (KMEDIhub 인턴 트랙)

**Vercel 프로젝트 하나로 끝납니다.** 파이썬도, ML 라이브러리도, 백엔드 서비스도
배포에 필요하지 않습니다. K-Health 안심존과는 무관합니다(폐쇄망은 배포 자체가 불가).

> 이 문서는 한때 Render(FastAPI) + Vercel(Next) 두 조각을 올리는 절차였습니다.
> 코호트 분석을 통째로 정적 JSON으로 굽게 되면서 백엔드가 사라졌습니다.
> 아래 §5에 옛 구조가 왜 없어졌는지 남겨 둡니다.

## 0. 배포 전에 반드시

- **Groq 키를 새로 발급하세요.** 예전 `.env`에 있던 키는 대화 로그에 평문으로
  남았습니다. 배포하면 인터넷에 노출된 서비스가 그 키를 씁니다. Groq 콘솔에서
  기존 키를 폐기하고 새 키를 **Vercel 환경변수에만** 넣으세요. 저장소에는 넣지 않습니다.
- 공개 URL이 생기므로 화면 하단 면책 문구(연구·교육용, 의료기기 아님, 처치 권고
  없음, 데이터는 패혈증이지 심정지가 아님)를 지우지 마세요.
- 데이터는 PhysioNet/CinC Challenge 2019 공개셋입니다. 인증이 필요 없는 자료라
  공개 배포에 문제가 없습니다. **MIMIC-IV나 경북대 데이터로 만든 아티팩트는
  절대 배포하지 마세요** — DUA 위반입니다.

## 1. 아티팩트를 만든다 (로컬에서 한 번)

원본 20,336명을 읽고 학습하는 데 **약 5분 / 최대 4.1GB**가 듭니다. 로컬에서 한 번
굽고 결과만 올립니다.

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

> pickle은 역직렬화 시 코드를 실행할 수 있습니다. 이 파일은 우리가 만든 것만
> 쓰고, 외부에서 받은 pkl을 `EWS_ARTIFACT`로 지정하지 마세요.

## 2. 정적 스냅샷으로 굽는다

```bash
python scripts/export_dashboard.py \
    --artifact models/dashboard_challenge2019.pkl \
    --out web/public/data
```

약 3분, 40MB. `index.json`(코호트 지표 + 환자 목록)과 환자당 하나씩
`patients/<id>.json`이 나옵니다. 모양은 `web/src/lib/types.ts`와 1:1입니다.

**모델·코호트·임계값이 바뀌면 다시 구워야 합니다.** 스냅샷은 스냅샷입니다.

## 3. Vercel에 올린다

```bash
cd web
npx vercel --prod
```

- Import 시 **Root Directory를 `web`으로** 지정 (저장소 루트가 아닙니다).
- 빌드는 `next build` 기본값. 그 외 설정 없음.
- **`.vercelignore`가 있어야 합니다.** 없으면 CLI가 `.gitignore`를 따라가서
  `web/public/data`가 통째로 빠지고, 사이트는 뜨는데 모든 데이터가 404 납니다.

### 환경변수 (둘 다 선택)

| 변수 | 없으면 | 비고 |
|---|---|---|
| `GROQ_API_KEY` | 자연어 설명 버튼만 비활성화 | 나머지는 전부 동작 |
| `GROQ_MODEL` | `route.ts`의 기본값 사용 | **언젠가 반드시 바꿔야 합니다** ↓ |

`GROQ_MODEL`은 이 배포에서 가장 먼저 썩는 값입니다. 호스팅 모델 이름은 제공자
일정에 따라 사라지고, 실제로 `llama-3.3-70b-versatile`은 2026-06-17부로 무료·개발자
티어에서 폐기됐습니다. "생성" 버튼이 400/404로 실패하기 시작하면 **코드가 아니라 이
환경변수를 바꾸세요.** 화면의 "자연어 설명" 항목에 지금 부를 모델 이름이 적혀 있고,
실패 메시지에도 그 이름이 들어갑니다. 쓸 수 있는 이름은
<https://console.groq.com/docs/models>.

바꿀 때 `src/vitals_narrate.py`의 `DEFAULT_MODEL`도 같이 맞추세요.

## 4. 확인

```bash
curl https://<vercel-도메인>/data/index.json | head -c 200   # 코호트 스냅샷
curl https://<vercel-도메인>/api/explain                      # {"available":…,"model":…}
```

`/api/explain`이 `{"available":false}`면 `GROQ_API_KEY`가 안 붙은 것입니다.
`available:true`인데 생성이 실패하면 응답의 모델 이름을 보세요 — 위의 폐기 문제입니다.

브라우저에서 화면을 열고 환자 하나를 클릭하는 것까지 해보세요. `index.json`은 떠도
`patients/*.json`이 빠지는 경우가 `.vercelignore` 사고의 전형적인 증상입니다.

## 5. 옛 구조 (Render + FastAPI) — 왜 없앴나

예전에는 `src/vitals_api.py`를 Render에 올리고 Next가 서버 사이드로 프록시했습니다.
측정값이 문제였습니다.

| 항목 | 값 |
|---|---|
| import 직후 RSS | 186 MB |
| 첫 상세 요청 | 929ms (SHAP explainer 생성) |
| **정상 상태 RSS** | **461 MB** |

Render 무료·Starter가 512MB라 여유가 50MB뿐이었고, 동시 요청이 겹치면 OOM으로
죽었습니다. 원인은 명확했습니다 — `libxgboost.so` 226MB(대부분 CUDA 커널)와
`libllvmlite.so` 171MB(shap→numba)가 706MB 중 396MB인데, **이미 계산된 숫자를
돌려주는 데는 그 둘이 필요 없습니다.**

그래서 근거를 전부 사전계산해 정적 파일로 내보냈습니다(§2). 서버가 사라지니
메모리 문제도, 콜드스타트도, CORS도 같이 사라졌습니다.

`render.yaml`은 저장소에 남아 있습니다. 대시보드 배포에는 쓰이지 않지만,
FastAPI를 라이브로 띄우고 싶을 때(예: `/api/patients`를 다른 클라이언트에서
호출) 그대로 쓸 수 있습니다. 그 경우에도 `web/`은 여전히 자기 정적 스냅샷을
읽습니다 — 둘은 이제 연결돼 있지 않습니다.

## 6. 알아둘 한계

- 실시간이 아닙니다. 후향 코호트를 고정해 들고 있습니다. 화면의 "syncing" 류
  표현을 다시 넣지 마세요 — 갱신되는 게 없습니다.
- 쓰기 경로도 인증도 없습니다. 읽기 전용 데모입니다.
- `/api/explain`은 같은 페이지에서만(Origin 확인), 1분에 10회까지만 호출됩니다.
  둘 다 보안 경계가 아니라 비용 상한입니다 — 실사용이라면 공유 저장소를 쓴
  rate limit이 필요합니다.
