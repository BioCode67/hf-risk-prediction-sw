import { NextResponse } from "next/server";

/**
 * Words an already-computed evidence block in Korean, via Groq.
 *
 * This is the only server-side code the dashboard needs. Everything else is a
 * static file, because everything else was decided when the snapshot was baked.
 *
 * The LLM does not judge. The risk score, the alarm decision and the SHAP
 * ranking all arrive in `text`, computed by the Python pipeline; the model's
 * whole job is to turn those numbers into sentences. Constraining it to the
 * supplied figures is what keeps a fluent paragraph from inventing a diagnosis.
 */

export const runtime = "nodejs";

const GROQ_URL = "https://api.groq.com/openai/v1/chat/completions";
const MODEL = process.env.GROQ_MODEL ?? "llama-3.3-70b-versatile";
const MAX_EVIDENCE_CHARS = 4000;

const RATE_LIMIT = 10; // requests
const RATE_WINDOW_MS = 60_000; // per minute, per client

/**
 * Per-client request counters.
 *
 * Honest about what this is: serverless instances are created, duplicated and
 * discarded per region, so this Map is per *instance*, not global. A determined
 * caller spread across instances is not stopped by it. What it does stop is the
 * ordinary case — one script hammering one endpoint — and it costs nothing.
 * A deployment where the bill matters wants a shared store (Vercel KV, Upstash)
 * behind the same check.
 */
const hits = new Map<string, { count: number; resetAt: number }>();

function overRateLimit(client: string, now: number): boolean {
  const entry = hits.get(client);
  if (!entry || now > entry.resetAt) {
    hits.set(client, { count: 1, resetAt: now + RATE_WINDOW_MS });
    // Opportunistic sweep: without it the Map is a slow memory leak on a
    // long-lived instance, since every distinct client leaves an entry behind.
    if (hits.size > 1000) {
      for (const [key, value] of hits) if (now > value.resetAt) hits.delete(key);
    }
    return false;
  }
  entry.count += 1;
  return entry.count > RATE_LIMIT;
}

/**
 * Same-origin only. The dashboard is the sole intended caller, and browsers
 * attach `Origin` to every POST, so a missing or foreign one is not this page.
 * This is not a security boundary — `Origin` is trivially forged outside a
 * browser — it just stops the endpoint being casually reused as a free LLM
 * proxy from another site.
 */
function wrongOrigin(request: Request): boolean {
  const origin = request.headers.get("origin");
  if (!origin) return true;
  try {
    return new URL(origin).host !== request.headers.get("host");
  } catch {
    return true;
  }
}

/** Mirrors SYSTEM_PROMPT in src/vitals_narrate.py — keep the two in step. */
const SYSTEM_PROMPT = `당신은 병동 조기경보 시스템의 설명 생성기입니다. 모델이 계산한 근거를 임상의료진이 읽을 한국어 문장으로 옮기는 것이 유일한 역할입니다.

규칙:
- 제공된 숫자만 사용하십시오. 주어지지 않은 수치를 만들어내지 마십시오.
- 진단명을 붙이지 마십시오 (예: "패혈증 의심" 금지).
- 처치나 처방을 권고하지 마십시오. 확인이 필요한 지점까지만 언급하십시오.
- 기저선 값이 함께 주어진 항목만 "이 환자 평소 대비"로 표현하십시오. 추세(slope)처럼 기저선이 없는 항목에 그 표현을 쓰지 마십시오.
- 경보가 없는 경우에는 왜 울리지 않았는지를 같은 방식으로 설명하십시오.

형식: 3문장 이내, 불릿 없이 이어지는 산문. 활력징후를 전부 나열하지 말고 상위 요인과 그와 직접 관련된 값만 언급하십시오. 위험도·NEWS 숫자를 그대로 되읽지 말고, 무엇이 어떻게 변하고 있는지를 먼저 쓰십시오.`;

/** Lets the UI disable the button rather than offer one that always fails. */
export async function GET() {
  return NextResponse.json({ available: Boolean(process.env.GROQ_API_KEY) });
}

export async function POST(request: Request) {
  if (wrongOrigin(request)) {
    return NextResponse.json({ detail: "이 페이지에서만 호출할 수 있습니다." }, { status: 403 });
  }

  const client = request.headers.get("x-forwarded-for")?.split(",")[0].trim() ?? "unknown";
  if (overRateLimit(client, Date.now())) {
    return NextResponse.json(
      { detail: `요청이 너무 잦습니다. 1분에 ${RATE_LIMIT}회까지만 생성할 수 있습니다.` },
      { status: 429, headers: { "Retry-After": "60" } },
    );
  }

  const apiKey = process.env.GROQ_API_KEY;
  if (!apiKey) {
    return NextResponse.json(
      { detail: "GROQ_API_KEY가 설정되지 않았습니다. 계산된 근거만으로도 설명은 완결됩니다." },
      { status: 503 },
    );
  }

  let text: unknown;
  try {
    ({ text } = await request.json());
  } catch {
    return NextResponse.json({ detail: "요청 본문을 읽을 수 없습니다." }, { status: 400 });
  }
  if (typeof text !== "string" || !text.trim()) {
    return NextResponse.json({ detail: "근거 블록(text)이 비어 있습니다." }, { status: 400 });
  }
  // `text` is whatever the caller sends, and this route is a public URL holding
  // a billable API key. A real evidence block is a few hundred characters; the
  // cap keeps an oversized body from being forwarded at our expense. It bounds
  // the cost of abuse, it does not prevent it — anything beyond a demo needs a
  // rate limit and an origin check here.
  if (text.length > MAX_EVIDENCE_CHARS) {
    return NextResponse.json(
      { detail: `근거 블록이 너무 깁니다 (${MAX_EVIDENCE_CHARS}자 이하).` },
      { status: 413 },
    );
  }

  let response: Response;
  try {
    response = await fetch(GROQ_URL, {
      method: "POST",
      headers: { Authorization: `Bearer ${apiKey}`, "Content-Type": "application/json" },
      body: JSON.stringify({
        model: MODEL,
        temperature: 0.2,
        messages: [
          { role: "system", content: SYSTEM_PROMPT },
          { role: "user", content: text },
        ],
      }),
    });
  } catch {
    return NextResponse.json({ detail: "LLM에 연결할 수 없습니다." }, { status: 502 });
  }

  if (!response.ok) {
    return NextResponse.json(
      { detail: `LLM 오류 ${response.status}` },
      { status: response.status === 429 ? 429 : 502 },
    );
  }

  const body = (await response.json()) as { choices?: { message?: { content?: string } }[] };
  const content = body.choices?.[0]?.message?.content?.trim();
  if (!content) {
    return NextResponse.json({ detail: "LLM이 빈 응답을 반환했습니다." }, { status: 502 });
  }
  return NextResponse.json({ text: content, llm_model: MODEL });
}
