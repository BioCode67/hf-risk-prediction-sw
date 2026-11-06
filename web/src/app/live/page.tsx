"use client";

import { Activity, ArrowLeft, Bluetooth, Pause, Play, RotateCcw, Unplug } from "lucide-react";
import Link from "next/link";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { bluetoothAvailable, connectDevice, DEVICES, type Connection, type DeviceKind } from "@/lib/ble";
import { binReadings, liveFeatures, liveNews, type LiveReading, type Vital, VITALS } from "@/lib/live-features";
import { contributions, type LiveModel, predictRisk } from "@/lib/live-model";
import { SCENARIOS, type ScenarioKind, scenarioReading } from "@/lib/live-scenarios";

/**
 * Live bedside demo: real devices (or a built-in scenario) → the same features,
 * personal baseline and model as the analysis → risk, NEWS and why.
 *
 * Everything runs in the browser from `public/live/model.json`; nothing is sent
 * anywhere and nothing is stored. Time is compressed — one model step is
 * `STEP_SECONDS` of wall clock — and the screen says so.
 */

const STEP_SECONDS = 10;
const KEEP_SECONDS = 15 * 60;
const NEWS_THRESHOLD = 5;

const VITAL_META: Record<Vital, { label: string; unit: string; digits: number }> = {
  pulse: { label: "맥박", unit: "회/분", digits: 0 },
  spo2: { label: "SpO2", unit: "%", digits: 0 },
  sbp: { label: "수축기혈압", unit: "mmHg", digits: 0 },
  dbp: { label: "이완기혈압", unit: "mmHg", digits: 0 },
  resp_rate: { label: "호흡수", unit: "회/분", digits: 0 },
  temperature: { label: "체온", unit: "°C", digits: 1 },
};

const STAT_KO: Record<string, string> = {
  mean: "최근 8단계 평균",
  std: "최근 8단계 흔들림",
  min: "최근 8단계 최저",
  max: "최근 8단계 최고",
  last: "현재값",
  slope: "단계당 변화 추세",
  delta: "최근 8단계 변화량",
  last_dev: "현재값의 개인 기저선 대비 편차",
  mean_dev: "평균의 개인 기저선 대비 편차",
};

function describe(feature: string): string {
  if (feature === "shock_index_last") return "쇼크 지수(맥박/수축기혈압)";
  if (feature === "pulse_pressure_last") return "맥압(수축기-이완기)";
  const vital = VITALS.find((v) => feature.startsWith(`${v}_`));
  if (!vital) return feature;
  const stat = feature.slice(vital.length + 1);
  return `${VITAL_META[vital].label} ${STAT_KO[stat] ?? stat}`;
}

type Level = "good" | "warning" | "serious" | "critical";
function levelOf(risk: number, threshold: number): Level {
  if (risk >= Math.max(threshold, 0.5)) return "critical";
  if (risk >= threshold) return "serious";
  if (risk >= threshold / 2) return "warning";
  return "good";
}
const LEVEL_KO: Record<Level, string> = { good: "평소 수준", warning: "주의", serious: "경계", critical: "위험" };
const LEVEL_CLASS: Record<Level, string> = {
  good: "bg-success",
  warning: "bg-warning",
  serious: "bg-serious",
  critical: "bg-critical",
};

interface Point {
  t: number;
  risk: number | null;
  news: number | null;
}

export default function LivePage() {
  const [model, setModel] = useState<LiveModel | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [readings, setReadings] = useState<LiveReading[]>([]);
  const [history, setHistory] = useState<Point[]>([]);
  const [scenario, setScenario] = useState<ScenarioKind>("worsening");
  const [running, setRunning] = useState(false);
  const [connections, setConnections] = useState<Partial<Record<DeviceKind, Connection>>>({});
  const [lastSeen, setLastSeen] = useState<Partial<Record<DeviceKind, number>>>({});
  const [deviceError, setDeviceError] = useState<string | null>(null);
  const [manual, setManual] = useState<Partial<Record<Vital, string>>>({});
  const [clock, setClock] = useState(0);
  const startRef = useRef<number>(Date.now());

  useEffect(() => {
    fetch("/live/model.json")
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then(setModel)
      .catch((e: Error) => setLoadError(`모델을 불러오지 못했습니다: ${e.message}`));
  }, []);

  const now = useCallback(() => (Date.now() - startRef.current) / 1000, []);

  const push = useCallback(
    (values: Partial<Record<Vital, number>>) => {
      const t = now();
      setReadings((prev) => [...prev.filter((r) => r.t > t - KEEP_SECONDS), { t, values }]);
    },
    [now],
  );

  // Scenario feed: one reading per second while running.
  useEffect(() => {
    if (!running) return;
    const id = setInterval(() => push(scenarioReading(scenario, now())), 1000);
    return () => clearInterval(id);
  }, [running, scenario, push, now]);

  // Clock tick drives rescoring once a second, whatever the source.
  useEffect(() => {
    const id = setInterval(() => setClock(now()), 1000);
    return () => clearInterval(id);
  }, [now]);

  const result = useMemo(() => {
    if (!model || !readings.length) return null;
    const binned = binReadings(readings, STEP_SECONDS);
    const features = liveFeatures(binned, model.features, model.window_steps, model.baseline_steps);
    const news = liveNews(features.named);
    if (!features.baselineReady) return { features, news, risk: null, factors: [] };
    const risk = predictRisk(model, features.row);
    const factors = contributions(model, features.row).slice(0, 4);
    return { features, news, risk, factors };
    // `clock` is in the deps on purpose: rescoring follows wall time, so a
    // device that goes quiet shows up as missing steps rather than freezing.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model, readings, clock]);

  useEffect(() => {
    if (!result) return;
    setHistory((prev) => [
      ...prev.slice(-300),
      { t: Math.round(clock), risk: result.risk, news: result.news.measured ? result.news.score : null },
    ]);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [clock]);

  const reset = () => {
    setRunning(false);
    setReadings([]);
    setHistory([]);
    startRef.current = Date.now();
    setClock(0);
  };

  const connect = async (kind: DeviceKind) => {
    setDeviceError(null);
    setRunning(false);
    try {
      const connection = await connectDevice(
        kind,
        (values) => {
          push(values);
          setLastSeen((prev) => ({ ...prev, [kind]: Date.now() }));
        },
        () => setConnections((prev) => ({ ...prev, [kind]: undefined })),
      );
      setConnections((prev) => ({ ...prev, [kind]: connection }));
    } catch (error) {
      const message = (error as Error).message ?? String(error);
      if (!/cancel/i.test(message)) setDeviceError(message);
    }
  };

  const submitManual = () => {
    const values: Partial<Record<Vital, number>> = {};
    for (const [vital, raw] of Object.entries(manual)) {
      const number = Number(raw);
      if (raw && Number.isFinite(number)) values[vital as Vital] = number;
    }
    if (Object.keys(values).length) push(values);
  };

  const threshold = model?.threshold ?? 0.5;
  const risk = result?.risk ?? null;
  const level = risk === null ? null : levelOf(risk, threshold);
  const modelAlarm = risk !== null && risk >= threshold;
  const newsAlarm = !!result && result.news.measured > 0 && result.news.score >= NEWS_THRESHOLD;
  const latest = result?.features.named ?? {};
  const baseline = result?.features.baseline ?? {};

  return (
    <div className="bg-background min-h-svh">
      <div className="mx-auto flex max-w-[1500px] flex-col gap-4 p-4 md:p-6">
        <header className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-2.5">
            <div className="bg-primary text-primary-foreground flex size-9 items-center justify-center rounded-xl shadow-sm">
              <Activity className="size-5" aria-hidden />
            </div>
            <div className="leading-tight">
              <h1 className="text-sm font-semibold tracking-tight">실시간 병상 시연</h1>
              <p className="text-subtle-foreground text-xs">측정기기 → 개인 기저선 → 위험도 · NEWS · 근거</p>
            </div>
          </div>
          <Link href="/" className="text-muted-foreground hover:text-foreground inline-flex items-center gap-1 text-xs">
            <ArrowLeft className="size-3.5" aria-hidden /> 후향 코호트 대시보드
          </Link>
        </header>

        <p className="border-border bg-card text-muted-foreground rounded-lg border px-3 py-2 text-xs leading-relaxed">
          시연용 화면입니다. 모델은 <strong>합성 데이터로 학습한 데모 모델</strong>(분석과 같은 파이프라인: 슬라이딩 윈도우 +
          개인 기저선 편차 + 비용민감 XGBoost)이며, 시연을 위해 <strong>{STEP_SECONDS}초 = 1단계</strong>로 시간을
          압축했습니다. 처음 {model?.baseline_steps ?? 6}단계({(model?.baseline_steps ?? 6) * STEP_SECONDS}초)는 이 환자의 개인
          기저선을 재는 구간입니다. 측정값은 브라우저 밖으로 전송·저장되지 않습니다. 의료기기가 아닙니다.
        </p>
        {loadError && <p className="text-critical text-sm">{loadError}</p>}

        <div className="grid grid-cols-1 gap-4 lg:grid-cols-[340px_minmax(0,1fr)]">
          {/* Sources */}
          <div className="flex flex-col gap-4">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle>시나리오</CardTitle>
                <CardDescription>기기 없이도 같은 흐름을 재현합니다</CardDescription>
              </CardHeader>
              <CardContent className="flex flex-col gap-2">
                {SCENARIOS.map((s) => (
                  <label
                    key={s.kind}
                    className={`border-border flex cursor-pointer flex-col rounded-lg border px-3 py-2 text-xs ${scenario === s.kind ? "ring-primary ring-2" : ""}`}
                  >
                    <span className="flex items-center gap-2 font-medium">
                      <input
                        type="radio"
                        name="scenario"
                        checked={scenario === s.kind}
                        onChange={() => {
                          reset();
                          setScenario(s.kind);
                        }}
                      />
                      {s.label}
                    </span>
                    <span className="text-muted-foreground mt-0.5 pl-5">{s.story}</span>
                  </label>
                ))}
                <div className="flex gap-2 pt-1">
                  <Button size="sm" onClick={() => setRunning((r) => !r)} disabled={!model}>
                    {running ? <Pause className="mr-1 size-3.5" /> : <Play className="mr-1 size-3.5" />}
                    {running ? "일시정지" : "시작"}
                  </Button>
                  <Button size="sm" variant="outline" onClick={reset}>
                    <RotateCcw className="mr-1 size-3.5" /> 처음부터
                  </Button>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle>측정기기 연결</CardTitle>
                <CardDescription>
                  {bluetoothAvailable() ? "Chrome/Edge · 블루투스를 켜고 기기를 선택하세요" : "이 브라우저는 Web Bluetooth를 지원하지 않습니다"}
                </CardDescription>
              </CardHeader>
              <CardContent className="flex flex-col gap-1.5">
                {DEVICES.map((d) => {
                  const connection = connections[d.kind];
                  const seen = lastSeen[d.kind];
                  return (
                    <div key={d.kind} className="border-border flex items-center justify-between gap-2 rounded-lg border px-3 py-1.5">
                      <div className="min-w-0 text-xs">
                        <div className="font-medium">{d.label}</div>
                        <div className="text-muted-foreground truncate">
                          {connection
                            ? `${connection.name} · ${seen ? `${Math.max(0, Math.round((Date.now() - seen) / 1000))}초 전 수신` : "측정 대기"}`
                            : d.hint}
                        </div>
                      </div>
                      {connection ? (
                        <Button size="sm" variant="ghost" onClick={() => connection.disconnect()} aria-label="연결 해제">
                          <Unplug className="size-3.5" />
                        </Button>
                      ) : (
                        <Button size="sm" variant="outline" className="shrink-0 whitespace-nowrap" onClick={() => connect(d.kind)} disabled={!bluetoothAvailable()}>
                          <Bluetooth className="mr-1 size-3.5" /> 연결
                        </Button>
                      )}
                    </div>
                  );
                })}
                {deviceError && <p className="text-critical text-xs">{deviceError}</p>}
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle>간호 기록 입력</CardTitle>
                <CardDescription>기기가 없는 항목(호흡수 등)은 직접 입력합니다</CardDescription>
              </CardHeader>
              <CardContent className="grid grid-cols-3 gap-2">
                {(["resp_rate", "temperature", "sbp", "dbp", "pulse", "spo2"] as Vital[]).map((v) => (
                  <label key={v} className="text-xs">
                    <span className="text-muted-foreground">{VITAL_META[v].label}</span>
                    <input
                      inputMode="decimal"
                      className="border-border bg-background mt-0.5 w-full rounded-md border px-2 py-1 text-sm"
                      value={manual[v] ?? ""}
                      onChange={(e) => setManual((m) => ({ ...m, [v]: e.target.value }))}
                    />
                  </label>
                ))}
                <Button size="sm" className="col-span-3" variant="outline" onClick={submitManual}>
                  기록
                </Button>
              </CardContent>
            </Card>
          </div>

          {/* Monitor */}
          <div className="flex min-w-0 flex-col gap-4">
            <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
              {VITALS.map((v) => {
                const value = latest[`${v}_last`];
                const ref = baseline[v];
                const meta = VITAL_META[v];
                return (
                  <Card key={v} className="px-3 py-2.5">
                    <div className="text-muted-foreground text-xs">{meta.label}</div>
                    <div className="text-2xl font-semibold tabular-nums">
                      {Number.isFinite(value) ? value.toFixed(meta.digits) : "—"}
                      <span className="text-muted-foreground ml-1 text-xs font-normal">{meta.unit}</span>
                    </div>
                    <div className="text-subtle-foreground text-xs tabular-nums">
                      평소 {ref !== undefined ? ref.toFixed(meta.digits) : "측정 중"}
                    </div>
                  </Card>
                );
              })}
            </div>

            <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
              <Card>
                <CardHeader className="pb-1">
                  <CardTitle>모델 위험도</CardTitle>
                  <CardDescription>보정 전 점수 · 경보 기준 {(threshold * 100).toFixed(1)}% (특이도 95% 지점)</CardDescription>
                </CardHeader>
                <CardContent>
                  {risk === null ? (
                    <p className="text-muted-foreground py-4 text-sm">
                      {result
                        ? `개인 기저선 측정 중… ${result.features.stepsObserved}/${model?.baseline_steps ?? 6}단계`
                        : "시나리오를 시작하거나 기기를 연결하세요"}
                    </p>
                  ) : (
                    <div className="flex items-end gap-3 py-1">
                      <span className="text-5xl font-semibold tabular-nums">{(risk * 100).toFixed(1)}%</span>
                      <span className={`${LEVEL_CLASS[level!]} mb-2 rounded-full px-2.5 py-0.5 text-xs font-semibold text-white`}>
                        {LEVEL_KO[level!]}
                      </span>
                    </div>
                  )}
                </CardContent>
              </Card>
              <Card>
                <CardHeader className="pb-1">
                  <CardTitle>NEWS (기존 임상 점수)</CardTitle>
                  <CardDescription>
                    경보 기준 {NEWS_THRESHOLD}점 · 측정된 {result?.news.measured ?? 0}/5개 항목으로 계산
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="flex items-end gap-3 py-1">
                    <span className="text-5xl font-semibold tabular-nums">{result?.news.measured ? result.news.score : "—"}</span>
                    {result?.news.measured ? (
                      <span
                        className={`${newsAlarm ? "bg-critical" : "bg-success"} mb-2 rounded-full px-2.5 py-0.5 text-xs font-semibold text-white`}
                      >
                        {newsAlarm ? "경보" : "정상"}
                      </span>
                    ) : null}
                  </div>
                </CardContent>
              </Card>
            </div>

            {risk !== null && <Verdict modelAlarm={modelAlarm} newsAlarm={newsAlarm} />}

            <Card>
              <CardHeader className="pb-1">
                <CardTitle>위험도 · NEWS 추이</CardTitle>
                <CardDescription>초 단위 · 점선은 각 경보 기준</CardDescription>
              </CardHeader>
              <CardContent className="h-56">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={history} margin={{ top: 8, right: 16, bottom: 0, left: -8 }}>
                    <XAxis dataKey="t" tick={{ fontSize: 11 }} unit="s" />
                    <YAxis yAxisId="risk" domain={[0, 1]} tickFormatter={(v) => `${Math.round(v * 100)}%`} tick={{ fontSize: 11 }} />
                    <YAxis yAxisId="news" orientation="right" domain={[0, 15]} tick={{ fontSize: 11 }} />
                    <Tooltip
                      formatter={(value: number, name: string) =>
                        name === "risk" ? [`${(value * 100).toFixed(1)}%`, "모델 위험도"] : [value, "NEWS"]
                      }
                      labelFormatter={(t) => `${t}초`}
                    />
                    <ReferenceLine yAxisId="risk" y={threshold} stroke="#2a78d6" strokeDasharray="4 4" />
                    <ReferenceLine yAxisId="news" y={NEWS_THRESHOLD} stroke="#eb6834" strokeDasharray="4 4" />
                    <Line yAxisId="risk" type="monotone" dataKey="risk" stroke="#2a78d6" strokeWidth={2} dot={false} isAnimationActive={false} connectNulls />
                    <Line yAxisId="news" type="stepAfter" dataKey="news" stroke="#eb6834" strokeWidth={2} dot={false} isAnimationActive={false} connectNulls />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-1">
                <CardTitle>왜 이 점수인가</CardTitle>
                <CardDescription>모델 기여도 상위 요인 (트리 경로 기반 근사 · 로그 오즈)</CardDescription>
              </CardHeader>
              <CardContent>
                {result?.factors.length ? (
                  <ul className="flex flex-col gap-1.5">
                    {result.factors.map((f) => (
                      <li key={f.feature} className="flex items-center justify-between gap-3 text-sm">
                        <span>
                          {describe(f.feature)}
                          <span className="text-muted-foreground ml-1 tabular-nums">= {Number.isFinite(f.value) ? f.value.toFixed(2) : "—"}</span>
                        </span>
                        <span className={`tabular-nums text-xs font-semibold ${f.contribution > 0 ? "text-critical" : "text-success"}`}>
                          {f.contribution > 0 ? "위험 ↑" : "위험 ↓"} {Math.abs(f.contribution).toFixed(2)}
                        </span>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="text-muted-foreground text-sm">기저선 측정이 끝나면 근거가 표시됩니다.</p>
                )}
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}

function Verdict({ modelAlarm, newsAlarm }: { modelAlarm: boolean; newsAlarm: boolean }) {
  const [tone, text] =
    modelAlarm && newsAlarm
      ? ["border-critical bg-critical/10", "모델과 NEWS 모두 경보 — 즉시 확인이 필요한 악화 신호입니다."]
      : modelAlarm
        ? ["border-serious bg-serious/10", "모델만 경보 — 아직 정상 범위지만 이 환자의 기록 초기 값에서 벗어나고 있습니다. 활력징후를 다시 확인해 주세요."]
        : newsAlarm
          ? ["border-warning bg-warning/10", "NEWS 경보 — 수치가 이 환자의 기록 초기와 비슷합니다. 모델은 NEWS 경보를 끄지 않으니, 기존 절차대로 확인해 주세요."]
          : ["border-success bg-success/10", "둘 다 경보 없음 — 기록 초기와 비슷한 상태입니다."];
  return <div className={`rounded-lg border-l-4 px-4 py-3 text-sm font-medium ${tone}`}>{text}</div>;
}
