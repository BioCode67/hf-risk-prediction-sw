"use client";

import { useState } from "react";
import { ShieldQuestion } from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  LabelList,
  Line,
  LineChart,
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { alarmRanges } from "@/lib/alarms";
import { SERIES, formatRisk } from "@/lib/risk";
import type { BurdenRow, Evidence, PatientDetail } from "@/lib/types";

/**
 * The model against NEWS — per patient over time, and over the whole cohort.
 *
 * Left: both scores across this patient's stay, as two stacked single-series
 * plots. Not one plot with two y-axes: a probability and a 0–15 ordinal score
 * share no scale, and overlaying them fabricates a correlation.
 *
 * Right: alarms per 100 windows once both are tuned to the *same* detection
 * rate — the only comparison that means anything. Specificity travels with it,
 * because a NEWS column near zero specificity means "alarms on everybody", and
 * beating that is arithmetic rather than evidence.
 */
export function AlarmNews({
  detail,
  burden,
  selectedHour,
  onSelectHour,
}: {
  detail: PatientDetail;
  burden: BurdenRow[];
  selectedHour: number | null;
  onSelectHour: (hour: number) => void;
}) {
  const pinned = selectedHour ?? detail.timeline.at(-1)?.hour ?? null;
  const point = detail.timeline.find((item) => item.hour === pinned) ?? null;

  const handleClick = (next: { activeLabel?: string | number }) => {
    if (next?.activeLabel != null) onSelectHour(Number(next.activeLabel));
  };

  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-5">
      <Card className="lg:col-span-3">
        <CardHeader>
          <CardTitle>경보 추이 — 본 모델 vs NEWS</CardTitle>
          <CardDescription>
            점을 클릭하면 그 시점의 근거를 봅니다. 두 점수는 축이 달라 따로 그리고, 붉은 음영은 각 점수가 경보한 구간입니다.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-1">
          <p className="text-muted-foreground text-xs">
            {point ? (
              <>
                <span className="text-foreground font-medium tabular-nums">{point.hour}시간째</span> · 위험도{" "}
                <span className="text-foreground font-medium tabular-nums">{formatRisk(point.risk)}</span> · NEWS{" "}
                <span className="text-foreground font-medium tabular-nums">{point.news_score.toFixed(0)}</span>
              </>
            ) : (
              "선택된 시점이 없습니다."
            )}
          </p>

          <TimelinePlot
            height={150}
            label="본 모델 위험도"
            color={SERIES.model.color}
            data={detail.timeline}
            dataKey="risk"
            domain={[0, 1]}
            tickFormatter={(value) => `${Math.round(value * 100)}%`}
            threshold={detail.threshold}
            eventHour={detail.arrest_hour}
            pinned={pinned}
            onClick={handleClick}
            formatValue={formatRisk}
          />
          <TimelinePlot
            height={120}
            label="NEWS (임상 표준)"
            color={SERIES.news.color}
            data={detail.timeline}
            dataKey="news_score"
            domain={[0, "dataMax + 2"]}
            tickFormatter={(value) => value.toFixed(0)}
            threshold={detail.news_threshold}
            eventHour={detail.arrest_hour}
            pinned={pinned}
            onClick={handleClick}
            formatValue={(value) => value.toFixed(0)}
            showXAxis
          />

          <Disagreement evidence={detail.evidence} />
        </CardContent>
      </Card>

      <Card className="lg:col-span-2">
        <CardHeader>
          <CardTitle>알람 부담 — 같은 검출률에서</CardTitle>
          <CardDescription>
            두 점수를 같은 민감도로 맞춘 뒤, 그 검출률에 드는 알람 수(윈도우 100개당).
          </CardDescription>
        </CardHeader>
        <CardContent>
          <BurdenPanel burden={burden} />
        </CardContent>
      </Card>
    </div>
  );
}

function TimelinePlot({
  height,
  label,
  color,
  data,
  dataKey,
  domain,
  tickFormatter,
  threshold,
  eventHour,
  pinned,
  onClick,
  formatValue,
  showXAxis = false,
}: {
  height: number;
  label: string;
  color: string;
  data: PatientDetail["timeline"];
  dataKey: "risk" | "news_score";
  domain: [number, number | string];
  tickFormatter: (value: number) => string;
  threshold: number;
  eventHour: number | null;
  pinned: number | null;
  onClick: (next: { activeLabel?: string | number }) => void;
  formatValue: (value: number) => string;
  showXAxis?: boolean;
}) {
  // Shade this score's own alarm stretches — the NEWS plot marks NEWS alarms,
  // so the two panels read as a like-for-like comparison rather than one
  // score's bands imposed on the other.
  const ranges = alarmRanges(data, threshold, dataKey);

  return (
    <figure className="m-0">
      <figcaption className="text-subtle-foreground mb-1 flex items-center gap-1.5 text-xs">
        <span aria-hidden className="h-0.5 w-3 rounded-full" style={{ background: color }} />
        {label}
      </figcaption>
      <div style={{ height }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 4, right: 12, bottom: showXAxis ? 6 : 0, left: 0 }} onClick={onClick}>
            <CartesianGrid vertical={false} />
            <XAxis
              dataKey="hour"
              type="number"
              domain={["dataMin", "dataMax"]}
              hide={!showXAxis}
              tickLine={false}
              axisLine={false}
              tickMargin={6}
              tickFormatter={(hour: number) => `${hour}h`}
              minTickGap={28}
              height={26}
              label={
                showXAxis
                  ? { value: "기록 시작 후 경과 시간", position: "insideBottom", offset: -4, fontSize: 11 }
                  : undefined
              }
            />
            {ranges.map((range) => (
              <ReferenceArea
                key={`${range.from}-${range.to}`}
                x1={range.from}
                x2={range.to}
                fill="var(--critical)"
                fillOpacity={0.12}
                strokeOpacity={0}
              />
            ))}
            <YAxis domain={domain} tickFormatter={tickFormatter} width={40} tickLine={false} axisLine={false} />
            <Tooltip
              cursor={{ stroke: "var(--axis)", strokeWidth: 1 }}
              content={({ active, payload, label: hour }) =>
                active && payload?.length ? (
                  <div className="bg-card border-border rounded-lg border px-2.5 py-1.5 text-xs shadow-sm">
                    <p className="text-subtle-foreground mb-0.5">{hour}시간째</p>
                    <p className="font-medium tabular-nums">{formatValue(Number(payload[0].value))}</p>
                  </div>
                ) : null
              }
            />
            <ReferenceLine
              y={threshold}
              stroke="var(--axis)"
              strokeWidth={1}
              label={{ value: "경보 기준", position: "insideTopLeft", fontSize: 10, fill: "var(--subtle-foreground)" }}
            />
            {eventHour != null && <ReferenceLine x={eventHour} stroke="var(--critical)" strokeWidth={1} />}
            {pinned != null && <ReferenceLine x={pinned} stroke={color} strokeWidth={1} strokeOpacity={0.45} />}
            <Line
              type="monotone"
              dataKey={dataKey}
              stroke={color}
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 4, strokeWidth: 2, stroke: "var(--card)" }}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </figure>
  );
}

function BurdenPanel({ burden }: { burden: BurdenRow[] }) {
  const [asTable, setAsTable] = useState(false);
  const data = burden.map((row) => ({
    label: `${Math.round(row.sensitivity * 100)}%`,
    model: row.model_alarms_per_100,
    news: row.news_alarms_per_100,
  }));

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between gap-3">
        <ul className="text-muted-foreground flex gap-3 text-xs">
          {[SERIES.model, SERIES.news].map((series) => (
            <li key={series.label} className="flex items-center gap-1.5">
              <span aria-hidden className="size-2 rounded-full" style={{ background: series.color }} />
              {series.label}
            </li>
          ))}
        </ul>
        <Button variant="ghost" size="sm" onClick={() => setAsTable((value) => !value)}>
          {asTable ? "그래프" : "표"}
        </Button>
      </div>

      {asTable ? (
        <div className="overflow-x-auto">
          <table className="w-full border-collapse text-xs">
            <caption className="sr-only">검출률별 알람 부담 및 특이도</caption>
            <thead>
              <tr className="text-subtle-foreground border-border border-b text-left">
                <th scope="col" className="py-1.5 pr-2 font-medium">검출률</th>
                <th scope="col" className="py-1.5 pr-2 text-right font-medium">모델</th>
                <th scope="col" className="py-1.5 pr-2 text-right font-medium">특이도</th>
                <th scope="col" className="py-1.5 pr-2 text-right font-medium">NEWS</th>
                <th scope="col" className="py-1.5 text-right font-medium">특이도</th>
              </tr>
            </thead>
            <tbody className="tabular-nums">
              {burden.map((row) => (
                <tr key={row.sensitivity} className="border-border border-b">
                  <th scope="row" className="py-1.5 pr-2 text-left font-medium">
                    {Math.round(row.sensitivity * 100)}%
                  </th>
                  <td className="py-1.5 pr-2 text-right">{row.model_alarms_per_100.toFixed(1)}</td>
                  <td className="text-muted-foreground py-1.5 pr-2 text-right">
                    {row.model_specificity.toFixed(3)}
                  </td>
                  <td className="py-1.5 pr-2 text-right">{row.news_alarms_per_100.toFixed(1)}</td>
                  <td className="text-muted-foreground py-1.5 text-right">
                    {row.news_specificity.toFixed(3)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="h-56">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data} margin={{ top: 16, right: 8, bottom: 6, left: 0 }} barGap={2}>
              <CartesianGrid vertical={false} />
              <XAxis
                dataKey="label"
                tickLine={false}
                axisLine={false}
                tickMargin={6}
                label={{ value: "검출률(민감도)", position: "insideBottom", offset: -4, fontSize: 11 }}
              />
              <YAxis width={34} tickLine={false} axisLine={false} />
              <Tooltip
                cursor={{ fill: "color-mix(in oklab, var(--axis) 18%, transparent)" }}
                content={({ active, payload, label }) =>
                  active && payload?.length ? (
                    <div className="bg-card border-border space-y-0.5 rounded-lg border px-2.5 py-1.5 text-xs shadow-sm">
                      <p className="text-subtle-foreground">검출률 {label}</p>
                      {payload.map((entry) => (
                        <p key={entry.name} className="tabular-nums">
                          {entry.name === "model" ? SERIES.model.label : SERIES.news.label}{" "}
                          <span className="font-medium">{Number(entry.value).toFixed(1)}</span> 알람/100
                        </p>
                      ))}
                    </div>
                  ) : null
                }
              />
              <Bar dataKey="model" fill={SERIES.model.color} radius={[4, 4, 0, 0]} isAnimationActive={false}>
                <LabelList dataKey="model" position="top" fontSize={10} fill="var(--muted-foreground)" />
              </Bar>
              <Bar dataKey="news" fill={SERIES.news.color} radius={[4, 4, 0, 0]} isAnimationActive={false}>
                <LabelList dataKey="news" position="top" fontSize={10} fill="var(--muted-foreground)" />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}

/** Rendered only when the two scores disagree — that is the whole point. */
function Disagreement({ evidence }: { evidence: Evidence }) {
  if (evidence.disagreement === "none") return null;
  const newsOnly = evidence.disagreement === "news_only";
  const color = newsOnly ? SERIES.news.color : "var(--critical)";

  return (
    <div
      className="mt-2 flex gap-2.5 rounded-lg p-3"
      style={{ background: `color-mix(in oklab, ${color} 10%, transparent)` }}
    >
      <ShieldQuestion size={16} className="mt-0.5 shrink-0" style={{ color }} aria-hidden />
      <div className="space-y-0.5">
        <p className="text-sm font-medium">
          {newsOnly ? "NEWS는 경보, 본 모델은 경보 없음" : "본 모델은 경보, NEWS는 경보 없음"}
        </p>
        <p className="text-muted-foreground text-xs leading-relaxed">
          {newsOnly
            ? "고정 임계값이 울린 지점입니다. 위 기여 요인이 침묵을 정당화하는지 확인하세요 — 개인 기저선이 이유라면 오경보를 줄인 것이고, 아니라면 따져봐야 할 사례입니다."
            : "고정 임계값은 아직 조용한데 개인 기저선 이탈이 잡힌 지점입니다. 조기 검출이라면 여기가 lead-time이 벌어지는 자리입니다."}
        </p>
      </div>
    </div>
  );
}
