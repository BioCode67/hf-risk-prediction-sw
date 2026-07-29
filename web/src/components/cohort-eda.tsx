"use client";

import { useState } from "react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { SERIES } from "@/lib/risk";
import type { Eda, NormalBand } from "@/lib/types";

/**
 * What the cohort looks like before any single patient is opened.
 *
 * Four questions, in the order you have to answer them to trust anything
 * downstream: is there a signal at all, can the model separate the classes, how
 * much warning does an alarm buy, and is the data good enough to ask.
 *
 * All of it is precomputed aggregates over the held-out set — nothing here is
 * derived from an individual, and nothing recomputes in the browser.
 */
export function CohortEda({
  eda,
  normalBand,
  vitalLabels,
}: {
  eda: Eda;
  normalBand: Record<string, NormalBand>;
  vitalLabels: Record<string, { label: string; unit: string }>;
}) {
  const [open, setOpen] = useState(false);

  return (
    <Card>
      <CardHeader className="flex-row items-start justify-between gap-3">
        <div className="space-y-1">
          <CardTitle>코호트 탐색 (EDA)</CardTitle>
          <CardDescription>
            test 환자 {(eda.case_patients + eda.control_patients).toLocaleString()}명 —
            발병 {eda.case_patients.toLocaleString()}명 / 대조군 {eda.control_patients.toLocaleString()}명.
            개별 환자를 열기 전에 확인할 것들입니다.
          </CardDescription>
        </div>
        <Button variant="outline" size="sm" onClick={() => setOpen((value) => !value)}>
          {open ? "접기" : "펼치기"}
        </Button>
      </CardHeader>

      {open && (
        <CardContent className="space-y-6">
          <OnsetAligned eda={eda} normalBand={normalBand} vitalLabels={vitalLabels} />
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
            <RiskSeparation eda={eda} />
            <LeadTime eda={eda} />
          </div>
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
            <GlobalDrivers eda={eda} />
            <Missingness eda={eda} />
          </div>
        </CardContent>
      )}
    </Card>
  );
}

/**
 * The first question: does anything change before onset?
 *
 * Each case patient is aligned at their own onset hour and the vitals averaged
 * across the 24 hours leading into it. If these curves were flat, no model
 * trained on this label could work, and everything else on the page would be
 * noise fitted to noise.
 */
function OnsetAligned({
  eda,
  normalBand,
  vitalLabels,
}: {
  eda: Eda;
  normalBand: Record<string, NormalBand>;
  vitalLabels: Record<string, { label: string; unit: string }>;
}) {
  const vitals = Object.keys(vitalLabels);

  return (
    <section>
      <h4 className="text-sm font-medium">발병 전 24시간, 실제로 변하는가</h4>
      <p className="text-subtle-foreground mt-0.5 mb-3 text-xs leading-relaxed">
        발병 환자를 각자의 발병 시각에 맞춰 정렬한 뒤 평균한 궤적입니다. 0이 발병 시점이고, 초록 선은
        대조군 중앙값입니다. 두 선이 겹쳐 있으면 이 라벨로는 아무것도 학습할 수 없다는 뜻입니다.
      </p>
      <div className="grid grid-cols-1 gap-x-5 gap-y-4 sm:grid-cols-2 xl:grid-cols-3">
        {vitals.map((vital) => {
          const band = normalBand[vital];
          const meta = vitalLabels[vital];
          return (
            <figure key={vital} className="m-0">
              <figcaption className="text-muted-foreground mb-1 text-xs">
                {meta.label}
                <span className="text-subtle-foreground ml-1 text-[11px]">{meta.unit}</span>
              </figcaption>
              <div className="h-24">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={eda.onset_aligned} margin={{ top: 6, right: 4, bottom: 0, left: 0 }}>
                    <CartesianGrid vertical={false} />
                    <XAxis
                      dataKey="hours_before"
                      type="number"
                      domain={[-24, 0]}
                      ticks={[-24, -18, -12, -6, 0]}
                      tickFormatter={(hour: number) => (hour === 0 ? "발병" : `${hour}h`)}
                      tickLine={false}
                      axisLine={false}
                      height={18}
                      tickMargin={4}
                    />
                    <YAxis width={32} domain={["auto", "auto"]} tickLine={false} axisLine={false} />
                    <Tooltip
                      cursor={{ stroke: "var(--axis)", strokeWidth: 1 }}
                      content={({ active, payload, label }) =>
                        active && payload?.length ? (
                          <div className="bg-card border-border rounded-lg border px-2 py-1 text-xs shadow-sm">
                            <span className="text-subtle-foreground">
                              {Number(label) === 0 ? "발병 시점" : `발병 ${-Number(label)}시간 전`} ·{" "}
                            </span>
                            <span className="font-medium tabular-nums">
                              {Number(payload[0].value).toFixed(1)}
                            </span>
                          </div>
                        ) : null
                      }
                    />
                    {band && <ReferenceLine y={band.p50} stroke="var(--success)" strokeWidth={1.5} />}
                    <ReferenceLine x={0} stroke="var(--critical)" strokeWidth={1} />
                    <Line
                      type="monotone"
                      dataKey={vital}
                      stroke={SERIES.model.color}
                      strokeWidth={2}
                      dot={false}
                      connectNulls
                      isAnimationActive={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </figure>
          );
        })}
      </div>
    </section>
  );
}

/**
 * Class separation under the model.
 *
 * Plotted as share *within* each class rather than raw counts: negatives
 * outnumber positives about 80 to 1, so on a count axis the positive class
 * would be a flat line on the floor and the chart would say nothing.
 */
function RiskSeparation({ eda }: { eda: Eda }) {
  return (
    <section>
      <h4 className="text-sm font-medium">위험도 분포 — 두 집단이 갈리는가</h4>
      <p className="text-subtle-foreground mt-0.5 mb-2 text-xs leading-relaxed">
        각 집단 내부 비율입니다. 음성이 양성보다 80배 많아 실제 개수로 그리면 양성이 바닥에 깔려 보이지
        않습니다.
      </p>
      <ul className="text-muted-foreground mb-2 flex gap-3 text-xs">
        <li className="flex items-center gap-1.5">
          <span aria-hidden className="size-2 rounded-full" style={{ background: "var(--critical)" }} />
          발병 윈도우
        </li>
        <li className="flex items-center gap-1.5">
          <span aria-hidden className="size-2 rounded-full" style={{ background: SERIES.model.color }} />
          비발병 윈도우
        </li>
      </ul>
      <div className="h-44">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={eda.risk_histogram} margin={{ top: 4, right: 8, bottom: 6, left: 0 }}>
            <CartesianGrid vertical={false} />
            <XAxis
              dataKey="from"
              type="number"
              domain={[0, 1]}
              ticks={[0, 0.25, 0.5, 0.75, 1]}
              tickFormatter={(value: number) => `${Math.round(value * 100)}%`}
              tickLine={false}
              axisLine={false}
              tickMargin={4}
              label={{ value: "모델 위험도", position: "insideBottom", offset: -4, fontSize: 11 }}
              height={26}
            />
            <YAxis
              width={38}
              tickFormatter={(value: number) => `${Math.round(value * 100)}%`}
              tickLine={false}
              axisLine={false}
            />
            <Tooltip
              cursor={{ stroke: "var(--axis)", strokeWidth: 1 }}
              content={({ active, payload, label }) =>
                active && payload?.length ? (
                  <div className="bg-card border-border space-y-0.5 rounded-lg border px-2.5 py-1.5 text-xs shadow-sm">
                    <p className="text-subtle-foreground">
                      위험도 {Math.round(Number(label) * 100)}~{Math.round(Number(label) * 100) + 5}%
                    </p>
                    {payload.map((entry) => (
                      <p key={entry.name} className="tabular-nums">
                        {entry.name === "positive" ? "발병" : "비발병"}{" "}
                        <span className="font-medium">
                          {(Number(entry.value) * 100).toFixed(1)}%
                        </span>
                      </p>
                    ))}
                  </div>
                ) : null
              }
            />
            <Area
              type="monotone"
              dataKey="negative"
              stroke={SERIES.model.color}
              fill={SERIES.model.color}
              fillOpacity={0.18}
              strokeWidth={2}
              isAnimationActive={false}
            />
            <Area
              type="monotone"
              dataKey="positive"
              stroke="var(--critical)"
              fill="var(--critical)"
              fillOpacity={0.18}
              strokeWidth={2}
              isAnimationActive={false}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}

/** How much warning an alarm actually buys, as a hero number plus its spread. */
function LeadTime({ eda }: { eda: Eda }) {
  const lead = eda.lead_time;
  const hasData = "median_h" in lead;

  return (
    <section>
      <h4 className="text-sm font-medium">Lead-time — 개입할 시간이 있는가</h4>
      <p className="text-subtle-foreground mt-0.5 mb-3 text-xs leading-relaxed">
        발병 환자별로 모델이 <em>처음</em> 경보한 시점과 실제 발병 사이의 간격입니다.
      </p>

      {hasData ? (
        <div className="space-y-3">
          <p className="flex items-baseline gap-2">
            <span className="text-4xl font-semibold tracking-tight">{lead.median_h}</span>
            <span className="text-muted-foreground text-sm">시간 전 (중앙값)</span>
          </p>
          <dl className="text-muted-foreground grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
            <div className="flex justify-between gap-2">
              <dt>사분위 범위</dt>
              <dd className="text-foreground font-medium tabular-nums">
                {lead.p25_h}~{lead.p75_h}h
              </dd>
            </div>
            <div className="flex justify-between gap-2">
              <dt>검출</dt>
              <dd className="text-foreground font-medium tabular-nums">
                {lead.detected}/{lead.case_patients}명 (
                {Math.round((lead.detected / lead.case_patients) * 100)}%)
              </dd>
            </div>
          </dl>
          <p className="text-subtle-foreground text-[11px] leading-relaxed">
            길다고 좋기만 한 건 아닙니다. 첫 경보 기준이라, 계속 경보 상태였던 환자는 lead-time이
            길게 잡히면서 알람 부담도 같이 커집니다. 위의 알람 부담과 함께 읽어야 합니다.
          </p>
        </div>
      ) : (
        <p className="text-subtle-foreground text-sm">발병 환자가 부족해 계산할 수 없습니다.</p>
      )}
    </section>
  );
}

/** Which features carry the decisions overall — the global counterpart of the per-alarm SHAP. */
function GlobalDrivers({ eda }: { eda: Eda }) {
  const max = Math.max(...eda.global_drivers.map((driver) => driver.value), 1e-9);

  return (
    <section>
      <h4 className="text-sm font-medium">전역 기여 요인 (평균 |SHAP|)</h4>
      <p className="text-subtle-foreground mt-0.5 mb-3 text-xs leading-relaxed">
        경보 하나가 아니라 모델 전체가 무엇에 기대고 있는지. `_dev`로 끝나는 항목이 위에 있으면
        개인 기저선 이탈이 실제로 쓰이고 있다는 뜻입니다.
      </p>
      <ul className="space-y-1.5">
        {eda.global_drivers.map((driver) => (
          <li key={driver.feature} className="space-y-1">
            <div className="flex items-baseline justify-between gap-3">
              <span className="text-xs">{driver.description}</span>
              <span className="text-subtle-foreground shrink-0 text-[11px] tabular-nums">
                {driver.value.toFixed(3)}
              </span>
            </div>
            <div className="bg-muted h-1.5 overflow-hidden rounded-full">
              <div
                className="h-full rounded-full"
                style={{ width: `${(driver.value / max) * 100}%`, background: SERIES.model.color }}
              />
            </div>
          </li>
        ))}
      </ul>
    </section>
  );
}

/** Data quality: a vital that is mostly absent cannot carry a trend feature. */
function Missingness({ eda }: { eda: Eda }) {
  return (
    <section>
      <h4 className="text-sm font-medium">결측률 — 물어볼 수 있는 질문인가</h4>
      <p className="text-subtle-foreground mt-0.5 mb-3 text-xs leading-relaxed">
        윈도우에서 해당 활력징후를 한 번도 관측하지 못한 비율. 높을수록 그 vital의 추세 피처는
        의미가 옅어집니다.
      </p>
      <div className="h-44">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={eda.missingness}
            layout="vertical"
            margin={{ top: 4, right: 40, bottom: 4, left: 0 }}
          >
            <CartesianGrid horizontal={false} />
            <XAxis
              type="number"
              domain={[0, 1]}
              tickFormatter={(value: number) => `${Math.round(value * 100)}%`}
              tickLine={false}
              axisLine={false}
            />
            <YAxis
              type="category"
              dataKey="label"
              width={70}
              tickLine={false}
              axisLine={false}
            />
            <Tooltip
              cursor={{ fill: "color-mix(in oklab, var(--axis) 18%, transparent)" }}
              content={({ active, payload }) =>
                active && payload?.length ? (
                  <div className="bg-card border-border rounded-lg border px-2.5 py-1.5 text-xs shadow-sm">
                    <span className="font-medium tabular-nums">
                      결측 {(Number(payload[0].value) * 100).toFixed(1)}%
                    </span>
                  </div>
                ) : null
              }
            />
            <Bar dataKey="missing" radius={[0, 4, 4, 0]} isAnimationActive={false}>
              {eda.missingness.map((row) => (
                <Cell
                  key={row.vital}
                  // Missingness is a magnitude with a bad end, not an identity:
                  // past a half the feature is effectively unusable, so that
                  // crossing gets the status colour rather than a series hue.
                  fill={row.missing > 0.5 ? "var(--critical)" : SERIES.model.color}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}
