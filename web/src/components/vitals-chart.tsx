"use client";

import { Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { SERIES } from "@/lib/risk";
import type { TrajectoryPoint, VitalPoint } from "@/lib/types";

/**
 * Vital-sign trends as small multiples — one plot per vital, each on its own
 * scale. A single frame would need six y-axes (SpO2 %, SBP mmHg, °C …), and any
 * shared scale would flatten five of them into noise.
 *
 * The reference design showed a "6-hour forward projection" on this chart. There
 * is none here: the model outputs a risk score, not forecast vitals, and drawing
 * an invented future onto a clinical trend is the one thing this panel must not
 * do. The forward-looking part lives in the prediction card, as a probability.
 */
export function VitalsChart({
  trajectory,
  vitals,
  selectedHour,
  eventHour,
}: {
  trajectory: TrajectoryPoint[];
  vitals: VitalPoint[];
  selectedHour: number | null;
  eventHour: number | null;
}) {
  return (
    <Card className="h-full">
      <CardHeader>
        <CardTitle>활력징후 추이</CardTitle>
        <CardDescription>
          관찰된 전 구간. 가로선은 이 환자의 초기 안정기 평균이고, 세로선은 지금 보고 있는 시점입니다.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 gap-x-5 gap-y-4 sm:grid-cols-2 xl:grid-cols-3">
          {vitals.map((vital) => (
            <VitalPlot
              key={vital.vital}
              vital={vital}
              trajectory={trajectory}
              selectedHour={selectedHour}
              eventHour={eventHour}
            />
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

function VitalPlot({
  vital,
  trajectory,
  selectedHour,
  eventHour,
}: {
  vital: VitalPoint;
  trajectory: TrajectoryPoint[];
  selectedHour: number | null;
  eventHour: number | null;
}) {
  const series = trajectory.map((point) => ({
    hour: point.hour,
    value: point[vital.vital] as number | null,
  }));

  return (
    <figure className="m-0">
      <figcaption className="mb-1 flex items-baseline justify-between gap-2">
        <span className="text-muted-foreground text-xs">{vital.label}</span>
        <span className="text-xs font-semibold tabular-nums">
          {vital.value?.toFixed(1) ?? "—"}
          <span className="text-subtle-foreground ml-0.5 font-normal">{vital.unit}</span>
        </span>
      </figcaption>

      <div className="h-24">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={series} margin={{ top: 6, right: 4, bottom: 0, left: 0 }}>
            <XAxis dataKey="hour" hide />
            <YAxis width={32} domain={["dataMin - 2", "dataMax + 2"]} tickLine={false} axisLine={false} />
            <Tooltip
              cursor={{ stroke: "var(--axis)", strokeWidth: 1 }}
              content={({ active, payload, label }) =>
                active && payload?.length ? (
                  <div className="bg-card border-border rounded-lg border px-2 py-1 text-xs shadow-sm">
                    <span className="text-subtle-foreground">{label}h · </span>
                    <span className="font-medium tabular-nums">
                      {Number(payload[0].value).toFixed(1)} {vital.unit}
                    </span>
                  </div>
                ) : null
              }
            />
            {vital.baseline != null && (
              <ReferenceLine y={vital.baseline} stroke="var(--axis)" strokeWidth={1} />
            )}
            {eventHour != null && (
              <ReferenceLine x={eventHour} stroke="var(--critical)" strokeWidth={1} />
            )}
            {selectedHour != null && (
              <ReferenceLine x={selectedHour} stroke="var(--series-1)" strokeWidth={1} strokeOpacity={0.5} />
            )}
            <Line
              type="monotone"
              dataKey="value"
              stroke={SERIES.model.color}
              strokeWidth={2}
              dot={false}
              connectNulls
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <p className="text-subtle-foreground text-[11px] tabular-nums">
        {vital.baseline == null ? "기저선 없음" : `기저선 ${vital.baseline.toFixed(1)} ${vital.unit}`}
      </p>
    </figure>
  );
}
