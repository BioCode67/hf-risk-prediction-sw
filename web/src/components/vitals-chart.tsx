"use client";

import {
  Line,
  LineChart,
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { alarmRanges } from "@/lib/alarms";
import { SERIES } from "@/lib/risk";
import type { TimelinePoint, TrajectoryPoint, VitalPoint } from "@/lib/types";

/**
 * Vital-sign trends as small multiples — one plot per vital, each on its own
 * scale. A single frame would need six y-axes (SpO2 %, SBP mmHg, °C …), and any
 * shared scale would flatten five of them into noise.
 *
 * Every plot carries the same x-axis in hours since the record starts, so a
 * reading can be placed in time without counting gridlines, and the stretches
 * where the model was alarming are shaded across all six at once — which is
 * what makes "the alarm fired while SpO2 was already falling" visible at a
 * glance rather than inferable.
 *
 * There is no forward projection here. The model outputs a risk score, not
 * forecast vitals, and drawing an invented future onto a clinical trend is the
 * one thing this panel must not do.
 */
export function VitalsChart({
  trajectory,
  vitals,
  timeline,
  threshold,
  selectedHour,
  eventHour,
}: {
  trajectory: TrajectoryPoint[];
  vitals: VitalPoint[];
  timeline: TimelinePoint[];
  threshold: number;
  selectedHour: number | null;
  eventHour: number | null;
}) {
  const ranges = alarmRanges(timeline, threshold);

  return (
    <div className="bg-card text-card-foreground border-border h-full rounded-xl border shadow-sm">
      <div className="flex flex-col gap-1 px-5 pt-4 pb-3">
        <h3 className="text-sm font-semibold tracking-tight">활력징후 추이</h3>
        <p className="text-subtle-foreground text-xs leading-relaxed">
          가로축은 기록 시작 후 경과 시간(시간)입니다. 가로선은 이 환자의 초기 안정기 평균입니다.
        </p>
        <ul className="text-subtle-foreground mt-1 flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px]">
          <li className="flex items-center gap-1.5">
            <span
              aria-hidden
              className="h-3 w-4 rounded-[2px]"
              style={{ background: "color-mix(in oklab, var(--critical) 18%, transparent)" }}
            />
            모델 경보 구간 {ranges.length > 0 && `(${ranges.length}회)`}
          </li>
          {eventHour != null && (
            <li className="flex items-center gap-1.5">
              <span aria-hidden className="h-3 w-0.5" style={{ background: "var(--critical)" }} />
              실제 발병 {eventHour.toFixed(0)}h
            </li>
          )}
          <li className="flex items-center gap-1.5">
            <span aria-hidden className="h-3 w-0.5" style={{ background: "var(--primary)" }} />
            보고 있는 시점 {selectedHour != null && `${selectedHour}h`}
          </li>
        </ul>
      </div>

      <div className="grid grid-cols-1 gap-x-5 gap-y-4 px-5 pb-5 sm:grid-cols-2 xl:grid-cols-3">
        {vitals.map((vital) => (
          <VitalPlot
            key={vital.vital}
            vital={vital}
            trajectory={trajectory}
            ranges={ranges}
            selectedHour={selectedHour}
            eventHour={eventHour}
          />
        ))}
      </div>
    </div>
  );
}

function VitalPlot({
  vital,
  trajectory,
  ranges,
  selectedHour,
  eventHour,
}: {
  vital: VitalPoint;
  trajectory: TrajectoryPoint[];
  ranges: { from: number; to: number }[];
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

      {/* Height covers the plot *and* the axis band, so the card never nests a scrollbar. */}
      <div className="h-28">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={series} margin={{ top: 6, right: 4, bottom: 0, left: 0 }}>
            <XAxis
              dataKey="hour"
              type="number"
              domain={["dataMin", "dataMax"]}
              tickLine={false}
              axisLine={false}
              tickMargin={4}
              tickFormatter={(hour: number) => `${hour}h`}
              minTickGap={24}
              height={18}
            />
            <YAxis width={32} domain={["dataMin - 2", "dataMax + 2"]} tickLine={false} axisLine={false} />
            <Tooltip
              cursor={{ stroke: "var(--axis)", strokeWidth: 1 }}
              content={({ active, payload, label }) =>
                active && payload?.length ? (
                  <div className="bg-card border-border rounded-lg border px-2 py-1 text-xs shadow-sm">
                    <span className="text-subtle-foreground">기록 시작 +{label}시간 · </span>
                    <span className="font-medium tabular-nums">
                      {Number(payload[0].value).toFixed(1)} {vital.unit}
                    </span>
                  </div>
                ) : null
              }
            />

            {ranges.map((range) => (
              <ReferenceArea
                key={`${range.from}-${range.to}`}
                x1={range.from}
                x2={range.to}
                fill="var(--critical)"
                fillOpacity={0.14}
                strokeOpacity={0}
              />
            ))}

            {vital.baseline != null && (
              <ReferenceLine y={vital.baseline} stroke="var(--axis)" strokeWidth={1} />
            )}
            {eventHour != null && (
              <ReferenceLine x={eventHour} stroke="var(--critical)" strokeWidth={1.5} />
            )}
            {selectedHour != null && (
              <ReferenceLine x={selectedHour} stroke="var(--primary)" strokeWidth={1} />
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
