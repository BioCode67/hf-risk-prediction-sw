"use client";

import { ArrowDown, ArrowUp, Minus } from "lucide-react";

import { Card } from "@/components/ui/card";
import type { VitalPoint } from "@/lib/types";

/**
 * Current vitals as stat tiles — six numbers, so six tiles rather than a chart.
 *
 * The delta is against **this patient's own baseline**, not a ward threshold.
 * That is the project's whole thesis, so the baseline is printed on every tile
 * instead of being implied by a colour.
 *
 * Deliberately no severity colouring here. Whether a value is dangerous is the
 * model's call, made from the whole window and shown in the prediction card; a
 * per-vital red pill would be a second, contradictory verdict rendered by the UI.
 */
export function VitalsKpiCards({ vitals }: { vitals: VitalPoint[] }) {
  return (
    <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
      {vitals.map((vital) => (
        <VitalTile key={vital.vital} vital={vital} />
      ))}
    </div>
  );
}

function VitalTile({ vital }: { vital: VitalPoint }) {
  const deviation =
    vital.value != null && vital.baseline != null ? vital.value - vital.baseline : null;
  const Trend = deviation == null || Math.abs(deviation) < 0.05 ? Minus : deviation > 0 ? ArrowUp : ArrowDown;

  return (
    <Card className="px-4 py-3">
      <p className="text-subtle-foreground text-xs">{vital.label}</p>
      <p className="mt-0.5 flex items-baseline gap-1">
        <span className="text-2xl font-semibold tracking-tight">{vital.value?.toFixed(1) ?? "—"}</span>
        <span className="text-subtle-foreground text-xs">{vital.unit}</span>
      </p>
      {vital.baseline == null ? (
        <p className="text-subtle-foreground mt-1 text-[11px]">기저선 없음</p>
      ) : (
        <p className="text-muted-foreground mt-1 flex items-center gap-1 text-[11px] tabular-nums">
          <Trend className="size-3" aria-hidden />
          <span>
            {deviation == null
              ? "—"
              : `${deviation > 0 ? "+" : deviation < 0 ? "−" : "±"}${Math.abs(deviation).toFixed(1)}`}
          </span>
          <span className="text-subtle-foreground">평소 {vital.baseline.toFixed(1)}</span>
        </p>
      )}
    </Card>
  );
}
