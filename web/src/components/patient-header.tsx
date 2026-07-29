"use client";

import { Biohazard, Clock, ShieldCheck, Timer, User } from "lucide-react";

import { RiskBadge } from "@/components/RiskBadge";
import { Card } from "@/components/ui/card";
import { formatRisk } from "@/lib/risk";
import type { PatientDetail, PatientSummary } from "@/lib/types";

/**
 * Patient identity strip. Selection lives in the ward list beside it, so this
 * only reports who is on screen.
 *
 * Only fields the source record actually carries are shown. Challenge-2019 has
 * age and sex and nothing else, so there is no bed number, ward or admission
 * timestamp here; inventing them would make a demo look like a real chart.
 */
export function PatientHeader({
  detail,
  summary,
}: {
  detail: PatientDetail | null;
  summary: PatientSummary | null;
}) {
  const evidence = detail?.evidence;

  return (
    <Card className="flex flex-col gap-3 px-5 py-4 lg:flex-row lg:items-center lg:justify-between">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
        <div className="flex items-center gap-2.5">
          <span className="bg-muted flex size-9 items-center justify-center rounded-lg">
            <User className="size-4.5" aria-hidden />
          </span>
          <div className="leading-tight">
            <p className="text-subtle-foreground text-[11px]">환자</p>
            <p className="text-base font-semibold tabular-nums">{detail?.patient_id ?? "—"}</p>
          </div>
        </div>

        <Pill icon={User} label={demographyLabel(detail)} />
        <Pill icon={Clock} label={`관찰 ${detail?.hours_observed ?? "—"}개 윈도우`} />
        {evidence?.hours_to_arrest != null && (
          <Pill icon={Timer} label={`보고 있는 시점: 발병 ${evidence.hours_to_arrest.toFixed(0)}시간 전`} />
        )}
        <GroundTruth detail={detail} />
      </div>

      <div className="flex items-center gap-3">
        {evidence && <RiskBadge level={evidence.risk_level} size="lg" />}
        {summary && (
          <span className="text-muted-foreground text-xs tabular-nums">
            최근 위험도 <span className="text-foreground font-semibold">{formatRisk(summary.risk)}</span>
          </span>
        )}
      </div>
    </Card>
  );
}

/**
 * The recorded outcome, not a prediction.
 *
 * Kept visually distinct from the risk badge beside it — one is the answer key
 * and the other is the model's guess, and a screen that lets those blur is
 * worse than useless when you are judging whether the model was right.
 */
function GroundTruth({ detail }: { detail: PatientDetail | null }) {
  if (!detail) return null;
  const onset = detail.arrest_hour;
  const color = onset == null ? "var(--success)" : "var(--critical)";

  return (
    <span
      className="inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-medium"
      style={{ color, backgroundColor: `color-mix(in oklab, ${color} 14%, transparent)` }}
    >
      {onset == null ? <ShieldCheck className="size-3.5" aria-hidden /> : <Biohazard className="size-3.5" aria-hidden />}
      <span className="text-subtle-foreground font-normal">실제 결과</span>
      {onset == null ? "패혈증 없음" : `${onset.toFixed(0)}시간째 발병`}
    </span>
  );
}

function demographyLabel(detail: PatientDetail | null) {
  if (!detail) return "—";
  const parts = [
    detail.age == null ? null : `${detail.age}세`,
    detail.sex == null ? null : detail.sex === "M" ? "남" : "여",
  ].filter(Boolean);
  return parts.length ? parts.join(" · ") : "인적사항 없음";
}

function Pill({
  icon: Icon,
  label,
}: {
  icon: React.ComponentType<{ className?: string; "aria-hidden"?: boolean }>;
  label: string;
}) {
  return (
    <span className="border-border text-muted-foreground inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs">
      <Icon className="size-3.5" aria-hidden />
      {label}
    </span>
  );
}
