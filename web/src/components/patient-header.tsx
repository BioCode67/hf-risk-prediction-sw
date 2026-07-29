"use client";

import { Clock, Timer, User } from "lucide-react";

import { RiskBadge } from "@/components/RiskBadge";
import { Card } from "@/components/ui/card";
import { formatRisk } from "@/lib/risk";
import type { PatientDetail, PatientSummary } from "@/lib/types";

/**
 * Patient identity strip, with the selector the reference design lacks — that
 * mock showed a single hard-coded patient, and this cohort has 400.
 *
 * Only fields the source record actually carries are shown. Challenge-2019 has
 * age and sex and nothing else, so there is no bed number, ward or admission
 * timestamp here; inventing them would make a demo look like a real chart.
 */
export function PatientHeader({
  patients,
  detail,
  selectedId,
  onSelect,
}: {
  patients: PatientSummary[];
  detail: PatientDetail | null;
  selectedId: string | null;
  onSelect: (patientId: string) => void;
}) {
  const summary = patients.find((patient) => patient.patient_id === selectedId) ?? null;
  const evidence = detail?.evidence;

  return (
    <Card className="flex flex-col gap-3 px-5 py-4 lg:flex-row lg:items-center lg:justify-between">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
        <div className="flex items-center gap-2.5">
          <span className="bg-muted flex size-9 items-center justify-center rounded-lg">
            <User className="size-4.5" aria-hidden />
          </span>
          <div className="leading-tight">
            <label htmlFor="patient-select" className="text-subtle-foreground text-[11px]">
              환자
            </label>
            <select
              id="patient-select"
              value={selectedId ?? ""}
              onChange={(event) => onSelect(event.target.value)}
              className="border-border bg-card block rounded-md border px-2 py-1 text-sm font-semibold tabular-nums"
            >
              {patients.map((patient) => (
                <option key={patient.patient_id} value={patient.patient_id}>
                  {patient.patient_id} · {formatRisk(patient.risk)}
                </option>
              ))}
            </select>
          </div>
        </div>

        <Pill icon={User} label={demographyLabel(detail)} />
        <Pill icon={Clock} label={`관찰 ${detail?.hours_observed ?? "—"}개 윈도우`} />
        <Pill
          icon={Timer}
          label={
            evidence?.hours_to_arrest != null
              ? `발병 ${evidence.hours_to_arrest.toFixed(0)}시간 전 시점`
              : "발병 없음 (대조군)"
          }
        />
      </div>

      <div className="flex items-center gap-3">
        {evidence && <RiskBadge level={evidence.risk_level} size="lg" />}
        {summary && (
          <span className="text-subtle-foreground text-xs tabular-nums">
            최근 위험도 <span className="text-foreground font-semibold">{formatRisk(summary.risk)}</span>
          </span>
        )}
      </div>
    </Card>
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
