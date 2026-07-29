"use client";

import { useMemo, useState } from "react";
import { ArrowDown, ArrowUp, Biohazard, Minus, Search, TriangleAlert } from "lucide-react";

import { RiskBadge } from "@/components/RiskBadge";
import { Button } from "@/components/ui/button";
import { formatDelta, formatRisk } from "@/lib/risk";
import type { PatientSummary } from "@/lib/types";
import { cn } from "@/lib/utils";

type Filter = "all" | "sepsis" | "disagree";

/**
 * The ward list. A table, not cards — the job is comparing one number down a
 * column, and the risk value is written out so nothing depends on reading a
 * colour.
 *
 * The "불일치" filter is the one worth having: windows where the two scores
 * part ways are where this project's claim lives or dies, and they are
 * otherwise scattered through 400 rows.
 */
export function PatientList({
  patients,
  selectedId,
  onSelect,
  stale,
}: {
  patients: PatientSummary[];
  selectedId: string | null;
  onSelect: (patientId: string) => void;
  stale?: boolean;
}) {
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState<Filter>("all");

  const rows = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return patients.filter((patient) => {
      if (needle && !patient.patient_id.toLowerCase().includes(needle)) return false;
      if (filter === "sepsis") return patient.arrest_hour != null;
      if (filter === "disagree") return patient.model_alarm !== patient.news_alarm;
      return true;
    });
  }, [patients, query, filter]);

  return (
    <div className={cn("flex h-full flex-col", stale && "is-stale")}>
      <div className="space-y-2 px-4 pb-3">
        <label className="relative block">
          <Search
            size={14}
            className="text-subtle-foreground pointer-events-none absolute top-1/2 left-2.5 -translate-y-1/2"
            aria-hidden
          />
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="환자 번호 검색"
            aria-label="환자 번호 검색"
            className="border-border bg-background focus-visible:outline-primary h-8 w-full rounded-lg border pr-2 pl-8 text-xs focus-visible:outline-2 focus-visible:outline-offset-1"
          />
        </label>

        <div className="flex gap-1.5">
          {(
            [
              ["all", "전체"],
              ["sepsis", "발병"],
              ["disagree", "불일치"],
            ] as [Filter, string][]
          ).map(([value, label]) => (
            <Button
              key={value}
              size="sm"
              variant={filter === value ? "default" : "outline"}
              onClick={() => setFilter(value)}
              className="flex-1"
            >
              {label}
            </Button>
          ))}
        </div>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto">
        <table className="w-full border-collapse text-sm">
          <caption className="sr-only">위험도순 환자 목록</caption>
          <thead className="bg-card sticky top-0 z-10">
            <tr className="text-subtle-foreground border-border border-b text-left text-xs">
              <th scope="col" className="px-4 py-2 font-medium">환자</th>
              <th scope="col" className="px-2 py-2 text-right font-medium">위험도</th>
              <th scope="col" className="px-2 py-2 text-right font-medium">NEWS</th>
              <th scope="col" className="px-4 py-2 font-medium">
                실제 결과
                <span className="sr-only"> (정답 라벨, 예측 아님)</span>
              </th>
            </tr>
          </thead>
          <tbody>
            {rows.map((patient) => {
              const selected = patient.patient_id === selectedId;
              const Trend =
                patient.risk_delta > 0.005 ? ArrowUp : patient.risk_delta < -0.005 ? ArrowDown : Minus;
              const disagrees = patient.model_alarm !== patient.news_alarm;

              return (
                <tr
                  key={patient.patient_id}
                  onClick={() => onSelect(patient.patient_id)}
                  aria-selected={selected}
                  className={cn(
                    "border-border cursor-pointer border-b transition",
                    selected
                      ? "bg-[color-mix(in_oklab,var(--primary)_12%,transparent)]"
                      : "hover:bg-muted",
                  )}
                >
                  <th scope="row" className="px-4 py-2 text-left font-medium">
                    <span className="flex items-center gap-2">
                      <span
                        aria-hidden
                        className="h-4 w-0.5 shrink-0 rounded-full"
                        style={{ background: selected ? "var(--primary)" : "transparent" }}
                      />
                      <span className="tabular-nums">{patient.patient_id}</span>
                      {disagrees && (
                        <TriangleAlert
                          size={11}
                          style={{ color: "var(--series-2)" }}
                          aria-label="두 점수가 불일치"
                        />
                      )}
                    </span>
                    <span className="text-subtle-foreground ml-4 flex items-center gap-1 text-[11px] tabular-nums">
                      <Trend size={10} aria-hidden />
                      {formatDelta(patient.risk_delta)}
                      <RiskBadge level={patient.risk_level} className="ml-1" />
                    </span>
                  </th>
                  <td className="px-2 py-2 text-right font-medium tabular-nums">
                    {formatRisk(patient.risk)}
                  </td>
                  <td
                    className={cn(
                      "px-2 py-2 text-right tabular-nums",
                      patient.news_alarm ? "font-medium" : "text-subtle-foreground",
                    )}
                  >
                    {patient.news_score.toFixed(0)}
                  </td>
                  <td className="px-4 py-2">
                    {patient.arrest_hour == null ? (
                      <span className="text-subtle-foreground text-xs">없음</span>
                    ) : (
                      <span
                        className="inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium tabular-nums"
                        style={{
                          color: "var(--critical)",
                          backgroundColor: "color-mix(in oklab, var(--critical) 14%, transparent)",
                        }}
                      >
                        <Biohazard size={11} aria-hidden />
                        {patient.arrest_hour.toFixed(0)}h
                      </span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>

        {rows.length === 0 && (
          <p className="text-subtle-foreground px-4 py-10 text-center text-sm">
            조건에 맞는 환자가 없습니다.
          </p>
        )}
      </div>

      <p className="text-subtle-foreground border-border border-t px-4 py-2 text-[11px]">
        {rows.length}명 표시 / 전체 {patients.length}명 · 실제 발병{" "}
        {patients.filter((patient) => patient.arrest_hour != null).length}명
      </p>
    </div>
  );
}
