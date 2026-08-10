"use client";

import { useMemo, useRef, useState } from "react";
import { ArrowDown, ArrowUp, Biohazard, ChevronDown, ChevronUp, Minus, Search, ShieldCheck, TriangleAlert } from "lucide-react";

import { RiskBadge } from "@/components/RiskBadge";
import { Button } from "@/components/ui/button";
import { formatDelta, formatRisk } from "@/lib/risk";
import type { PatientSummary } from "@/lib/types";
import { cn } from "@/lib/utils";

type Filter = "all" | "sepsis" | "disagree";
type SortKey = "patient_id" | "risk" | "news_score";
type SortDir = "asc" | "desc";

/**
 * The ward list. A table, not cards — the job is comparing one number down a
 * column, and the risk value is written out so nothing depends on reading a
 * colour.
 *
 * Three columns, not four. The sidebar is ~320px and a fourth column pushed the
 * table to 375px, where `overflow-y-auto` silently cropped the outcome cell off
 * the right edge with no scrollbar to reveal it. The ground-truth chip moved
 * onto the row's second line instead, which is also where it belongs: it is an
 * annotation on the patient, not a quantity you scan down a column.
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
  const [sort, setSort] = useState<{ key: SortKey; dir: SortDir }>({ key: "risk", dir: "desc" });
  const body = useRef<HTMLTableSectionElement>(null);

  const rows = useMemo(() => {
    const needle = query.trim().toLowerCase();
    const filtered = patients.filter((patient) => {
      if (needle && !patient.patient_id.toLowerCase().includes(needle)) return false;
      if (filter === "sepsis") return patient.arrest_hour != null;
      if (filter === "disagree") return patient.model_alarm !== patient.news_alarm;
      return true;
    });

    const sign = sort.dir === "asc" ? 1 : -1;
    return [...filtered].sort((a, b) => {
      if (sort.key === "patient_id") {
        return sign * a.patient_id.localeCompare(b.patient_id, undefined, { numeric: true });
      }
      // Ties on a coarse column (NEWS is a 0–15 integer) fall back to risk, so
      // the order is total and re-sorting never reshuffles equal rows.
      return sign * (a[sort.key] - b[sort.key]) || b.risk - a.risk;
    });
  }, [patients, query, filter, sort]);

  const toggleSort = (key: SortKey) =>
    setSort((current) =>
      current.key === key
        ? { key, dir: current.dir === "desc" ? "asc" : "desc" }
        : { key, dir: key === "patient_id" ? "asc" : "desc" },
    );

  /**
   * Roving tabindex: one stop for the whole list, arrows to walk it. Tabbing
   * through 400 rows to reach the next panel would be worse than no keyboard
   * support at all.
   */
  const onKeyDown = (event: React.KeyboardEvent<HTMLTableRowElement>, patientId: string) => {
    const step = event.key === "ArrowDown" ? 1 : event.key === "ArrowUp" ? -1 : 0;
    if (step !== 0) {
      event.preventDefault();
      const all = Array.from(body.current?.querySelectorAll<HTMLTableRowElement>("tr") ?? []);
      const next = all[all.indexOf(event.currentTarget) + step];
      next?.focus();
      return;
    }
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      onSelect(patientId);
    }
  };

  // The list's own tab stop follows the selection, falling back to the first
  // row so the list is always reachable even before anything is chosen.
  const focusable = rows.some((patient) => patient.patient_id === selectedId)
    ? selectedId
    : (rows[0]?.patient_id ?? null);

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
              aria-pressed={filter === value}
              className="flex-1"
            >
              {label}
            </Button>
          ))}
        </div>
      </div>

      <div className="min-h-0 flex-1 overflow-x-auto overflow-y-auto">
        <table className="w-full border-collapse text-sm">
          <caption className="sr-only">
            환자 목록. 위·아래 화살표로 이동하고 Enter로 선택합니다.
          </caption>
          <thead className="bg-card sticky top-0 z-10">
            <tr className="text-subtle-foreground border-border border-b text-left text-xs">
              {/* `w-full` on the first column and `w-px` on the numeric ones is
                  the auto-layout idiom for "these two take what they need, that
                  one absorbs the rest". Without it the browser splits the width
                  evenly-ish and the patient cell — the only one that *can* wrap
                  — is the one that gets squeezed. */}
              <SortHeader
                className="w-full px-4 py-2"
                label="환자"
                sortKey="patient_id"
                sort={sort}
                onSort={toggleSort}
              />
              <SortHeader
                className="w-px px-2 py-2 text-right whitespace-nowrap"
                label="위험도"
                sortKey="risk"
                sort={sort}
                onSort={toggleSort}
                align="right"
              />
              <SortHeader
                className="w-px px-4 py-2 text-right whitespace-nowrap"
                label="NEWS"
                sortKey="news_score"
                sort={sort}
                onSort={toggleSort}
                align="right"
              />
            </tr>
          </thead>
          <tbody ref={body}>
            {rows.map((patient) => {
              const selected = patient.patient_id === selectedId;
              const Trend =
                patient.risk_delta > 0.005 ? ArrowUp : patient.risk_delta < -0.005 ? ArrowDown : Minus;
              const disagrees = patient.model_alarm !== patient.news_alarm;

              return (
                <tr
                  key={patient.patient_id}
                  onClick={() => onSelect(patient.patient_id)}
                  onKeyDown={(event) => onKeyDown(event, patient.patient_id)}
                  tabIndex={patient.patient_id === focusable ? 0 : -1}
                  aria-current={selected || undefined}
                  className={cn(
                    "border-border cursor-pointer border-b transition",
                    "focus-visible:outline-primary focus-visible:-outline-offset-2 focus-visible:outline-2",
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
                    <span className="ml-2.5 flex items-center gap-1.5 pt-1">
                      <RiskBadge level={patient.risk_level} />
                      <Outcome hour={patient.arrest_hour} />
                    </span>
                  </th>
                  {/* The delta sits under the number it qualifies, not on the
                      identity line — and at 320px this is the only arrangement
                      where none of the three chips wraps. */}
                  <td className="w-px px-2 py-2 align-top text-right whitespace-nowrap">
                    <span className="font-medium tabular-nums">{formatRisk(patient.risk)}</span>
                    <span className="text-subtle-foreground flex items-center justify-end gap-0.5 pt-1 text-[11px] font-normal tabular-nums">
                      <Trend size={10} aria-hidden />
                      {formatDelta(patient.risk_delta)}
                    </span>
                  </td>
                  <td
                    className={cn(
                      "w-px px-4 py-2 align-top text-right whitespace-nowrap tabular-nums",
                      patient.news_alarm ? "font-medium" : "text-subtle-foreground",
                    )}
                  >
                    {patient.news_score.toFixed(0)}
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

/** The recorded outcome, never a prediction — hence the separate icon vocabulary. */
function Outcome({ hour }: { hour: number | null }) {
  const onset = hour != null;
  const color = onset ? "var(--status-critical)" : "var(--status-good)";

  return (
    <span
      className="inline-flex items-center gap-1 rounded-full px-1.5 py-0.5 text-[11px] font-medium whitespace-nowrap tabular-nums"
      style={{ color, backgroundColor: `color-mix(in oklab, ${color} 12%, transparent)` }}
      title={onset ? `실제 결과: ${hour.toFixed(0)}시간째 발병` : "실제 결과: 발병 없음"}
    >
      {onset ? <Biohazard size={10} aria-hidden /> : <ShieldCheck size={10} aria-hidden />}
      <span className="sr-only">실제 결과 </span>
      {onset ? `${hour.toFixed(0)}h` : "없음"}
    </span>
  );
}

function SortHeader({
  label,
  sortKey,
  sort,
  onSort,
  className,
  align = "left",
}: {
  label: string;
  sortKey: SortKey;
  sort: { key: SortKey; dir: SortDir };
  onSort: (key: SortKey) => void;
  className?: string;
  align?: "left" | "right";
}) {
  const active = sort.key === sortKey;
  const Arrow = sort.dir === "asc" ? ChevronUp : ChevronDown;

  return (
    <th
      scope="col"
      className={cn("font-medium", className)}
      aria-sort={active ? (sort.dir === "asc" ? "ascending" : "descending") : "none"}
    >
      <button
        type="button"
        onClick={() => onSort(sortKey)}
        className={cn(
          "focus-visible:outline-primary hover:text-foreground inline-flex cursor-pointer items-center gap-0.5 transition focus-visible:outline-2 focus-visible:outline-offset-2",
          active && "text-foreground",
          align === "right" && "flex-row-reverse",
        )}
      >
        {label}
        <Arrow size={12} className={cn(!active && "invisible")} aria-hidden />
      </button>
    </th>
  );
}
