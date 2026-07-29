"use client";

import { Activity, AlertCircle, Moon, Sun } from "lucide-react";
import { useEffect, useState } from "react";

import { AlarmNews } from "@/components/alarm-news";
import { PatientHeader } from "@/components/patient-header";
import { SepsisPrediction } from "@/components/sepsis-prediction";
import { VitalsChart } from "@/components/vitals-chart";
import { VitalsKpiCards } from "@/components/vitals-kpi-cards";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { useDashboard } from "@/hooks/useDashboard";

/**
 * Follows the reference layout: header → patient strip → KPI row → 3/2 split of
 * trends and prediction → full-width alarm comparison. All server state lives in
 * `useDashboard`; this file only arranges it.
 */
export default function Page() {
  const {
    overview,
    patients,
    detail,
    narration,
    selectedPatientId,
    selectedHour,
    status,
    errors,
    selectPatient,
    selectHour,
    explain,
  } = useDashboard(400);

  const evidence = detail?.evidence ?? null;
  const stale = status.detail === "loading" ? "is-stale" : undefined;

  return (
    <div className="bg-background min-h-svh">
      <div className="mx-auto flex max-w-[1600px] flex-col gap-4 p-4 md:gap-5 md:p-6 lg:p-8">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-2.5">
            <div className="bg-primary text-primary-foreground flex size-9 items-center justify-center rounded-xl shadow-sm">
              <Activity className="size-5" aria-hidden="true" />
            </div>
            <div className="leading-tight">
              <p className="text-sm font-semibold tracking-tight">활력징후 조기경보</p>
              <p className="text-subtle-foreground text-xs">
                개인 기저선 이탈 기반 · 설명가능 경보
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <SourceChip overview={overview} />
            <ThemeToggle />
          </div>
        </div>

        {errors.overview && <ErrorNote message={errors.overview} />}
        {errors.patients && <ErrorNote message={errors.patients} />}
        {errors.detail && <ErrorNote message={errors.detail} />}

        <PatientHeader
          patients={patients}
          detail={detail}
          selectedId={selectedPatientId}
          onSelect={selectPatient}
        />

        {detail && evidence ? (
          <div className={`flex flex-col gap-4 md:gap-5 ${stale ?? ""}`}>
            <VitalsKpiCards vitals={evidence.vitals} />

            <div className="grid grid-cols-1 gap-4 lg:grid-cols-5">
              <div className="lg:col-span-3">
                <VitalsChart
                  trajectory={detail.trajectory}
                  vitals={evidence.vitals}
                  selectedHour={evidence.hour}
                  eventHour={detail.arrest_hour}
                />
              </div>
              <div className="lg:col-span-2">
                <SepsisPrediction
                  evidence={evidence}
                  narration={narration}
                  narrationStatus={status.narration}
                  narrationError={errors.narration}
                  llmAvailable={overview?.llm_available ?? false}
                  onExplain={explain}
                />
              </div>
            </div>

            <AlarmNews
              detail={detail}
              burden={overview?.burden ?? []}
              selectedHour={selectedHour}
              onSelectHour={selectHour}
            />
          </div>
        ) : (
          <Card className="text-subtle-foreground flex h-64 items-center justify-center text-sm">
            {status.patients === "loading" ? "코호트를 불러오는 중…" : "표시할 환자가 없습니다."}
          </Card>
        )}

        <footer className="text-subtle-foreground border-border mt-2 border-t pt-3 text-xs leading-relaxed">
          연구·교육용 데모입니다. 의료기기가 아니며 임상 의사결정에 사용할 수 없습니다. 화면의 근거는 모델의
          기여도 분해이지 진단이 아니고, 처치 권고는 제공하지 않습니다. 데이터는 공개된 PhysioNet/CinC
          Challenge 2019(패혈증)이며 심정지가 아닙니다.
        </footer>
      </div>
    </div>
  );
}

/**
 * What the numbers are actually measured on. The reference design had a
 * "syncing every 5s" pulse here; nothing streams — the backend serves a
 * retrospective cohort fixed at build time — so it says what is true instead.
 */
function SourceChip({ overview }: { overview: ReturnType<typeof useDashboard>["overview"] }) {
  return (
    <span className="border-border bg-card text-muted-foreground inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-xs font-medium shadow-sm">
      <span aria-hidden className="bg-success size-2 rounded-full" />
      {overview
        ? `${overview.source} · 후향 코호트 ${overview.patients.toLocaleString()}명 · 열람 ${overview.browsable}명`
        : "연결 중…"}
    </span>
  );
}

function ThemeToggle() {
  const [theme, setTheme] = useState<"light" | "dark">("light");

  useEffect(() => {
    setTheme((document.documentElement.dataset.theme as "light" | "dark") ?? "light");
  }, []);

  const toggle = () => {
    const next = theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    localStorage.setItem("ews-theme", next);
    setTheme(next);
  };

  return (
    <Button variant="outline" size="sm" onClick={toggle} aria-label="명암 전환">
      {theme === "dark" ? <Sun size={14} aria-hidden /> : <Moon size={14} aria-hidden />}
    </Button>
  );
}

function ErrorNote({ message }: { message: string }) {
  return (
    <p
      className="flex items-start gap-2 rounded-lg p-3 text-xs leading-relaxed"
      style={{
        color: "var(--critical)",
        background: "color-mix(in oklab, var(--critical) 10%, transparent)",
      }}
    >
      <AlertCircle size={14} className="mt-0.5 shrink-0" aria-hidden />
      {message}
    </p>
  );
}
