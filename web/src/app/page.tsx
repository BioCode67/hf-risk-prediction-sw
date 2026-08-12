"use client";

import { Activity, AlertCircle, Moon, Sun } from "lucide-react";
import { useEffect, useState } from "react";

import { AlarmNews } from "@/components/alarm-news";
import { CohortEda } from "@/components/cohort-eda";
import { ModelScopeWarning } from "@/components/model-scope-warning";
import { PatientHeader } from "@/components/patient-header";
import { PatientList } from "@/components/patient-list";
import { SepsisPrediction } from "@/components/sepsis-prediction";
import { ThresholdControl } from "@/components/threshold-control";
import { VitalsChart } from "@/components/vitals-chart";
import { VitalsKpiCards } from "@/components/vitals-kpi-cards";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { useDashboard } from "@/hooks/useDashboard";
import { decide, reevaluate } from "@/lib/decision";

/**
 * Ward list on the left, one patient's story on the right.
 *
 * The list is the anchor: this is a ward screen, and picking who to look at is
 * the first decision. The right column then reads top-down the way a clinician
 * would ask — who is this, what are they doing now, what is the trend, what
 * does the model think, and how does that compare to the standard score.
 *
 * All server state lives in `useDashboard`; this file only arranges it.
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
    threshold,
    selectPatient,
    selectHour,
    setThreshold,
    explain,
  } = useDashboard(400);

  // The baked decisions are only valid at the exported threshold. Once the
  // slider exists they must be recomputed, or the badges and the alarm bands
  // would disagree with the control the user just moved.
  const cut = threshold ?? overview?.threshold ?? detail?.threshold ?? 0.5;
  const evidence = detail ? reevaluate(detail.evidence, cut) : null;
  const rows = patients.map((patient) => {
    const derived = decide(patient.risk, patient.news_score, cut, overview?.news_threshold ?? 5);
    return {
      ...patient,
      risk_level: derived.level,
      model_alarm: derived.modelAlarm,
      news_alarm: derived.newsAlarm,
    };
  });
  const stale = status.detail === "loading";

  return (
    <div className="bg-background min-h-svh">
      <div className="mx-auto flex max-w-[1700px] flex-col gap-4 p-4 md:gap-5 md:p-6 lg:p-8">
        <Header overview={overview} />

        {errors.overview && <ErrorNote message={errors.overview} />}
        {errors.patients && <ErrorNote message={errors.patients} />}

        <div className="grid grid-cols-1 gap-4 xl:grid-cols-[320px_minmax(0,1fr)] xl:items-start">
          <Card className="overflow-hidden p-0 xl:sticky xl:top-6 xl:max-h-[calc(100svh-6rem)]">
            <CardHeader className="pb-2">
              <CardTitle>병동 환자</CardTitle>
              <CardDescription>
                {overview
                  ? `열람 ${overview.browsable}명 · 코호트 ${overview.patients.toLocaleString()}명 기준`
                  : "불러오는 중…"}
              </CardDescription>
            </CardHeader>
            <div className="flex min-h-0 flex-1 flex-col xl:h-[calc(100svh-11rem)]">
              <PatientList
                patients={rows}
                selectedId={selectedPatientId}
                onSelect={selectPatient}
                stale={status.patients === "loading"}
              />
            </div>
          </Card>

          <div className="flex min-w-0 flex-col gap-4 md:gap-5">
            {errors.detail && <ErrorNote message={errors.detail} />}

            <PatientHeader
              detail={detail && evidence ? { ...detail, evidence, threshold: cut } : null}
              summary={rows.find((p) => p.patient_id === selectedPatientId) ?? null}
            />

            {detail && evidence ? (
              <div className={`flex flex-col gap-4 md:gap-5 ${stale ? "is-stale" : ""}`}>
                <ModelScopeWarning evidence={evidence} />

                <VitalsKpiCards vitals={evidence.vitals} />

                <div className="grid grid-cols-1 gap-4 lg:grid-cols-5">
                  <div className="lg:col-span-3">
                    <VitalsChart
                      trajectory={detail.trajectory}
                      vitals={evidence.vitals}
                      timeline={detail.timeline}
                      threshold={cut}
                      normalBand={overview?.normal_band ?? {}}
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
                  detail={{ ...detail, evidence, threshold: cut }}
                  burden={overview?.burden ?? []}
                  selectedHour={selectedHour}
                  onSelectHour={selectHour}
                />

                {overview && (
                  <ThresholdControl
                    eda={overview.eda}
                    threshold={cut}
                    defaultThreshold={overview.threshold}
                    onChange={setThreshold}
                  />
                )}

                {overview && (
                  <CohortEda
                    eda={overview.eda}
                    normalBand={overview.normal_band}
                    vitalLabels={Object.fromEntries(
                      evidence.vitals.map((vital) => [vital.vital, { label: vital.label, unit: vital.unit }]),
                    )}
                  />
                )}
              </div>
            ) : (
              <Card className="text-subtle-foreground flex h-64 items-center justify-center text-sm">
                {status.patients === "loading" ? "코호트를 불러오는 중…" : "왼쪽에서 환자를 고르세요."}
              </Card>
            )}
          </div>
        </div>

        <Footer source={overview?.source} />
      </div>
    </div>
  );
}

function Header({ overview }: { overview: ReturnType<typeof useDashboard>["overview"] }) {
  return (
    <header className="flex flex-wrap items-center justify-between gap-3">
      <div className="flex items-center gap-2.5">
        <div className="bg-primary text-primary-foreground flex size-9 items-center justify-center rounded-xl shadow-sm">
          <Activity className="size-5" aria-hidden />
        </div>
        <div className="leading-tight">
          <h1 className="text-sm font-semibold tracking-tight">활력징후 조기경보</h1>
          <p className="text-subtle-foreground text-xs">개인 기저선 이탈 기반 · 설명가능 경보</p>
        </div>
      </div>

      <div className="flex items-center gap-2">
        {/* No "syncing" pulse: the cohort is retrospective and fixed. */}
        <span className="border-border bg-card text-muted-foreground inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-xs font-medium shadow-sm">
          <span aria-hidden className="bg-success size-2 rounded-full" />
          {overview
            ? `${overview.source} · 후향 코호트 · 윈도우 ${overview.windows.toLocaleString()}개`
            : "연결 중…"}
        </span>
        <ThemeToggle />
      </div>
    </header>
  );
}

function Footer({ source }: { source?: string }) {
  return (
    <footer className="text-subtle-foreground border-border space-y-1.5 border-t pt-3 text-xs leading-relaxed">
      <p>
        연구·교육용 데모입니다. 의료기기가 아니며 임상 의사결정에 사용할 수 없습니다. 화면의 근거는 모델의
        기여도 분해이지 진단이 아니고, 처치 권고는 제공하지 않습니다.
      </p>
      {source === "synthetic" ? (
        <p>
          지금 화면의 데이터는 <strong>합성 코호트</strong>입니다 — 실제 환자 기록이 아니라
          파이프라인 시연용으로 생성된 값이며, 대상 사건은 <strong>병원 내 심정지</strong>입니다.
          여기의 수치는 성능 근거가 아니라 화면 동작 예시로만 보십시오.
        </p>
      ) : (
        <p>
          데이터는 공개된 PhysioNet/CinC Challenge 2019이며 대상 사건은 <strong>패혈증 발병</strong>입니다
          (심정지가 아닙니다). 모델이 학습한 라벨은 &ldquo;악화&rdquo;가 아니라 &ldquo;6시간 내 발병&rdquo;이므로,
          이미 붕괴 중인 환자에게 낮은 점수가 나올 수 있습니다 — 그 경우 화면 상단에 경고가 표시됩니다.
        </p>
      )}
    </footer>
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
