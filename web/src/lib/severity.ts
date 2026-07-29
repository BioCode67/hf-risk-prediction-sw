import type { VitalPoint } from "./types";

/**
 * Physiological red flags, read straight off the vitals — no model involved.
 *
 * The model answers one question well: will sepsis begin in the next six hours.
 * It is not a general deterioration monitor, and at the extremes the two come
 * apart. Across every Challenge-2019 window with SpO2 below 70%, the onset rate
 * is zero — patients that far gone are past onset or failing for another reason
 * — so catastrophic vitals genuinely lower the *sepsis* probability. The score
 * is right; a screen that renders it as "stable" is not.
 *
 * Hence this second, model-free channel. It never touches the prediction; it
 * just refuses to let a low sepsis score stand unqualified beside a collapsing
 * patient.
 *
 * The thresholds are deliberately far below the usual ward escalation triggers.
 * This is an ICU cohort where a respiratory rate of 28 or a systolic of 85 is
 * ordinary; the ward bands fired on 84% of patients, which is the alarm fatigue
 * this project exists to reduce. Tightened to genuine extremis, and requiring
 * either one critical value or two severe ones, it fires on about 7%.
 */

interface Flag {
  vital: string;
  label: string;
  value: number;
  unit: string;
  reason: string;
  tier: "critical" | "severe";
}

/** Any single one of these is enough — values seen in peri-arrest states. */
const CRITICAL: Record<string, (value: number) => string | null> = {
  spo2: (v) => (v < 80 ? "심한 저산소혈증" : null),
  sbp: (v) => (v < 75 ? "심한 저혈압" : null),
  pulse: (v) => (v < 40 ? "심한 서맥" : null),
  temperature: (v) => (v < 33 ? "심한 저체온" : null),
};

/** Two or more together. Individually common in an ICU; jointly, not. */
const SEVERE: Record<string, (value: number) => string | null> = {
  spo2: (v) => (v < 88 ? "저산소혈증" : null),
  sbp: (v) => (v < 85 ? "저혈압" : null),
  pulse: (v) => (v < 45 ? "서맥" : v > 150 ? "심한 빈맥" : null),
  resp_rate: (v) => (v > 35 ? "심한 빈호흡" : null),
};

function evaluate(vitals: VitalPoint[], rules: typeof CRITICAL, tier: Flag["tier"]): Flag[] {
  return vitals.flatMap((vital) => {
    if (vital.value == null) return [];
    const reason = rules[vital.vital]?.(vital.value);
    return reason
      ? [{ vital: vital.vital, label: vital.label, value: vital.value, unit: vital.unit, reason, tier }]
      : [];
  });
}

/**
 * Vitals in genuine extremis, whatever the model thinks. Empty unless the
 * one-critical-or-two-severe bar is met, so the caller can treat any non-empty
 * result as worth interrupting for.
 */
export function redFlags(vitals: VitalPoint[]): Flag[] {
  const critical = evaluate(vitals, CRITICAL, "critical");
  if (critical.length > 0) {
    // Report the severe readings alongside, so the picture is complete.
    const severe = evaluate(vitals, SEVERE, "severe").filter(
      (flag) => !critical.some((c) => c.vital === flag.vital),
    );
    return [...critical, ...severe];
  }

  const severe = evaluate(vitals, SEVERE, "severe");
  return severe.length >= 2 ? severe : [];
}
