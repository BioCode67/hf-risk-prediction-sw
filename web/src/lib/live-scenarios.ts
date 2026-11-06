/**
 * Built-in bedside scenarios, so the live page works with no device attached
 * and the presenter can show each behaviour on cue.
 *
 * Kept identical to `scenario()` in `scripts/export_live_model.py` so the
 * numbers on screen match what the export script printed.
 */

import type { Vital } from "./live-features";

export type ScenarioKind = "stable" | "worsening" | "chronic";

export const SCENARIOS: { kind: ScenarioKind; label: string; story: string }[] = [
  { kind: "stable", label: "안정 환자", story: "모든 수치가 정상 범위에서 흔들림만 있음 → 둘 다 경보 없음" },
  {
    kind: "worsening",
    label: "서서히 악화",
    story: "80초 뒤부터 맥박·호흡↑, 혈압·SpO2↓ → 개인 기저선 이탈이 커지며 경보",
  },
  {
    kind: "chronic",
    label: "평소 수치가 나쁜 만성 환자",
    story: "맥박 112·SpO2 92%가 이 환자의 평소 값 → NEWS는 경보, 모델은 '평소와 같음'으로 판단",
  },
];

const STABLE: Record<Vital, number> = { pulse: 78, sbp: 122, dbp: 76, temperature: 36.8, spo2: 98, resp_rate: 16 };
const CRASH: Record<Vital, number> = { pulse: 128, sbp: 82, dbp: 52, temperature: 36.2, spo2: 86, resp_rate: 30 };
const CHRONIC: Record<Vital, number> = { pulse: 112, sbp: 100, dbp: 62, temperature: 37.4, spo2: 92, resp_rate: 22 };

/** Box–Muller; scenario noise only, not a statistical model of anything. */
function gaussian(): number {
  const u = 1 - Math.random();
  const v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

export function scenarioReading(kind: ScenarioKind, t: number): Record<Vital, number> {
  const base = kind === "chronic" ? CHRONIC : STABLE;
  const fraction = kind === "worsening" ? Math.max(0, Math.min(1, (t - 80) / 140)) : 0;
  const out = {} as Record<Vital, number>;
  for (const vital of Object.keys(base) as Vital[]) {
    const value = base[vital] + fraction * (CRASH[vital] - base[vital]) + gaussian() * 0.4;
    out[vital] = Math.round(value * 10) / 10;
  }
  return out;
}
