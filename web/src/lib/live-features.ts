/**
 * Live bedside readings → the exact feature row the model was trained on.
 *
 * A line-for-line port of `src/vitals_live.py` + `vitals_data._window_features_from_arrays`.
 * The two must agree to the 4th decimal; `scripts/check_live_parity.mjs` runs
 * the Python export's parity cases through this file and fails on any drift.
 *
 * Imports are relative (no `@/`) so Node can run this file directly.
 */

export const VITALS = ["pulse", "sbp", "dbp", "temperature", "spo2", "resp_rate"] as const;
export type Vital = (typeof VITALS)[number];

const WINDOW_STATS = ["mean", "std", "min", "max", "last", "slope", "delta"] as const;

/** Readings outside these are sensor glitches (probe off the finger reads 0 or 127). */
export const PLAUSIBLE: Record<Vital, [number, number]> = {
  pulse: [20, 250],
  sbp: [40, 260],
  dbp: [20, 160],
  temperature: [30, 43],
  spo2: [50, 100],
  resp_rate: [2, 70],
};

export interface LiveReading {
  /** Seconds since the session started. */
  t: number;
  values: Partial<Record<Vital, number | null>>;
}

export type Binned = { step: number; values: Record<Vital, number> }[];

export function sanitize(value: unknown, vital: Vital): number {
  if (value === null || value === undefined) return NaN;
  const number = Number(value);
  if (!Number.isFinite(number)) return NaN;
  const [low, high] = PLAUSIBLE[vital];
  return number >= low && number <= high ? number : NaN;
}

/** Average readings into `stepSeconds` bins from the first reading; silent bins stay NaN. */
export function binReadings(readings: LiveReading[], stepSeconds: number): Binned {
  if (!(stepSeconds > 0)) throw new Error("stepSeconds must be positive");
  if (!readings.length) return [];
  const start = Math.min(...readings.map((r) => r.t));
  const sums = new Map<number, { sum: Record<string, number>; n: Record<string, number> }>();
  let maxStep = 0;
  for (const reading of readings) {
    const step = Math.floor((reading.t - start) / stepSeconds);
    maxStep = Math.max(maxStep, step);
    let acc = sums.get(step);
    if (!acc) {
      acc = { sum: {}, n: {} };
      sums.set(step, acc);
    }
    for (const vital of VITALS) {
      const value = sanitize(reading.values[vital], vital);
      if (Number.isFinite(value)) {
        acc.sum[vital] = (acc.sum[vital] ?? 0) + value;
        acc.n[vital] = (acc.n[vital] ?? 0) + 1;
      }
    }
  }
  const out: Binned = [];
  for (let step = 0; step <= maxStep; step++) {
    const acc = sums.get(step);
    const values = {} as Record<Vital, number>;
    for (const vital of VITALS) {
      values[vital] = acc && acc.n[vital] ? acc.sum[vital] / acc.n[vital] : NaN;
    }
    out.push({ step, values });
  }
  return out;
}

function leastSquaresSlope(x: number[], y: number[]): number {
  const n = x.length;
  const mx = x.reduce((a, b) => a + b, 0) / n;
  const my = y.reduce((a, b) => a + b, 0) / n;
  let num = 0;
  let den = 0;
  for (let i = 0; i < n; i++) {
    num += (x[i] - mx) * (y[i] - my);
    den += (x[i] - mx) ** 2;
  }
  return den === 0 ? 0 : num / den;
}

/** Port of `_window_features_from_arrays`: per-vital stats plus two derived features. */
export function windowFeatures(steps: number[], rows: Record<Vital, number>[]): Record<string, number> {
  const feats: Record<string, number> = {};
  for (const vital of VITALS) {
    const xs: number[] = [];
    const ys: number[] = [];
    rows.forEach((row, i) => {
      if (Number.isFinite(row[vital])) {
        xs.push(steps[i]);
        ys.push(row[vital]);
      }
    });
    if (!ys.length) {
      for (const stat of WINDOW_STATS) feats[`${vital}_${stat}`] = NaN;
      continue;
    }
    const mean = ys.reduce((a, b) => a + b, 0) / ys.length;
    feats[`${vital}_mean`] = mean;
    feats[`${vital}_std`] = Math.sqrt(ys.reduce((a, b) => a + (b - mean) ** 2, 0) / ys.length);
    feats[`${vital}_min`] = Math.min(...ys);
    feats[`${vital}_max`] = Math.max(...ys);
    feats[`${vital}_last`] = ys[ys.length - 1];
    feats[`${vital}_delta`] = ys[ys.length - 1] - ys[0];
    const span = Math.max(...xs) - Math.min(...xs);
    feats[`${vital}_slope`] = ys.length >= 2 && span > 0 ? leastSquaresSlope(xs, ys) : 0;
  }
  const pulse = feats.pulse_last;
  const sbp = feats.sbp_last;
  const dbp = feats.dbp_last;
  feats.shock_index_last = Number.isFinite(pulse) && Number.isFinite(sbp) && sbp !== 0 ? pulse / sbp : NaN;
  feats.pulse_pressure_last = Number.isFinite(sbp) && Number.isFinite(dbp) ? sbp - dbp : NaN;
  return feats;
}

export interface LiveFeatureResult {
  /** In the model's column order. */
  row: number[];
  named: Record<string, number>;
  baseline: Partial<Record<Vital, number>>;
  stepsObserved: number;
  baselineReady: boolean;
}

/** Feature row for the latest window, plus the session-opening baseline. */
export function liveFeatures(
  binned: Binned,
  columns: string[],
  windowSteps: number,
  baselineSteps: number,
): LiveFeatureResult {
  const last = binned.length ? binned[binned.length - 1].step : 0;
  const inWindow = binned.filter((b) => b.step > last - windowSteps);
  const named = windowFeatures(
    inWindow.map((b) => b.step),
    inWindow.map((b) => b.values),
  );

  const baseline: Partial<Record<Vital, number>> = {};
  const opening = binned.filter((b) => b.step < baselineSteps);
  for (const vital of VITALS) {
    const present = opening.map((b) => b.values[vital]).filter(Number.isFinite);
    if (present.length) baseline[vital] = present.reduce((a, b) => a + b, 0) / present.length;
  }
  for (const vital of VITALS) {
    for (const stat of ["last", "mean"] as const) {
      const deviation = named[`${vital}_${stat}`] - (baseline[vital] ?? NaN);
      // Same convention as training: no baseline yet means "no deviation".
      named[`${vital}_${stat}_dev`] = Number.isFinite(deviation) ? deviation : 0;
    }
  }

  const stepsObserved = binned.filter((b) => VITALS.some((v) => Number.isFinite(b.values[v]))).length;
  return {
    row: columns.map((name) => (name in named ? named[name] : NaN)),
    named,
    baseline,
    stepsObserved,
    baselineReady: stepsObserved >= baselineSteps,
  };
}

// --- NEWS (five vitals; consciousness and supplemental O2 are not measured) ---

const newsParts: Record<Exclude<Vital, "dbp">, (x: number) => number> = {
  resp_rate: (x) => (x <= 8 ? 3 : x <= 11 ? 1 : x <= 20 ? 0 : x <= 24 ? 2 : 3),
  spo2: (x) => (x <= 91 ? 3 : x <= 93 ? 2 : x <= 95 ? 1 : 0),
  temperature: (x) => (x <= 35 ? 3 : x <= 36 ? 1 : x <= 38 ? 0 : x <= 39 ? 1 : 2),
  sbp: (x) => (x <= 90 ? 3 : x <= 100 ? 2 : x <= 110 ? 1 : x <= 219 ? 0 : 3),
  pulse: (x) => (x <= 40 ? 3 : x <= 50 ? 1 : x <= 90 ? 0 : x <= 110 ? 1 : x <= 130 ? 2 : 3),
};

/** NEWS over the vitals actually measured; absent vitals score 0 (see `_live_news`). */
export function liveNews(named: Record<string, number>): { score: number; measured: number } {
  let score = 0;
  let measured = 0;
  for (const [vital, part] of Object.entries(newsParts)) {
    const value = named[`${vital}_last`];
    if (Number.isFinite(value)) {
      score += part(value);
      measured += 1;
    }
  }
  return { score, measured };
}
