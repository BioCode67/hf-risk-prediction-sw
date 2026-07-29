/**
 * Data access for the dashboard.
 *
 * The cohort analysis is a *snapshot*, baked by `scripts/export_dashboard.py`
 * into `public/data`, so reading it is a static file fetch — no backend, no ML
 * runtime, nothing to keep warm. Only the LLM wording needs a server, and that
 * is one Next route handler.
 *
 * Re-run the export whenever the model or threshold changes; nothing here picks
 * that up on its own.
 */

import type {
  Evidence,
  Narration,
  Overview,
  PatientDetail,
  PatientSummary,
  TimelinePoint,
  TrajectoryPoint,
} from "./types";

/** Where the baked snapshot lives. Same origin, so no CORS anywhere. */
const DATA_BASE = "/data";

/** An HTTP failure carrying the status, so callers can tell 403 from 502. */
export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

/** The shape `export_dashboard.py` writes for one patient. */
interface PatientFile extends Omit<PatientDetail, "evidence"> {
  /** Every window end the user can click, keyed by hour. */
  evidence: Record<string, Evidence>;
  timeline: TimelinePoint[];
  trajectory: TrajectoryPoint[];
}

/** The shape it writes for the cohort. */
interface IndexFile {
  source: string;
  threshold: number;
  news_threshold: number;
  cohort: { patients: number; windows: number; positive_rate: number; alarming: number };
  browsable: number;
  burden: Overview["burden"];
  patients: PatientSummary[];
}

async function getJson<T>(path: string, signal?: AbortSignal): Promise<T> {
  let response: Response;
  try {
    response = await fetch(path, { signal });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") throw error;
    throw new ApiError(0, `데이터를 불러오지 못했습니다 (${path}).`);
  }
  if (!response.ok) {
    throw new ApiError(
      response.status,
      response.status === 404
        ? `데이터가 없습니다 (${path}). scripts/export_dashboard.py 를 실행했는지 확인하세요.`
        : `${response.status} ${response.statusText}`,
    );
  }
  return (await response.json()) as T;
}

let indexCache: Promise<IndexFile> | null = null;

/** The index backs both the overview and the patient list; fetch it once. */
function loadIndex(signal?: AbortSignal): Promise<IndexFile> {
  indexCache ??= getJson<IndexFile>(`${DATA_BASE}/index.json`, signal).catch((error) => {
    indexCache = null; // let a later mount retry
    throw error;
  });
  return indexCache;
}

export const api = {
  overview: async (signal?: AbortSignal): Promise<Overview> => {
    const index = await loadIndex(signal);
    return {
      source: index.source,
      patients: index.cohort.patients,
      windows: index.cohort.windows,
      positive_rate: index.cohort.positive_rate,
      threshold: index.threshold,
      news_threshold: index.news_threshold,
      alarming: index.cohort.alarming,
      browsable: index.browsable,
      burden: index.burden,
      // Resolved by the route handler, which alone knows whether a key is set.
      llm_available: await hasLlm(signal),
    };
  },

  patients: async (limit = 400, signal?: AbortSignal): Promise<PatientSummary[]> =>
    (await loadIndex(signal)).patients.slice(0, limit),

  /** Omit `hour` for the patient's most recent window. */
  patient: async (
    patientId: string,
    hour?: number | null,
    signal?: AbortSignal,
  ): Promise<PatientDetail> => {
    const file = await getJson<PatientFile>(
      `${DATA_BASE}/patients/${encodeURIComponent(patientId)}.json`,
      signal,
    );

    const hours = file.timeline.map((point) => point.hour);
    const wanted = hour ?? hours.at(-1);
    const evidence = file.evidence[String(wanted)];
    if (!evidence) {
      throw new ApiError(404, `${wanted}시간째 윈도우가 없습니다.`);
    }
    return { ...file, evidence };
  },

  explain: async (
    evidence: Evidence,
    signal?: AbortSignal,
  ): Promise<Narration> => {
    const response = await fetch("/api/explain", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      // Only the already-computed evidence block travels; the model decided
      // everything before this call, and the LLM may only reword these numbers.
      body: JSON.stringify({ text: evidence.text }),
      signal,
    });
    if (!response.ok) {
      const detail = await response
        .json()
        .then((body: { detail?: string }) => body.detail)
        .catch(() => undefined);
      throw new ApiError(response.status, detail ?? `${response.status} ${response.statusText}`);
    }
    return (await response.json()) as Narration;
  },
};

let llmCache: Promise<boolean> | null = null;

/** Whether the deployment has a Groq key, so the UI can disable the button. */
function hasLlm(signal?: AbortSignal): Promise<boolean> {
  llmCache ??= fetch("/api/explain", { method: "GET", signal })
    .then((response) => (response.ok ? response.json() : { available: false }))
    .then((body: { available?: boolean }) => Boolean(body.available))
    .catch(() => false);
  return llmCache;
}
