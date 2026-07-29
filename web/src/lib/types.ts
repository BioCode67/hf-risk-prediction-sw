/**
 * Wire types for the FastAPI service in `src/vitals_api.py`.
 *
 * These mirror the pydantic response models one-for-one. If you change a model
 * there, change it here — the two files are the contract, and nothing else in
 * the app should reshape the payload.
 */

/** Reserved status bands. Never used for series identity. */
export type RiskLevel = "good" | "warning" | "serious" | "critical";

/** Which score fired when the two disagree — the false-alarm story. */
export type Disagreement = "none" | "model_only" | "news_only";

/** One vital at one hour, with the patient's own baseline for context. */
export interface VitalPoint {
  vital: string;
  label: string;
  unit: string;
  value: number | null;
  baseline: number | null;
}

/** A row in the ward list. */
export interface PatientSummary {
  patient_id: string;
  risk: number;
  risk_level: RiskLevel;
  risk_delta: number;
  model_alarm: boolean;
  news_score: number;
  news_alarm: boolean;
  hours_observed: number;
  last_hour: number;
  arrest_hour: number | null;
}

/** Model risk and NEWS at one window end. */
export interface TimelinePoint {
  hour: number;
  risk: number;
  news_score: number;
}

/** One SHAP driver, already described in Korean by the backend. */
export interface Factor {
  feature: string;
  description: string;
  unit: string;
  value: number;
  shap: number;
  effect: string;
}

/** Why the model did or did not alarm at one hour. */
export interface Evidence {
  patient_id: string;
  hour: number;
  risk: number;
  threshold: number;
  risk_level: RiskLevel;
  model_alarm: boolean;
  news_score: number;
  news_threshold: number;
  news_alarm: boolean;
  disagreement: Disagreement;
  top_factors: Factor[];
  vitals: VitalPoint[];
  hours_to_arrest: number | null;
  /** Deterministic rendering of the evidence — always present, no LLM involved. */
  text: string;
}

/** One hour of raw observed vitals. Keys beyond `hour` are vital names. */
export interface TrajectoryPoint {
  hour: number;
  [vital: string]: number | null;
}

/** Everything the detail view needs for one patient. */
export interface PatientDetail {
  patient_id: string;
  /** From the source record only — null when it does not carry the field. */
  age: number | null;
  sex: "M" | "F" | null;
  hours_observed: number;
  threshold: number;
  news_threshold: number;
  arrest_hour: number | null;
  timeline: TimelinePoint[];
  trajectory: TrajectoryPoint[];
  baseline: Record<string, number>;
  evidence: Evidence;
}

/** LLM-worded explanation. Optional everywhere — the app works without it. */
export interface Narration {
  text: string;
  llm_model: string;
}

/** Alarm burden for both scores at one matched sensitivity. */
export interface BurdenRow {
  sensitivity: number;
  model_alarms_per_100: number;
  model_specificity: number;
  news_alarms_per_100: number;
  news_specificity: number;
}

/** Cohort-level context shown above the ward list. */
export interface Overview {
  source: string;
  /** Held-out patients the figures below are measured on. */
  patients: number;
  windows: number;
  positive_rate: number;
  threshold: number;
  news_threshold: number;
  alarming: number;
  /** How many of those patients the ward list can actually open. */
  browsable: number;
  burden: BurdenRow[];
  llm_available: boolean;
}
