import type { Disagreement, Evidence, RiskLevel } from "./types";

/**
 * Alarm decisions, derived from the threshold the user currently has set.
 *
 * The snapshot bakes `model_alarm`, `risk_level` and `disagreement` at the
 * threshold that was in force when it was exported. Once the threshold is a
 * control on screen those fields are stale the moment it moves, so nothing in
 * the UI may read them directly — the probability is the durable number, and
 * every decision is recomputed from it here.
 *
 * The threshold was never part of the model. It is an operating point chosen
 * after training, which is exactly why moving it is legitimate: no retraining,
 * no new data, just a different answer to "how much noise is a catch worth".
 */

/** Risk bands. "critical" starts at the operating threshold, so colour and alarm never disagree. */
const BAND_WARNING = 0.25;
const BAND_SERIOUS = 0.5;

export function riskLevel(risk: number, threshold: number): RiskLevel {
  if (risk >= Math.max(threshold, BAND_SERIOUS)) return "critical";
  if (risk >= BAND_SERIOUS) return "serious";
  if (risk >= BAND_WARNING) return "warning";
  return "good";
}

export interface Decision {
  modelAlarm: boolean;
  newsAlarm: boolean;
  level: RiskLevel;
  disagreement: Disagreement;
}

export function decide(risk: number, newsScore: number, threshold: number, newsThreshold: number): Decision {
  const modelAlarm = risk >= threshold;
  const newsAlarm = newsScore >= newsThreshold;
  return {
    modelAlarm,
    newsAlarm,
    level: riskLevel(risk, threshold),
    disagreement: modelAlarm === newsAlarm ? "none" : modelAlarm ? "model_only" : "news_only",
  };
}

/** An evidence packet with its decision fields recomputed at `threshold`. */
export function reevaluate(evidence: Evidence, threshold: number): Evidence {
  const decision = decide(evidence.risk, evidence.news_score, threshold, evidence.news_threshold);
  return {
    ...evidence,
    threshold,
    model_alarm: decision.modelAlarm,
    news_alarm: decision.newsAlarm,
    risk_level: decision.level,
    disagreement: decision.disagreement,
  };
}
