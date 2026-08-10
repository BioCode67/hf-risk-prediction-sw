import { AlertTriangle, CheckCircle2, CircleAlert, Siren } from "lucide-react";
import type { LucideIcon } from "lucide-react";

import type { RiskLevel } from "./types";

/**
 * Status bands. Each carries an icon and a Korean label so the state is never
 * conveyed by colour alone.
 *
 * These read the `--status-*` ramp, which is the *text* cut of the status
 * palette and is themed for legibility — not `--success`/`--critical`/…, which
 * are the chart fills and are far too light to set type in. The names must
 * exist in `globals.css`: an undeclared custom property makes the whole `color`
 * invalid at computed-value time, and that does not fail loudly — the badge
 * silently inherits body text and the status system stops meaning anything.
 */
export const RISK_META: Record<RiskLevel, { label: string; color: string; Icon: LucideIcon }> = {
  good: { label: "안정", color: "var(--status-good)", Icon: CheckCircle2 },
  warning: { label: "주의", color: "var(--status-warning)", Icon: CircleAlert },
  serious: { label: "경계", color: "var(--status-serious)", Icon: AlertTriangle },
  critical: { label: "경보", color: "var(--status-critical)", Icon: Siren },
};

/** Series identity, fixed. Slot 1 is always the model, slot 2 is always NEWS. */
export const SERIES = {
  model: { label: "본 모델", color: "var(--series-1)" },
  news: { label: "NEWS", color: "var(--series-2)" },
} as const;

export const formatRisk = (risk: number) => `${(risk * 100).toFixed(1)}%`;

/** Signed delta with an explicit sign, for "risk is climbing" cells. */
export const formatDelta = (delta: number) =>
  `${delta > 0 ? "+" : delta < 0 ? "−" : "±"}${Math.abs(delta * 100).toFixed(1)}%p`;
