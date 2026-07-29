import { AlertTriangle, CheckCircle2, CircleAlert, Siren } from "lucide-react";
import type { LucideIcon } from "lucide-react";

import type { RiskLevel } from "./types";

/**
 * Status bands. Each carries an icon and a Korean label so the state is never
 * conveyed by colour alone — two of these four sit below 3:1 on the light
 * surface by design, and the icon + label pairing is what makes that legal.
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
