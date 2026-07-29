"use client";

import { TriangleAlert } from "lucide-react";

import { redFlags } from "@/lib/severity";
import type { Evidence } from "@/lib/types";

/**
 * Shown when the vitals are in escalation territory and the model is not.
 *
 * The honest thing a screen can do about a known blind spot is name it at the
 * moment it applies, rather than let a green badge speak for a collapsing
 * patient. See lib/severity.ts for why this case exists at all.
 */
export function ModelScopeWarning({ evidence }: { evidence: Evidence }) {
  const flags = redFlags(evidence.vitals);
  if (evidence.model_alarm || flags.length === 0) return null;

  return (
    <div
      className="flex gap-3 rounded-xl border p-4"
      style={{
        borderColor: "var(--critical)",
        background: "color-mix(in oklab, var(--critical) 8%, transparent)",
      }}
      role="alert"
    >
      <TriangleAlert size={18} className="mt-0.5 shrink-0" style={{ color: "var(--critical)" }} aria-hidden />
      <div className="space-y-1.5">
        <p className="text-sm font-semibold" style={{ color: "var(--critical)" }}>
          낮은 위험도 = 안전 아님 — 활력징후가 이미 붕괴 수준입니다
        </p>
        <ul className="flex flex-wrap gap-x-3 gap-y-1 text-xs">
          {flags.map((flag) => (
            <li key={flag.vital} className="tabular-nums">
              <span className="font-medium">{flag.label}</span> {flag.value}
              {flag.unit && ` ${flag.unit}`}
              <span className="text-muted-foreground"> · {flag.reason}</span>
            </li>
          ))}
        </ul>
        <p className="text-muted-foreground text-xs leading-relaxed">
          이 모델이 답하는 질문은 <strong>&ldquo;앞으로 6시간 내 패혈증이 시작되는가&rdquo;</strong> 하나입니다.
          Challenge-2019에서 SpO2 70% 미만 윈도우 232개 중 발병은 0건 — 이미 붕괴한 환자는 발병 시점을
          지났거나 다른 원인으로 악화 중이라, 패혈증 발병 확률은 실제로 낮습니다. 점수는 맞지만
          &ldquo;안정&rdquo;이라는 뜻은 아닙니다.
        </p>
      </div>
    </div>
  );
}
