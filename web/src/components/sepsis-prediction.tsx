"use client";

import { BellOff, BellRing, Loader2, Sparkles } from "lucide-react";

import { RiskBadge } from "@/components/RiskBadge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import type { LoadStatus } from "@/hooks/useDashboard";
import { RISK_META, formatRisk } from "@/lib/risk";
import type { Evidence, Narration } from "@/lib/types";

/**
 * The prediction card: probability, the SHAP drivers behind it, and an optional
 * plain-Korean paragraph.
 *
 * Three things the reference design showed are deliberately absent.
 * - A "model confidence" figure. The model emits one calibrated probability;
 *   a second number claiming confidence *in* that probability is not computed
 *   anywhere, and inventing it would be the most quotable false claim on screen.
 * - Treatment orders ("draw cultures", "give a fluid bolus"). Prescribing turns
 *   a research demo into something that reads as a medical device. What replaces
 *   it is a checklist of what to *look at*, derived from the actual top factors.
 * - A version string like "XGB v4.2". The model is whatever the artifact holds.
 */
export function SepsisPrediction({
  evidence,
  narration,
  narrationStatus,
  narrationError,
  llmAvailable,
  llmModel,
  onExplain,
}: {
  evidence: Evidence;
  narration: Narration | null;
  narrationStatus: LoadStatus;
  narrationError?: string;
  llmAvailable: boolean;
  llmModel: string | null;
  onExplain: () => void;
}) {
  const meta = RISK_META[evidence.risk_level];
  const strongest = Math.max(...evidence.top_factors.map((factor) => Math.abs(factor.shap)), 1e-9);

  return (
    <Card className="h-full">
      <CardHeader>
        <CardTitle>발병 위험도 예측</CardTitle>
        <CardDescription>
          이 창(과거 8시간)으로부터 향후 6시간 내 발병 확률. 코호트 임계값{" "}
          {formatRisk(evidence.threshold)}에서 경보합니다.
        </CardDescription>
      </CardHeader>

      <CardContent className="space-y-5">
        <div className="flex items-end justify-between gap-3">
          <div>
            <p className="text-5xl leading-none font-semibold tracking-tight" style={{ color: meta.color }}>
              {formatRisk(evidence.risk)}
            </p>
            <p className="text-subtle-foreground mt-1.5 text-xs tabular-nums">
              NEWS {evidence.news_score.toFixed(0)} · 기준 {evidence.news_threshold.toFixed(0)}
              {evidence.hours_to_arrest != null && ` · 발병 ${evidence.hours_to_arrest.toFixed(0)}시간 전`}
            </p>
          </div>
          <div className="flex flex-col items-end gap-1.5">
            <RiskBadge level={evidence.risk_level} size="lg" />
            <span className="text-muted-foreground inline-flex items-center gap-1 text-xs font-medium">
              {evidence.model_alarm ? (
                <BellRing size={13} style={{ color: "var(--critical)" }} aria-hidden />
              ) : (
                <BellOff size={13} aria-hidden />
              )}
              {evidence.model_alarm ? "경보" : "경보 없음"}
            </span>
          </div>
        </div>

        <div
          className="h-1.5 w-full overflow-hidden rounded-full"
          style={{ background: "var(--muted)" }}
          role="img"
          aria-label={`위험도 ${formatRisk(evidence.risk)}`}
        >
          <div
            className="h-full rounded-full"
            style={{ width: `${Math.min(100, evidence.risk * 100)}%`, background: meta.color }}
          />
        </div>

        <section>
          <h4 className="text-subtle-foreground mb-2 text-xs font-medium">기여 요인 (SHAP 상위 3)</h4>
          <ul className="space-y-2.5">
            {evidence.top_factors.map((factor) => {
              const color = factor.shap > 0 ? "var(--diverge-raise)" : "var(--diverge-lower)";
              return (
                <li key={factor.feature} className="space-y-1">
                  <div className="flex items-baseline justify-between gap-3">
                    <span className="text-sm leading-snug">{factor.description}</span>
                    <span className="text-muted-foreground shrink-0 text-xs tabular-nums">
                      {factor.value.toFixed(2)}
                      {factor.unit && ` ${factor.unit}`}
                    </span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="bg-muted h-1.5 flex-1 overflow-hidden rounded-full">
                      <div
                        className="h-full rounded-full"
                        style={{ width: `${(Math.abs(factor.shap) / strongest) * 100}%`, background: color }}
                      />
                    </div>
                    <span className="shrink-0 text-[11px] font-medium" style={{ color }}>
                      {factor.effect}
                    </span>
                  </div>
                </li>
              );
            })}
          </ul>
        </section>

        <section>
          <div className="mb-1.5 flex items-center justify-between gap-2">
            <h4 className="text-subtle-foreground text-xs font-medium">자연어 설명</h4>
            <Button
              variant="outline"
              size="sm"
              onClick={onExplain}
              disabled={!llmAvailable || narrationStatus === "loading"}
            >
              {narrationStatus === "loading" ? (
                <Loader2 size={13} className="animate-spin" aria-hidden />
              ) : (
                <Sparkles size={13} aria-hidden />
              )}
              {narration ? "다시 생성" : "생성"}
            </Button>
          </div>

          {!llmAvailable && (
            <p className="text-subtle-foreground text-xs">
              GROQ_API_KEY가 없어 비활성화됐습니다. 위 기여 요인만으로도 근거는 완결됩니다.
            </p>
          )}
          {/* Named before the call, not only after it. Hosted model names get
              retired on the provider's schedule, and when one does the failure
              is otherwise indistinguishable from "the button is broken". */}
          {llmAvailable && !narration && llmModel && (
            <p className="text-subtle-foreground text-[11px] tabular-nums">
              사용할 모델: {llmModel} · 바꾸려면 <code>GROQ_MODEL</code> 환경변수
            </p>
          )}
          {narrationError && (
            <p className="text-xs" style={{ color: "var(--critical)" }}>
              {narrationError}
            </p>
          )}
          {narration ? (
            <div className="space-y-1.5">
              <p className="text-sm leading-relaxed">{narration.text}</p>
              <p className="text-subtle-foreground text-[11px]">
                {narration.llm_model} · 위 수치만 쓰도록 제약된 문장이며, 판단은 모델이 했습니다.
              </p>
            </div>
          ) : (
            <details className="text-subtle-foreground text-xs">
              <summary className="cursor-pointer">계산된 근거 원문 보기</summary>
              <pre className="bg-muted mt-1.5 overflow-x-auto rounded-lg p-3 text-[11px] leading-relaxed whitespace-pre-wrap">
                {evidence.text}
              </pre>
            </details>
          )}
        </section>
      </CardContent>
    </Card>
  );
}
