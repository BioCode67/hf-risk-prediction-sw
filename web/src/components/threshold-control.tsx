"use client";

import { RotateCcw } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { SERIES, formatRisk } from "@/lib/risk";
import type { Eda } from "@/lib/types";

/**
 * The operating point, as a control rather than a fixed number.
 *
 * This is the project's argument made touchable. The model emits a probability;
 * where the alarm line sits was chosen after training and can be moved freely,
 * and moving it trades catches against noise on a curve that is steep in a way
 * a table of three rows cannot convey — the last few percent of detection cost
 * an order of magnitude in alarms.
 *
 * NEWS is shown at whatever detection rate the slider currently implies, so the
 * comparison stays matched at every position instead of only at the three
 * sensitivities the summary table reports.
 */
export function ThresholdControl({
  eda,
  threshold,
  defaultThreshold,
  positiveRate,
  onChange,
}: {
  eda: Eda;
  threshold: number;
  defaultThreshold: number;
  /** Share of windows that are actually pre-onset — the floor any alarm count sits on. */
  positiveRate: number;
  onChange: (threshold: number) => void;
}) {
  const point = nearest(eda.operating_points, threshold);
  // The clinical standard tuned to the same detection rate — the only fair
  // comparison, and the reason "fewer alarms" means anything here.
  const news = point ? matchSensitivity(eda.news_points, point.sensitivity) : null;

  const reduction =
    point && news && news.alarms_per_100 > 0
      ? 1 - point.alarms_per_100 / news.alarms_per_100
      : null;

  return (
    <Card>
      <CardHeader>
        <CardTitle>경보 임계값</CardTitle>
        <CardDescription>
          모델은 확률만 냅니다. 경보선을 어디에 둘지는 학습 이후의 운영 결정이라, 재학습 없이 바꿀 수
          있습니다. 옮기면 아래 지표와 모든 그래프의 경보 구간이 함께 움직입니다.
        </CardDescription>
      </CardHeader>

      <CardContent className="space-y-4">
        <div className="flex items-center gap-3">
          {/* The floor has to clear the exported operating point, which on an
              imbalanced cohort lands near 0.004. A hard `min` of 0.01 put the
              default below the track: the slider pinned to its minimum, showed
              a value it was not set to, and could never be dragged back. */}
          <input
            type="range"
            min={Math.min(0.01, defaultThreshold)}
            max={0.99}
            step="any"
            value={threshold}
            onChange={(event) => onChange(Number(event.target.value))}
            aria-label="경보 임계값"
            aria-valuetext={formatRisk(threshold)}
            className="accent-primary h-2 flex-1 cursor-pointer"
          />
          <span className="w-16 text-right text-lg font-semibold tabular-nums">
            {formatRisk(threshold)}
          </span>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => onChange(defaultThreshold)}
            disabled={Math.abs(threshold - defaultThreshold) < 0.0005}
            aria-label="기본 임계값으로 되돌리기"
          >
            <RotateCcw size={13} aria-hidden />
            기본값
          </Button>
        </div>

        {point ? (
          <>
            <dl className="grid grid-cols-2 gap-x-4 gap-y-3 sm:grid-cols-4">
              <Stat label="검출률 (민감도)" value={`${(point.sensitivity * 100).toFixed(0)}%`} />
              <Stat label="특이도" value={point.specificity.toFixed(3)} />
              <Stat
                label="알람 / 100 윈도우"
                value={point.alarms_per_100.toFixed(1)}
                color={SERIES.model.color}
              />
              <Stat
                label={`NEWS 동일 검출률`}
                value={news ? news.alarms_per_100.toFixed(1) : "—"}
                color={SERIES.news.color}
              />
            </dl>

            <p className="text-muted-foreground text-xs leading-relaxed">
              {reduction != null && reduction > 0 ? (
                <>
                  같은 검출률 {(point.sensitivity * 100).toFixed(0)}%에서 알람이{" "}
                  <strong className="text-foreground">{(reduction * 100).toFixed(0)}% 적습니다</strong>
                  {news && news.specificity < 0.05 && (
                    <span style={{ color: "var(--critical)" }}>
                      {" "}— 단 이 지점에서 NEWS 특이도가 {news.specificity.toFixed(3)}입니다. 사실상 전원
                      알람 상태라 비교가 성립하지 않습니다.
                    </span>
                  )}
                </>
              ) : (
                "이 지점에서는 NEWS 대비 이점이 없습니다."
              )}
            </p>

            {/* `nearest` snaps to the exported grid, which is 1%-spaced. At the
                default operating point — below 1% on an imbalanced cohort —
                the readout and the figures under it are therefore different
                thresholds, and saying so costs one line. */}
            {Math.abs(point.threshold - threshold) > 0.005 && (
              <p className="text-subtle-foreground text-[11px] leading-relaxed">
                위 지표는 {formatRisk(point.threshold)} 격자점에서 측정한 값입니다. 운영점 격자가 1%
                간격이라 슬라이더 값과 정확히 같지는 않습니다.
              </p>
            )}

            {/* The baseline the two alarm counts above sit on. Without it "2.0
                alarms per 100" reads as small, when the thing being caught
                occupies 1.2 of those 100 windows to begin with. */}
            <p className="text-subtle-foreground text-[11px] leading-relaxed">
              이 코호트의 양성률은 {(positiveRate * 100).toFixed(2)}%입니다 — 윈도우 100개 중{" "}
              {(positiveRate * 100).toFixed(1)}개가 실제 발병 직전 구간입니다. 위의 알람 수는 이 값을
              바닥에 두고 읽어야 합니다.
            </p>
          </>
        ) : (
          <p className="text-subtle-foreground text-sm">이 임계값의 지표를 계산할 수 없습니다.</p>
        )}
      </CardContent>
    </Card>
  );
}

function Stat({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div>
      <dt className="text-subtle-foreground text-[11px]">{label}</dt>
      <dd className="text-lg font-semibold tabular-nums" style={color ? { color } : undefined}>
        {value}
      </dd>
    </div>
  );
}

function nearest(points: Eda["operating_points"], threshold: number) {
  if (!points?.length) return null;
  return points.reduce((best, point) =>
    Math.abs(point.threshold - threshold) < Math.abs(best.threshold - threshold) ? point : best,
  );
}

/**
 * The NEWS cut whose detection rate is closest to the model's at this
 * threshold. NEWS is a 15-step integer score, so an exact match usually does
 * not exist — picking the nearest is the honest approximation, and its coarse
 * granularity is itself part of why it cannot be tuned finely.
 */
function matchSensitivity(points: Eda["news_points"], sensitivity: number) {
  if (!points?.length) return null;
  return points.reduce((best, point) =>
    Math.abs(point.sensitivity - sensitivity) < Math.abs(best.sensitivity - sensitivity) ? point : best,
  );
}
