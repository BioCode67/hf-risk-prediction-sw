import type { TimelinePoint } from "./types";

/** A stretch of consecutive hours during which a score stayed above its threshold. */
export interface AlarmRange {
  from: number;
  to: number;
}

/**
 * Collapse per-hour alarm decisions into contiguous ranges.
 *
 * Drawn as bands rather than one mark per hour: an alarm that persists for
 * twelve hours is one clinical event, and twelve vertical rules would read as
 * twelve. Ranges also survive gaps in the timeline — a window missing from the
 * middle breaks the band instead of silently bridging it.
 */
export function alarmRanges(timeline: TimelinePoint[], threshold: number, key: "risk" | "news_score" = "risk"): AlarmRange[] {
  const ranges: AlarmRange[] = [];
  let open: AlarmRange | null = null;
  let previousHour: number | null = null;

  for (const point of timeline) {
    const alarming = point[key] >= threshold;
    const contiguous = previousHour != null && point.hour === previousHour + 1;

    if (alarming && open && contiguous) {
      open.to = point.hour;
    } else if (alarming) {
      open = { from: point.hour, to: point.hour };
      ranges.push(open);
    } else {
      open = null;
    }
    previousHour = point.hour;
  }
  return ranges;
}
