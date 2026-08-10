/**
 * The view state that belongs in the address bar.
 *
 * Which patient, which hour, and where the alarm line sits are the three things
 * that make one screen different from another. Keeping them in the query string
 * is what turns "look at patient 184 an hour before onset, with the threshold at
 * 30%" into something you can send to someone or come back to after a reload —
 * which for a demo that gets presented is most of its use.
 *
 * Deliberately `replaceState`, not `pushState`: dragging a slider would
 * otherwise stack a hundred history entries and make Back useless.
 *
 * Read/write only. Nothing here validates that the patient exists — the fetch
 * does that, and reports it through the normal error path.
 */

export interface ViewState {
  patientId: string | null;
  hour: number | null;
  threshold: number | null;
}

const EMPTY: ViewState = { patientId: null, hour: null, threshold: null };

const number = (raw: string | null): number | null => {
  if (raw == null) return null;
  const value = Number(raw);
  return Number.isFinite(value) ? value : null;
};

/** Parse the current URL. Returns all-null during SSR, where there is no URL. */
export function readView(): ViewState {
  if (typeof window === "undefined") return EMPTY;
  const params = new URLSearchParams(window.location.search);
  const threshold = number(params.get("t"));
  return {
    patientId: params.get("p"),
    hour: number(params.get("h")),
    // Out-of-range values would move the slider somewhere it cannot return
    // from, so an unusable threshold is dropped rather than clamped.
    threshold: threshold != null && threshold > 0 && threshold < 1 ? threshold : null,
  };
}

/** Mirror the view into the URL, leaving the rest of the query string alone. */
export function writeView(view: ViewState): void {
  if (typeof window === "undefined") return;

  const params = new URLSearchParams(window.location.search);
  const set = (key: string, value: string | null) =>
    value == null ? params.delete(key) : params.set(key, value);

  set("p", view.patientId);
  set("h", view.hour == null ? null : String(view.hour));
  // Four decimals, trailing zeros dropped. Two is not enough: on an imbalanced
  // cohort the exported operating point sits around 0.004, which `toFixed(2)`
  // flattens to "0.00" — a value `readView` then rejects as out of range, so
  // the link silently drops the very threshold it was meant to carry.
  set("t", view.threshold == null ? null : String(Number(view.threshold.toFixed(4))));

  const query = params.toString();
  const next = `${window.location.pathname}${query ? `?${query}` : ""}`;
  if (next !== `${window.location.pathname}${window.location.search}`) {
    window.history.replaceState(null, "", next);
  }
}
