"""Score a live stream of bedside readings with the dashboard's model.

The dashboard's ``/live`` page reads consumer vital-sign devices in the browser
(Web Bluetooth) or a built-in scenario, and posts the raw readings here. This
module turns those readings into exactly the feature row the model was trained
on and returns risk, NEWS and the SHAP evidence for the latest window.

Two honest simplifications, both surfaced on screen:

* **Time is compressed.** The trained model's step is one hour. A demo cannot
  wait eight hours, so readings are binned into ``step_seconds`` bins and each
  bin plays the role of one step. Trends and slopes are therefore per *step*.
* **The personal baseline is this session's opening.** The method compares a
  patient against their own early, stable period; live, that is the mean of the
  first ``baseline_steps`` bins of the session.

Nothing here stores readings. Each request is scored and forgotten.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from vitals_data import (
    OBSERVATION_WINDOW_HOURS,
    PERSONAL_BASELINE_HOURS,
    VITALS,
    _window_features_from_arrays,
)

# Physiologically possible ranges. A reading outside is a sensor glitch (a
# pulse oximeter sliding off a finger reads 0 or 127), not a patient state.
PLAUSIBLE: dict[str, tuple[float, float]] = {
    "pulse": (20.0, 250.0),
    "sbp": (40.0, 260.0),
    "dbp": (20.0, 160.0),
    "temperature": (30.0, 43.0),
    "spo2": (50.0, 100.0),
    "resp_rate": (2.0, 70.0),
}


@dataclass(frozen=True)
class LiveReading:
    """One measurement; any vital may be missing."""

    t: float  # seconds since the session started
    values: dict[str, float | None]


def sanitize(value: Any, vital: str) -> float:
    """Return the value as float, or NaN if absent or physiologically impossible."""
    if value is None:
        return float("nan")
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    low, high = PLAUSIBLE[vital]
    return number if low <= number <= high else float("nan")


def bin_readings(readings: list[LiveReading], step_seconds: float) -> pd.DataFrame:
    """Average readings into consecutive bins; one row per bin, NaN where nothing arrived.

    Bins are indexed from the first reading, so a session that starts late is
    not padded with empty steps.
    """
    if step_seconds <= 0:
        raise ValueError("step_seconds must be positive")
    if not readings:
        return pd.DataFrame(columns=["step", *VITALS])

    start = min(reading.t for reading in readings)
    records = []
    for reading in readings:
        record = {"step": int((reading.t - start) // step_seconds)}
        for vital in VITALS:
            record[vital] = sanitize(reading.values.get(vital), vital)
        records.append(record)

    frame = pd.DataFrame.from_records(records)
    binned = frame.groupby("step")[list(VITALS)].mean()
    # Re-index so a silent stretch shows up as missing steps, not as no time passing.
    binned = binned.reindex(range(int(binned.index.max()) + 1))
    binned.index.name = "step"
    return binned.reset_index()


def live_features(
    binned: pd.DataFrame,
    columns: list[str],
    window_steps: int = OBSERVATION_WINDOW_HOURS,
    baseline_steps: int = PERSONAL_BASELINE_HOURS,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Feature row for the latest window, in the model's column order.

    Returns ``(row, baseline)``. ``baseline`` maps each vital to the session's
    opening mean (absent vitals are omitted).
    """
    steps = binned["step"].to_numpy(dtype=float)
    values = binned[list(VITALS)].to_numpy(dtype=float)

    last = int(steps.max()) if len(steps) else 0
    in_window = steps > last - window_steps
    features = _window_features_from_arrays(steps[in_window], values[in_window])

    opening = binned[binned["step"] < baseline_steps][list(VITALS)].mean()
    baseline = {vital: float(opening[vital]) for vital in VITALS if np.isfinite(opening[vital])}
    for vital in VITALS:
        for stat in ("last", "mean"):
            current = features.get(f"{vital}_{stat}", float("nan"))
            ref = baseline.get(vital, float("nan"))
            deviation = current - ref
            # Same convention as training: no baseline yet means "no deviation".
            features[f"{vital}_{stat}_dev"] = float(deviation) if np.isfinite(deviation) else 0.0

    row = pd.DataFrame([{name: features.get(name, float("nan")) for name in columns}], columns=columns)
    return row, baseline


def ready_state(binned: pd.DataFrame, baseline_steps: int = PERSONAL_BASELINE_HOURS) -> dict[str, Any]:
    """How far the session is from a trustworthy score.

    A score from two readings is noise; the screen says "measuring the
    baseline" until the opening period is covered, instead of showing a number
    nobody should act on.
    """
    observed = int(binned[list(VITALS)].notna().any(axis=1).sum()) if len(binned) else 0
    return {
        "steps_observed": observed,
        "baseline_steps": baseline_steps,
        "baseline_ready": observed >= baseline_steps,
    }
