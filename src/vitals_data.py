"""Vital-sign time-series data for cardiac-arrest early warning.

This module builds the *time-series* half of the project (distinct from the
static heart-failure pipeline in ``data_loader.py``). It turns hourly vital-sign
trajectories into sliding-window statistical features and labels each window by
whether a cardiac arrest occurs within a short prediction horizon.

Design notes grounded in the 2026 K-Health competition (경북대병원 활력징후):

- The competition cohort (``KHTH_PINFO`` / ``KHTH_VITAL``) contains only patients
  who suffered an in-hospital cardiac arrest and died — there are **no control
  patients**. We therefore use a *within-patient* labelling scheme: early windows
  of a patient are negatives, windows just before the arrest are positives. This
  is a "case-only" temporal design; see ``build_windows``.
- The analysis has to run inside an offline 안심존 (no internet, pre-declared
  packages only), so this pipeline stays deliberately lightweight (numpy/pandas +
  gradient boosting) and ships a synthetic generator so it runs end-to-end with
  no restricted data. Real data plugs in via ``cohort_from_khth``.

Vitals modelled (canonical names): pulse (맥박), sbp (수축기), dbp (이완기),
temperature (체온), spo2 (산소포화도), resp_rate (호흡수).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# --- Configuration --------------------------------------------------------

VITALS: Final[tuple[str, ...]] = (
    "pulse",
    "sbp",
    "dbp",
    "temperature",
    "spo2",
    "resp_rate",
)
WINDOW_STATS: Final[tuple[str, ...]] = (
    "mean",
    "std",
    "min",
    "max",
    "last",
    "slope",
    "delta",
)
DERIVED_FEATURES: Final[tuple[str, ...]] = ("shock_index_last", "pulse_pressure_last")

# Sliding-window defaults: observe the past N hours, predict an arrest within the
# next H hours. Step slides the window one hour at a time.
OBSERVATION_WINDOW_HOURS: Final[int] = 8
PREDICTION_HORIZON_HOURS: Final[int] = 1
GAP_HOURS: Final[int] = 0
STEP_HOURS: Final[int] = 1

# Synthetic physiology (development only — NOT real patient data).
# (mean, sd) for each patient's stable baseline.
_HEALTHY_BASELINE: Final[dict[str, tuple[float, float]]] = {
    "pulse": (78.0, 8.0),
    "sbp": (122.0, 12.0),
    "dbp": (76.0, 8.0),
    "temperature": (36.8, 0.3),
    "spo2": (97.5, 1.0),
    "resp_rate": (16.0, 2.0),
}
# Total shift applied at the moment of arrest (deterioration trajectory).
_DETERIORATION: Final[dict[str, float]] = {
    "pulse": 45.0,
    "sbp": -38.0,
    "dbp": -22.0,
    "temperature": -0.6,
    "spo2": -14.0,
    "resp_rate": 16.0,
}
_NOISE_SD: Final[dict[str, float]] = {
    "pulse": 4.0,
    "sbp": 6.0,
    "dbp": 4.0,
    "temperature": 0.2,
    "spo2": 0.6,
    "resp_rate": 1.2,
}
_CLIP_RANGE: Final[dict[str, tuple[float, float]]] = {
    "pulse": (30.0, 200.0),
    "sbp": (50.0, 240.0),
    "dbp": (30.0, 140.0),
    "temperature": (33.0, 42.0),
    "spo2": (60.0, 100.0),
    "resp_rate": (4.0, 50.0),
}


def feature_names() -> list[str]:
    """Return the ordered feature-column names produced per window."""
    names = [f"{vital}_{stat}" for vital in VITALS for stat in WINDOW_STATS]
    names += list(DERIVED_FEATURES)
    return names


@dataclass
class VitalSignsCohort:
    """A cohort of hourly vital-sign trajectories plus per-patient arrest times.

    Attributes:
        vitals: Long-per-hour frame with columns ``patient_id``, ``hour`` and one
            column per entry in :data:`VITALS`.
        events: One row per patient with ``patient_id`` and ``arrest_hour``
            (``NaN`` for patients who never arrest — i.e. controls).
        demographics: Optional per-patient static frame with ``patient_id``,
            ``age`` and ``sex`` (1=male, 0=female); ``None`` when unavailable.
    """

    vitals: pd.DataFrame
    events: pd.DataFrame
    demographics: pd.DataFrame | None = None


@dataclass
class WindowedDataset:
    """Sliding-window feature matrix, labels and patient grouping."""

    features: pd.DataFrame
    labels: pd.Series
    groups: pd.Series
    feature_names: list[str]
    # Hours from each window's end to the arrest (NaN for control windows);
    # used for lead-time analysis. Optional for backward compatibility.
    time_to_arrest: pd.Series | None = None


@dataclass
class PatientSplit:
    """Patient-level train/test split (no patient appears in both sets)."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    groups_train: np.ndarray
    groups_test: np.ndarray
    feature_names: list[str]
    imputation_medians: pd.Series
    # Test-set hours-to-arrest per window (aligned with X_test); for lead-time.
    time_to_arrest_test: np.ndarray | None = None


# --- Synthetic cohort -----------------------------------------------------


def generate_synthetic_cohort(
    n_patients: int = 500,
    arrest_fraction: float = 0.5,
    seed: int = 42,
    min_stay_hours: int = 24,
    max_stay_hours: int = 96,
    deterioration_ramp_hours: int = 6,
    missing_rate: float = 0.1,
) -> VitalSignsCohort:
    """Generate a synthetic vital-sign cohort for development and testing.

    This is **not** real patient data. Arrest patients follow a monotonic
    deterioration trajectory (rising pulse/resp_rate, falling spo2/blood
    pressure) over the ``deterioration_ramp_hours`` before the event, with
    physiological noise and random missingness.

    Set ``arrest_fraction=1.0`` to mirror the competition's *case-only* cohort
    (every patient arrests); lower values add control patients, which is useful
    for false-alarm estimation during development with external data.

    Args:
        n_patients: Number of synthetic patients.
        arrest_fraction: Fraction of patients who arrest (1.0 = case-only).
        seed: RNG seed for reproducibility.
        min_stay_hours: Minimum recorded hours per patient.
        max_stay_hours: Maximum recorded hours per patient.
        deterioration_ramp_hours: Hours of deterioration before arrest.
        missing_rate: Per-measurement probability of a missing value.

    Returns:
        A :class:`VitalSignsCohort`.
    """
    rng = np.random.default_rng(seed)
    vital_rows: list[dict[str, float]] = []
    event_rows: list[dict[str, float]] = []
    demo_rows: list[dict[str, float]] = []
    # Earliest arrest hour that still leaves room for stable history + ramp.
    earliest_arrest = deterioration_ramp_hours + OBSERVATION_WINDOW_HOURS + 2

    for patient_id in range(1, n_patients + 1):
        stay = int(rng.integers(min_stay_hours, max_stay_hours + 1))
        is_arrest = rng.random() < arrest_fraction
        baseline = {v: float(rng.normal(*_HEALTHY_BASELINE[v])) for v in VITALS}
        demo_rows.append(
            {"patient_id": patient_id, "age": int(rng.integers(20, 90)), "sex": int(rng.random() < 0.55)}
        )
        # Per-patient deterioration severity: some patients crash hard, others
        # decompensate subtly — this is what makes early warning genuinely hard.
        severity = float(rng.uniform(0.45, 1.1))

        if is_arrest:
            if stay <= earliest_arrest:
                stay = earliest_arrest + int(rng.integers(2, 12))
            arrest_hour = int(rng.integers(earliest_arrest, stay))
            recorded_hours = arrest_hour  # vitals recorded up to (but not at) arrest
        else:
            arrest_hour = None
            recorded_hours = stay

        for hour in range(recorded_hours):
            if is_arrest:
                hours_to_arrest = arrest_hour - hour
                frac = (
                    0.0
                    if hours_to_arrest > deterioration_ramp_hours
                    else (deterioration_ramp_hours - hours_to_arrest) / deterioration_ramp_hours
                )
                frac = float(np.clip(frac, 0.0, 1.0))
            else:
                frac = 0.0

            row: dict[str, float] = {"patient_id": patient_id, "hour": hour}
            for vital in VITALS:
                value = baseline[vital] + _DETERIORATION[vital] * frac * severity + rng.normal(0, _NOISE_SD[vital])
                low, high = _CLIP_RANGE[vital]
                value = float(np.clip(value, low, high))
                if rng.random() < missing_rate:
                    value = float("nan")
                row[vital] = value
            vital_rows.append(row)

        event_rows.append(
            {
                "patient_id": patient_id,
                "arrest_hour": float(arrest_hour) if arrest_hour is not None else float("nan"),
            }
        )

    return VitalSignsCohort(
        vitals=pd.DataFrame(vital_rows),
        events=pd.DataFrame(event_rows),
        demographics=pd.DataFrame(demo_rows),
    )


# --- Real-data sanitation (shared by every adapter) -----------------------

# Physiologically plausible ranges; values outside are treated as missing.
# Real monitors emit 0 / spikes on sensor disconnect — those are artefacts,
# not low heart rates, so we null them (to be imputed) rather than trust them.
PLAUSIBLE_RANGE: Final[dict[str, tuple[float, float]]] = {
    "pulse": (20.0, 250.0),
    "sbp": (40.0, 300.0),
    "dbp": (20.0, 200.0),
    "temperature": (30.0, 45.0),
    "spo2": (50.0, 100.0),
    "resp_rate": (3.0, 70.0),
}


def sanitize_vitals(vitals: pd.DataFrame, fix_temperature_units: bool = True) -> pd.DataFrame:
    """Null out implausible vital values (sensor artefacts) and fix temperature units.

    Real EHR/monitor data (MIMIC, and the competition's free-text ``VS_RSLT``)
    contains disconnect artefacts (pulse/RR = 0), out-of-range spikes, and
    temperatures charted in Fahrenheit under a Celsius label. We convert obvious
    Fahrenheit temperatures (> 45) to Celsius, then set any value outside a
    generous physiological range to ``NaN`` so it is imputed, not believed.
    """
    clean = vitals.copy()
    if fix_temperature_units and "temperature" in clean.columns:
        fahrenheit = clean["temperature"] > 45.0
        clean.loc[fahrenheit, "temperature"] = (clean.loc[fahrenheit, "temperature"] - 32.0) * 5.0 / 9.0
    for vital, (low, high) in PLAUSIBLE_RANGE.items():
        if vital in clean.columns:
            out_of_range = (clean[vital] < low) | (clean[vital] > high)
            clean.loc[out_of_range, vital] = np.nan
    return clean


# --- KHTH adapter (real competition data) ---------------------------------


# Maps the competition's ``VS_GBN`` codes to this module's canonical vitals.
_KHTH_VS_TYPE_MAP: Final[dict[str, str]] = {
    "HR": "pulse",
    "SBP": "sbp",
    "DBP": "dbp",
    "BT": "temperature",
    "SPO2": "spo2",
    "RR": "resp_rate",
}
# KHTH datetime columns are strings formatted YYYYMMDDHHMI (minute precision).
_KHTH_DATETIME_FMT: Final[str] = "%Y%m%d%H%M"


def _parse_khth_datetime(series: pd.Series) -> pd.Series:
    """Parse a KHTH ``YYYYMMDDHHMI`` string column to datetimes."""
    return pd.to_datetime(series.astype(str).str.strip(), format=_KHTH_DATETIME_FMT, errors="coerce")


def cohort_from_khth(
    vital: pd.DataFrame,
    pinfo: pd.DataFrame,
    *,
    patient_col: str = "PATID",
    admit_col: str = "INDD",
    time_col: str = "VSDT",
    type_col: str = "VS_GBN",
    value_col: str = "VS_RSLT",
    arrest_col: str = "CARDT",
    age_col: str = "AGE",
    sex_col: str = "SEX",
    bucket_hours: int = 1,
    vs_type_map: dict[str, str] | None = None,
) -> VitalSignsCohort:
    """Adapt the real ``KHTH_VITAL`` / ``KHTH_PINFO`` tables to a cohort.

    Schema (경북대병원 활력징후 데이터 설명서):

    - ``KHTH_VITAL`` is **long**: one row per measurement with ``VS_GBN`` (the
      vital code: HR/SBP/DBP/BT/SPO2/RR) and ``VS_RSLT`` (its value), timestamped
      by ``VSDT`` (``YYYYMMDDHHMI``). Rows join to ``KHTH_PINFO`` on
      ``PATID`` + ``INDD`` (a patient may have several admissions).
    - ``KHTH_PINFO`` provides the exact arrest time ``CARDT`` per stay, so labels
      are anchored to the true event — not inferred from the record's end.

    Each ``(PATID, INDD)`` admission becomes one ``patient_id``. Measurements are
    pivoted long→wide and bucketed to ``bucket_hours``-hour bins (default hourly;
    native sampling is ~minutes and irregular). Values are coerced to numeric.

    This is the single point binding to the real schema — the rest of the
    pipeline stays data-agnostic.
    """
    vs_type_map = vs_type_map or _KHTH_VS_TYPE_MAP

    records = vital.copy()
    records["patient_id"] = records[patient_col].astype(str) + "_" + records[admit_col].astype(str)
    records["_dt"] = _parse_khth_datetime(records[time_col])
    records["_canonical"] = records[type_col].map(vs_type_map)
    records["_value"] = pd.to_numeric(records[value_col], errors="coerce")
    records = records.dropna(subset=["_dt", "_canonical"])

    first_dt = records.groupby("patient_id")["_dt"].transform("min")
    bucket = pd.Timedelta(hours=bucket_hours)
    records["hour"] = (((records["_dt"] - first_dt) // bucket).astype(int) * bucket_hours)

    wide = records.pivot_table(
        index=["patient_id", "hour"],
        columns="_canonical",
        values="_value",
        aggfunc="mean",
    ).reset_index()
    wide.columns.name = None
    for vital_name in VITALS:
        if vital_name not in wide.columns:
            wide[vital_name] = np.nan
    vitals_wide = (
        wide[["patient_id", "hour", *VITALS]].sort_values(["patient_id", "hour"]).reset_index(drop=True)
    )

    stay_first_dt = records.groupby("patient_id")["_dt"].min().rename("_first").reset_index()
    info = pinfo.copy()
    info["patient_id"] = info[patient_col].astype(str) + "_" + info[admit_col].astype(str)
    info["_arrest"] = _parse_khth_datetime(info[arrest_col])

    demographics: pd.DataFrame | None = None
    if age_col in info.columns or sex_col in info.columns:
        demo = pd.DataFrame({"patient_id": info["patient_id"]})
        demo["age"] = pd.to_numeric(info[age_col], errors="coerce") if age_col in info.columns else np.nan
        demo["sex"] = info[sex_col].map({"M": 1, "F": 0}) if sex_col in info.columns else np.nan
        demographics = demo.drop_duplicates("patient_id").reset_index(drop=True)

    info = stay_first_dt.merge(info[["patient_id", "_arrest"]], on="patient_id", how="left")
    info["arrest_hour"] = (info["_arrest"] - info["_first"]) / pd.Timedelta(hours=1)
    events = info[["patient_id", "arrest_hour"]].copy()

    return VitalSignsCohort(vitals=sanitize_vitals(vitals_wide), events=events, demographics=demographics)


# --- MIMIC-IV adapter (real-data development & controls) ------------------

# MIMIC-IV (metavision) chartevents itemids for the six vitals we model.
MIMIC_ITEMID_MAP: Final[dict[int, str]] = {
    220045: "pulse",                 # Heart Rate
    220050: "sbp", 220179: "sbp",    # Arterial / NIBP systolic
    220051: "dbp", 220180: "dbp",    # Arterial / NIBP diastolic
    223762: "temperature", 223761: "temperature",  # Temp Celsius / Fahrenheit
    220277: "spo2",                  # SpO2
    220210: "resp_rate",             # Respiratory rate
}
_MIMIC_FAHRENHEIT_ITEMIDS: Final[frozenset[int]] = frozenset({223761})


def cohort_from_mimic(
    chartevents: pd.DataFrame,
    arrest_events: pd.DataFrame | None = None,
    *,
    id_col: str = "stay_id",
    time_col: str = "charttime",
    item_col: str = "itemid",
    value_col: str = "valuenum",
    itemid_map: dict[int, str] | None = None,
    arrest_id_col: str = "stay_id",
    arrest_time_col: str = "arrest_time",
    bucket_hours: int = 1,
) -> VitalSignsCohort:
    """Adapt MIMIC-IV ``chartevents`` to a :class:`VitalSignsCohort`.

    MIMIC-IV is the real, publicly available proxy used to *develop and validate*
    this pipeline before the 안심존: unlike the competition's case-only cohort it
    contains both arrest and non-arrest ICU stays, so it supplies the **controls**
    needed to measure false alarms honestly. Map cardiac-arrest times in via
    ``arrest_events`` (``stay_id`` -> ``arrest_time``); stays absent from it are
    treated as controls (``arrest_hour = NaN``).

    Fahrenheit temperatures (itemid 223761) are converted to Celsius. Timestamps
    are bucketed to ``bucket_hours``-hour bins, mirroring ``cohort_from_khth`` so
    the downstream windowing/labelling is identical across data sources.
    """
    itemid_map = itemid_map or MIMIC_ITEMID_MAP
    records = chartevents.copy()
    records["patient_id"] = records[id_col].astype(str)
    records["_dt"] = pd.to_datetime(records[time_col], errors="coerce")
    records["_value"] = pd.to_numeric(records[value_col], errors="coerce")

    fahrenheit = records[item_col].isin(_MIMIC_FAHRENHEIT_ITEMIDS)
    records.loc[fahrenheit, "_value"] = (records.loc[fahrenheit, "_value"] - 32.0) * 5.0 / 9.0
    records["_canonical"] = records[item_col].map(itemid_map)
    records = records.dropna(subset=["_dt", "_canonical"])

    first_dt = records.groupby("patient_id")["_dt"].transform("min")
    bucket = pd.Timedelta(hours=bucket_hours)
    records["hour"] = (((records["_dt"] - first_dt) // bucket).astype(int) * bucket_hours)

    wide = records.pivot_table(
        index=["patient_id", "hour"], columns="_canonical", values="_value", aggfunc="mean"
    ).reset_index()
    wide.columns.name = None
    for vital_name in VITALS:
        if vital_name not in wide.columns:
            wide[vital_name] = np.nan
    vitals_wide = (
        wide[["patient_id", "hour", *VITALS]].sort_values(["patient_id", "hour"]).reset_index(drop=True)
    )

    stay_first = records.groupby("patient_id")["_dt"].min().rename("_first").reset_index()
    if arrest_events is not None:
        arrests = arrest_events.copy()
        arrests["patient_id"] = arrests[arrest_id_col].astype(str)
        arrests["_arrest"] = pd.to_datetime(arrests[arrest_time_col], errors="coerce")
        merged = stay_first.merge(arrests[["patient_id", "_arrest"]], on="patient_id", how="left")
        merged["arrest_hour"] = (merged["_arrest"] - merged["_first"]) / pd.Timedelta(hours=1)
    else:
        merged = stay_first.copy()
        merged["arrest_hour"] = float("nan")  # every stay is a control
    cohort_events = merged[["patient_id", "arrest_hour"]].copy()

    return VitalSignsCohort(vitals=sanitize_vitals(vitals_wide), events=cohort_events)


# --- Windowing & labelling ------------------------------------------------


def _window_features(observation: pd.DataFrame) -> dict[str, float]:
    """Compute per-vital statistics for a single observation window."""
    feats: dict[str, float] = {}
    hours = observation["hour"].to_numpy(dtype=float)

    for vital in VITALS:
        values = observation[vital].to_numpy(dtype=float)
        mask = ~np.isnan(values)
        if not mask.any():
            for stat in WINDOW_STATS:
                feats[f"{vital}_{stat}"] = float("nan")
            continue

        present = values[mask]
        present_hours = hours[mask]
        feats[f"{vital}_mean"] = float(present.mean())
        feats[f"{vital}_std"] = float(present.std())  # population std (ddof=0)
        feats[f"{vital}_min"] = float(present.min())
        feats[f"{vital}_max"] = float(present.max())
        feats[f"{vital}_last"] = float(present[-1])
        feats[f"{vital}_delta"] = float(present[-1] - present[0])
        if present.size >= 2 and np.ptp(present_hours) > 0:
            feats[f"{vital}_slope"] = float(np.polyfit(present_hours, present, 1)[0])
        else:
            feats[f"{vital}_slope"] = 0.0

    pulse_last = feats.get("pulse_last", float("nan"))
    sbp_last = feats.get("sbp_last", float("nan"))
    dbp_last = feats.get("dbp_last", float("nan"))
    feats["shock_index_last"] = (
        float(pulse_last / sbp_last)
        if np.isfinite(pulse_last) and np.isfinite(sbp_last) and sbp_last != 0
        else float("nan")
    )
    feats["pulse_pressure_last"] = (
        float(sbp_last - dbp_last) if np.isfinite(sbp_last) and np.isfinite(dbp_last) else float("nan")
    )

    return {name: feats.get(name, float("nan")) for name in feature_names()}


def build_windows(
    cohort: VitalSignsCohort,
    observation_window_hours: int = OBSERVATION_WINDOW_HOURS,
    prediction_horizon_hours: int = PREDICTION_HORIZON_HOURS,
    gap_hours: int = GAP_HOURS,
    step_hours: int = STEP_HOURS,
    min_valid_fraction: float = 0.5,
) -> WindowedDataset:
    """Slide a window over each patient and label by imminent arrest.

    A window ending at hour ``t`` uses observations in ``[t - W + 1, t]`` and is
    labelled positive when an arrest falls in ``(t + gap, t + gap + horizon]``.
    Windows with fewer than ``min_valid_fraction`` of hours present are skipped.

    Because the competition cohort is case-only, positives come from the hours
    just before each patient's arrest and negatives from that same patient's
    earlier, more stable hours (plus any control patients present).
    """
    events = dict(zip(cohort.events["patient_id"], cohort.events["arrest_hour"]))
    names = feature_names()
    min_rows = max(2, int(np.ceil(min_valid_fraction * observation_window_hours)))

    rows: list[dict[str, float]] = []
    labels: list[int] = []
    groups: list[int] = []
    times_to_arrest: list[float] = []

    for patient_id, group in cohort.vitals.groupby("patient_id"):
        group = group.sort_values("hour")
        arrest_hour = events.get(patient_id, float("nan"))
        has_arrest = arrest_hour is not None and np.isfinite(arrest_hour)
        max_hour = int(group["hour"].max())

        for end in range(observation_window_hours - 1, max_hour + 1, step_hours):
            if has_arrest and end >= arrest_hour:
                continue  # cannot observe at/after the arrest
            low = end - observation_window_hours + 1
            observation = group[(group["hour"] >= low) & (group["hour"] <= end)]
            if len(observation) < min_rows:
                continue

            if has_arrest and (end + gap_hours) < arrest_hour <= (end + gap_hours + prediction_horizon_hours):
                label = 1
            else:
                label = 0

            rows.append(_window_features(observation))
            labels.append(label)
            groups.append(patient_id)
            times_to_arrest.append(float(arrest_hour - end) if has_arrest else float("nan"))

    features = pd.DataFrame(rows, columns=names)
    return WindowedDataset(
        features=features,
        labels=pd.Series(labels, name="label"),
        groups=pd.Series(groups, name="patient_id"),
        feature_names=names,
        time_to_arrest=pd.Series(times_to_arrest, name="time_to_arrest"),
    )


PERSONAL_BASELINE_HOURS: Final[int] = 6


def add_personalized_features(
    windowed: WindowedDataset,
    cohort: VitalSignsCohort,
    baseline_hours: int = PERSONAL_BASELINE_HOURS,
) -> WindowedDataset:
    """Append 'deviation from the patient's own baseline' features.

    Each patient's baseline is the mean of their first ``baseline_hours`` stable
    hours. For every window we add, per vital, how far its recent level sits from
    that personal baseline (``{vital}_last_dev`` / ``{vital}_mean_dev``). A value
    that looks normal for the ward can be a large *personal* deviation — this is
    the core of the false-alarm-reduction thesis and turns the case-only cohort's
    within-patient structure into the method itself.
    """
    early = cohort.vitals[cohort.vitals["hour"] < baseline_hours]
    baseline = early.groupby("patient_id")[list(VITALS)].mean()
    aligned = baseline.reindex(windowed.groups.to_numpy()).reset_index(drop=True)

    features = windowed.features.copy()
    new_names: list[str] = []
    for vital in VITALS:
        for stat in ("last", "mean"):
            dev_col = f"{vital}_{stat}_dev"
            features[dev_col] = features[f"{vital}_{stat}"].to_numpy() - aligned[vital].to_numpy()
            new_names.append(dev_col)
    features[new_names] = features[new_names].fillna(0.0)

    return WindowedDataset(
        features=features,
        labels=windowed.labels,
        groups=windowed.groups,
        feature_names=windowed.feature_names + new_names,
        time_to_arrest=windowed.time_to_arrest,
    )


def add_static_features(windowed: WindowedDataset, cohort: VitalSignsCohort) -> WindowedDataset:
    """Append per-patient static demographics (``static_age``, ``static_sex``).

    No-op when the cohort has no demographics. Missing values are filled with the
    column median so the features stay usable on partial data.
    """
    if cohort.demographics is None:
        return windowed
    demo = cohort.demographics.drop_duplicates("patient_id").set_index("patient_id")
    aligned = demo.reindex(windowed.groups.to_numpy()).reset_index(drop=True)

    features = windowed.features.copy()
    new_names: list[str] = []
    for col in ("age", "sex"):
        if col in aligned.columns:
            features[f"static_{col}"] = pd.to_numeric(aligned[col], errors="coerce").to_numpy()
            new_names.append(f"static_{col}")
    if new_names:
        features[new_names] = features[new_names].apply(lambda c: c.fillna(c.median()))
    return WindowedDataset(
        features=features,
        labels=windowed.labels,
        groups=windowed.groups,
        feature_names=windowed.feature_names + new_names,
        time_to_arrest=windowed.time_to_arrest,
    )


def patient_level_split(
    windowed: WindowedDataset,
    test_size: float = 0.2,
    seed: int = 42,
) -> PatientSplit:
    """Split windows by patient so no patient leaks across train/test.

    Patients are stratified by whether they contribute any positive window.
    Missing feature values are imputed with **train** medians (fit on train,
    applied to test) to avoid leakage.
    """
    groups = windowed.groups
    patients = np.array(sorted(groups.unique()))
    positive_patients = windowed.labels.groupby(groups).max().reindex(patients).fillna(0).astype(int)
    stratify = positive_patients.to_numpy() if positive_patients.nunique() > 1 else None

    train_patients, test_patients = train_test_split(
        patients,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )

    train_mask = groups.isin(train_patients).to_numpy()
    test_mask = groups.isin(test_patients).to_numpy()

    X_train = windowed.features[train_mask].reset_index(drop=True)
    X_test = windowed.features[test_mask].reset_index(drop=True)
    y_train = windowed.labels[train_mask].to_numpy()
    y_test = windowed.labels[test_mask].to_numpy()

    medians = X_train.median(numeric_only=True)
    X_train = X_train.fillna(medians).fillna(0.0)
    X_test = X_test.fillna(medians).fillna(0.0)

    time_to_arrest_test = (
        windowed.time_to_arrest[test_mask].to_numpy() if windowed.time_to_arrest is not None else None
    )

    return PatientSplit(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        groups_train=groups[train_mask].to_numpy(),
        groups_test=groups[test_mask].to_numpy(),
        feature_names=windowed.feature_names,
        imputation_medians=medians,
        time_to_arrest_test=time_to_arrest_test,
    )


def _class_summary(labels: np.ndarray, name: str) -> str:
    """Format a positive-class prevalence summary."""
    total = len(labels)
    positives = int(np.sum(labels == 1))
    pct = 100.0 * positives / total if total else 0.0
    return f"{name}: n={total}, positives={positives} ({pct:.2f}%)"


def main() -> None:
    """Build a synthetic cohort and report window/split statistics."""
    cohort = generate_synthetic_cohort(n_patients=500, arrest_fraction=0.5)
    n_arrest = int(cohort.events["arrest_hour"].notna().sum())
    print(f"Cohort: {len(cohort.events)} patients ({n_arrest} arrest), {len(cohort.vitals)} hourly rows")

    windowed = build_windows(cohort)
    print(f"Windows: {len(windowed.features)} x {len(windowed.feature_names)} features")
    print(_class_summary(windowed.labels.to_numpy(), "All windows"))

    split = patient_level_split(windowed)
    overlap = set(split.groups_train) & set(split.groups_test)
    print(f"Patient-level split — train/test patient overlap: {len(overlap)} (must be 0)")
    print(_class_summary(split.y_train, "Train"))
    print(_class_summary(split.y_test, "Test"))


if __name__ == "__main__":
    main()
