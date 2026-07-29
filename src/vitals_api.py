"""FastAPI service behind the early-warning dashboard (``web/``).

Serves the *time-series* track. Distinct from ``legacy/main.py``, which is the
archived static heart-failure API and stays untouched.

The service builds one cohort at startup, trains the model on the training
patients and then only ever exposes **test** patients — the dashboard must not
show scores a patient was fitted on. Everything is in memory; there is no
database and no write path.

Run it:

    uvicorn vitals_api:app --app-dir src --reload           # synthetic cohort
    EWS_DATA_DIR=data/challenge2019/training_setA \\
        uvicorn vitals_api:app --app-dir src --reload       # real Challenge-2019

Startup on real data takes a minute or two; set ``EWS_MAX_FILES`` to trim it.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from vitals_data import (
    VITALS,
    VitalSignsCohort,
    add_personalized_features,
    build_windows,
    cohort_from_challenge2019,
    generate_synthetic_cohort,
    patient_level_split,
)
from vitals_narrate import VITAL_KO, build_evidence, format_evidence, narrate
from vitals_train import alarm_burden, compute_news_scores, threshold_at_specificity, train_xgboost

RiskLevel = Literal["good", "warning", "serious", "critical"]

# Risk bands. "critical" starts at the operating threshold, so the colour a
# clinician sees and the alarm the model raises can never disagree.
BAND_WARNING = 0.25
BAND_SERIOUS = 0.50
NEWS_ALARM_THRESHOLD = 5.0


# --- Response models ------------------------------------------------------


class VitalPoint(BaseModel):
    """One vital at one hour, with the patient's own baseline for context."""

    vital: str
    label: str
    unit: str
    value: float | None
    baseline: float | None


class PatientSummary(BaseModel):
    """A row in the ward list."""

    patient_id: str
    risk: float
    risk_level: RiskLevel
    risk_delta: float
    model_alarm: bool
    news_score: float
    news_alarm: bool
    hours_observed: int
    last_hour: int
    arrest_hour: float | None


class TimelinePoint(BaseModel):
    """Model risk and NEWS at one window end."""

    hour: int
    risk: float
    news_score: float


class Factor(BaseModel):
    """One SHAP driver, already described in Korean."""

    feature: str
    description: str
    unit: str
    value: float
    shap: float
    effect: str


class Evidence(BaseModel):
    """Why the model did or did not alarm at one hour. Mirrors ``build_evidence``."""

    patient_id: str
    hour: int
    risk: float
    threshold: float
    risk_level: RiskLevel
    model_alarm: bool
    news_score: float
    news_threshold: float
    news_alarm: bool
    disagreement: Literal["none", "model_only", "news_only"]
    top_factors: list[Factor]
    vitals: list[VitalPoint]
    hours_to_arrest: float | None
    text: str


class PatientDetail(BaseModel):
    """Everything the detail view needs for one patient."""

    patient_id: str
    #: From the source record only. Null when it does not carry the field.
    age: int | None
    sex: Literal["M", "F"] | None
    hours_observed: int
    threshold: float
    news_threshold: float
    arrest_hour: float | None
    timeline: list[TimelinePoint]
    trajectory: list[dict[str, Any]]
    baseline: dict[str, float]
    evidence: Evidence


class Narration(BaseModel):
    """LLM-worded explanation of an evidence packet."""

    text: str
    llm_model: str


class BurdenRow(BaseModel):
    """Alarm burden for both scores at one matched sensitivity."""

    sensitivity: float
    model_alarms_per_100: float
    model_specificity: float
    news_alarms_per_100: float
    news_specificity: float


class Overview(BaseModel):
    """Cohort-level context shown above the ward list."""

    source: str
    #: Held-out patients the figures below are measured on.
    patients: int
    windows: int
    positive_rate: float
    threshold: float
    news_threshold: float
    alarming: int
    #: How many of those patients the ward list can actually open.
    browsable: int
    burden: list[BurdenRow]
    llm_available: bool


# --- In-memory state ------------------------------------------------------


@dataclass
class Service:
    """A fitted model plus everything the dashboard shows, ready to serve.

    Deliberately *not* the whole cohort. Building from raw Challenge-2019 costs
    ~5 minutes and 4 GB, almost all of it in patients the dashboard never
    displays. Cohort-level statistics are computed on the full held-out set and
    then reduced to scalars; only the top ``EWS_MAX_PATIENTS`` patients keep
    their rows. What survives is small enough to pickle and ship.
    """

    source: str
    model: Any
    features: pd.DataFrame
    patient_ids: np.ndarray
    hours: np.ndarray
    time_to_arrest: np.ndarray
    risk: np.ndarray
    news: np.ndarray
    threshold: float
    baselines: dict[str, dict[str, float]] = field(default_factory=dict)
    trajectories: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    #: patient_id -> {"age": float|None, "sex": "M"|"F"|None}. Only what the
    #: source actually records — the dashboard never invents a bed or a ward.
    demographics: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Cohort-level figures, measured on the full held-out set before trimming.
    cohort_patients: int = 0
    cohort_windows: int = 0
    cohort_positive_rate: float = 0.0
    cohort_alarming: int = 0
    burden: list[dict[str, float]] = field(default_factory=list)

    def rows_for(self, patient_id: str) -> np.ndarray:
        """Positional indices of one patient's windows, in time order."""
        rows = np.flatnonzero(self.patient_ids.astype(str) == patient_id)
        return rows[np.argsort(self.hours[rows])]


def risk_level(risk: float, threshold: float) -> RiskLevel:
    """Bucket a probability into the four reserved status bands."""
    if risk >= max(threshold, BAND_SERIOUS):
        return "critical"
    if risk >= BAND_SERIOUS:
        return "serious"
    if risk >= BAND_WARNING:
        return "warning"
    return "good"


def build_service() -> Service:
    """Load a cohort, fit on the training patients, keep the interesting ones.

    Everything the dashboard reports about the cohort is measured here, on the
    complete held-out set, *before* trimming — so the alarm-burden comparison is
    honest even though only a few hundred patients remain browsable.
    """
    data_dir = os.environ.get("EWS_DATA_DIR")
    raw_max_files = os.environ.get("EWS_MAX_FILES", "2000")
    max_files = None if raw_max_files.lower() in {"none", "all", "0"} else int(raw_max_files)
    horizon = int(os.environ.get("EWS_HORIZON", "6"))
    max_patients = int(os.environ.get("EWS_MAX_PATIENTS", "400"))

    if data_dir:
        source = "challenge2019"
        cohort = cohort_from_challenge2019(data_dir, max_files=max_files)
    else:
        source = "synthetic"
        cohort = generate_synthetic_cohort()
        horizon = 1

    windowed = add_personalized_features(
        build_windows(cohort, prediction_horizon_hours=horizon), cohort
    )
    split = patient_level_split(windowed)
    model, _ = train_xgboost(split)

    # Reconstruct the test mask so window hours line up with X_test row order.
    test_patients = set(np.unique(split.groups_test).tolist())
    mask = windowed.groups.isin(test_patients).to_numpy()

    risk = model.predict_proba(split.X_test)[:, 1]
    news = compute_news_scores(split.X_test)
    threshold = float(threshold_at_specificity(split.y_test, risk))

    burden = []
    for target in (0.5, 0.7, 0.9):
        model_row = alarm_burden(split.y_test, risk, target_sensitivity=target)
        news_row = alarm_burden(split.y_test, news, target_sensitivity=target)
        burden.append(
            {
                "sensitivity": target,
                "model_alarms_per_100": round(model_row["alarms_per_100_windows"], 2),
                "model_specificity": round(model_row["specificity"], 3),
                "news_alarms_per_100": round(news_row["alarms_per_100_windows"], 2),
                "news_specificity": round(news_row["specificity"], 3),
            }
        )

    service = Service(
        source=source,
        model=model,
        features=split.X_test,
        patient_ids=split.groups_test,
        hours=windowed.window_end_hour[mask].to_numpy(),
        time_to_arrest=windowed.time_to_arrest[mask].to_numpy(),
        risk=risk,
        news=news,
        threshold=threshold,
        cohort_patients=len(test_patients),
        cohort_windows=int(len(risk)),
        cohort_positive_rate=round(float(split.y_test.mean()), 4),
        cohort_alarming=int(np.sum(risk >= threshold)),
        burden=burden,
    )
    _trim_to_top_patients(service, cohort, max_patients)
    return service


def _trim_to_top_patients(service: Service, cohort: VitalSignsCohort, keep: int) -> None:
    """Keep only the ``keep`` riskiest patients' rows, plus their trajectories.

    Mutates ``service`` in place. The cohort-level fields are already computed,
    so dropping rows here changes what is browsable, never what is reported.
    """
    ids = service.patient_ids.astype(str)
    peak = pd.Series(service.risk).groupby(ids).max().sort_values(ascending=False)
    kept = set(peak.head(keep).index)

    row_mask = np.isin(ids, list(kept))
    service.features = service.features[row_mask].reset_index(drop=True)
    service.patient_ids = ids[row_mask]
    service.hours = service.hours[row_mask]
    service.time_to_arrest = service.time_to_arrest[row_mask]
    service.risk = service.risk[row_mask]
    service.news = service.news[row_mask]

    vitals = cohort.vitals[cohort.vitals["patient_id"].astype(str).isin(kept)]
    for patient_id, group in vitals.groupby(vitals["patient_id"].astype(str)):
        rows = group.sort_values("hour")
        service.trajectories[patient_id] = [
            {
                "hour": int(record["hour"]),
                **{
                    vital: (None if not np.isfinite(record[vital]) else round(float(record[vital]), 1))
                    for vital in VITALS
                },
            }
            for record in rows.to_dict("records")
        ]

    early = vitals[vitals["hour"] < 6]
    for patient_id, group in early.groupby(early["patient_id"].astype(str)):
        means = group[list(VITALS)].mean()
        service.baselines[patient_id] = {
            vital: round(float(means[vital]), 1) for vital in VITALS if np.isfinite(means[vital])
        }

    if cohort.demographics is not None:
        demo = cohort.demographics.copy()
        demo["patient_id"] = demo["patient_id"].astype(str)
        for record in demo[demo["patient_id"].isin(kept)].to_dict("records"):
            age, sex = record.get("age"), record.get("sex")
            service.demographics[record["patient_id"]] = {
                "age": None if age is None or not np.isfinite(age) else round(float(age)),
                # Challenge-2019 encodes Gender as 1=male, 0=female.
                "sex": None if sex is None or not np.isfinite(sex) else ("M" if sex >= 0.5 else "F"),
            }


def _artifact_path() -> Path | None:
    """Where a prebuilt service is cached, if caching is enabled."""
    configured = os.environ.get("EWS_ARTIFACT")
    return Path(configured) if configured else None


@lru_cache(maxsize=1)
def service() -> Service:
    """The single process-wide service, loaded from an artifact when available.

    Building from raw data takes minutes; loading the artifact takes about a
    second. Deployment should always ship the artifact — see ``scripts/``.
    """
    path = _artifact_path()
    if path and path.exists():
        with path.open("rb") as handle:
            return pickle.load(handle)

    built = build_service()
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(built, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return built


# --- Assembly helpers -----------------------------------------------------


def _vital_points(svc: Service, patient_id: str, row: int) -> list[VitalPoint]:
    """Current value plus personal baseline for every vital, for one window."""
    baseline = svc.baselines.get(patient_id, {})
    points = []
    for vital in VITALS:
        label, unit = VITAL_KO[vital]
        raw = svc.features[f"{vital}_last"].iloc[row]
        value = float(raw) if np.isfinite(raw) else None
        points.append(
            VitalPoint(
                vital=vital,
                label=label,
                unit=unit,
                value=None if value is None else round(value, 1),
                baseline=baseline.get(vital),
            )
        )
    return points


def _evidence(svc: Service, patient_id: str, row: int) -> Evidence:
    """Build one explanation packet, reusing the notebook's own code path."""
    packet = build_evidence(
        svc.model,
        svc.features,
        row,
        threshold=svc.threshold,
        news_threshold=NEWS_ALARM_THRESHOLD,
        source=svc.source,
        patient_id=patient_id,
    )
    hours_to_arrest = svc.time_to_arrest[row]
    model_alarm, news_alarm = packet["model_alarm"], packet["news_alarm"]
    if model_alarm == news_alarm:
        disagreement = "none"
    else:
        disagreement = "model_only" if model_alarm else "news_only"

    return Evidence(
        patient_id=patient_id,
        hour=int(svc.hours[row]),
        risk=packet["risk"],
        threshold=packet["threshold"],
        risk_level=risk_level(packet["risk"], svc.threshold),
        model_alarm=model_alarm,
        news_score=packet["news_score"],
        news_threshold=packet["news_threshold"],
        news_alarm=news_alarm,
        disagreement=disagreement,
        top_factors=[Factor(**factor) for factor in packet["top_factors"]],
        vitals=_vital_points(svc, patient_id, row),
        hours_to_arrest=None if not np.isfinite(hours_to_arrest) else float(hours_to_arrest),
        text=format_evidence(packet),
    )


# --- App ------------------------------------------------------------------

app = FastAPI(
    title="Cardiac-arrest early warning",
    description="Vital-sign early-warning scores, alarm decisions and their SHAP evidence.",
    version="0.1.0",
)

def _cors_origins() -> dict[str, Any]:
    """Allowed browser origins for the dashboard.

    The default covers a dev server reached over localhost or 127.0.0.1 — which
    is what VS Code / SSH port forwarding gives you. Tunnelled URLs (dev tunnels,
    Codespaces) have an unpredictable host, so set ``EWS_CORS_ORIGINS`` to that
    origin, or to ``*`` when you are on a trusted network and just want it to work.
    """
    configured = os.environ.get("EWS_CORS_ORIGINS", "").strip()
    if configured == "*":
        return {"allow_origin_regex": ".*"}
    if configured:
        return {"allow_origins": [origin.strip() for origin in configured.split(",")]}
    return {"allow_origins": ["http://localhost:3000", "http://127.0.0.1:3000"]}


app.add_middleware(
    CORSMiddleware,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    **_cors_origins(),
)


@app.get("/api/health")
def health() -> dict[str, str]:
    """Liveness probe that does not trigger the (slow) cohort build."""
    return {"status": "ok"}


@app.get("/api/overview", response_model=Overview)
def overview() -> Overview:
    """Cohort context plus the alarm-burden comparison against NEWS."""
    svc = service()
    return Overview(
        source=svc.source,
        patients=svc.cohort_patients,
        windows=svc.cohort_windows,
        positive_rate=svc.cohort_positive_rate,
        threshold=round(svc.threshold, 3),
        news_threshold=NEWS_ALARM_THRESHOLD,
        alarming=svc.cohort_alarming,
        browsable=int(len(np.unique(svc.patient_ids))),
        burden=[BurdenRow(**row) for row in svc.burden],
        llm_available=bool(os.environ.get("GROQ_API_KEY")) or (
            os.path.exists(os.path.join(os.path.dirname(__file__), "..", ".env"))
        ),
    )


@app.get("/api/patients", response_model=list[PatientSummary])
def patients(limit: int = Query(40, ge=1, le=1000)) -> list[PatientSummary]:
    """Ward list, most at-risk first. Each patient's latest window."""
    svc = service()
    summaries = []
    for patient_id in np.unique(svc.patient_ids.astype(str)):
        rows = svc.rows_for(patient_id)
        last = int(rows[-1])
        previous = int(rows[-2]) if len(rows) > 1 else last
        arrest = svc.hours[last] + svc.time_to_arrest[last]
        summaries.append(
            PatientSummary(
                patient_id=patient_id,
                risk=round(float(svc.risk[last]), 3),
                risk_level=risk_level(float(svc.risk[last]), svc.threshold),
                risk_delta=round(float(svc.risk[last] - svc.risk[previous]), 3),
                model_alarm=bool(svc.risk[last] >= svc.threshold),
                news_score=float(svc.news[last]),
                news_alarm=bool(svc.news[last] >= NEWS_ALARM_THRESHOLD),
                hours_observed=len(rows),
                last_hour=int(svc.hours[last]),
                arrest_hour=float(arrest) if np.isfinite(arrest) else None,
            )
        )

    summaries.sort(key=lambda item: item.risk, reverse=True)
    return summaries[:limit]


@app.get("/api/patients/{patient_id}", response_model=PatientDetail)
def patient_detail(patient_id: str, hour: int | None = None) -> PatientDetail:
    """One patient's risk timeline, vital trajectory and evidence at ``hour``."""
    svc = service()
    rows = svc.rows_for(patient_id)
    if not len(rows):
        raise HTTPException(status_code=404, detail=f"unknown patient {patient_id!r}")

    selected = int(rows[-1])
    if hour is not None:
        matched = [int(r) for r in rows if int(svc.hours[r]) == hour]
        if not matched:
            raise HTTPException(status_code=404, detail=f"no window ends at hour {hour}")
        selected = matched[0]

    arrest = svc.hours[selected] + svc.time_to_arrest[selected]
    demographics = svc.demographics.get(patient_id, {})
    return PatientDetail(
        patient_id=patient_id,
        age=demographics.get("age"),
        sex=demographics.get("sex"),
        hours_observed=len(rows),
        threshold=round(svc.threshold, 3),
        news_threshold=NEWS_ALARM_THRESHOLD,
        arrest_hour=float(arrest) if np.isfinite(arrest) else None,
        timeline=[
            TimelinePoint(
                hour=int(svc.hours[row]),
                risk=round(float(svc.risk[row]), 3),
                news_score=float(svc.news[row]),
            )
            for row in rows
        ],
        trajectory=svc.trajectories.get(patient_id, []),
        baseline=svc.baselines.get(patient_id, {}),
        evidence=_evidence(svc, patient_id, selected),
    )


@app.post("/api/patients/{patient_id}/explain", response_model=Narration)
def explain(patient_id: str, hour: int | None = None) -> Narration:
    """Word one window's evidence with the LLM.

    Refuses for credentialed cohorts — ``vitals_narrate.narrate`` raises and we
    surface that as 403 rather than silently sending the data.
    """
    svc = service()
    rows = svc.rows_for(patient_id)
    if not len(rows):
        raise HTTPException(status_code=404, detail=f"unknown patient {patient_id!r}")

    selected = int(rows[-1])
    if hour is not None:
        matched = [int(r) for r in rows if int(svc.hours[r]) == hour]
        if not matched:
            raise HTTPException(status_code=404, detail=f"no window ends at hour {hour}")
        selected = matched[0]

    packet = build_evidence(
        svc.model,
        svc.features,
        selected,
        threshold=svc.threshold,
        news_threshold=NEWS_ALARM_THRESHOLD,
        source=svc.source,
        patient_id=patient_id,
    )
    try:
        text = narrate(packet)
    except PermissionError as error:
        raise HTTPException(status_code=403, detail=str(error)) from error
    except Exception as error:  # noqa: BLE001 - upstream/network failure
        raise HTTPException(status_code=502, detail=f"LLM unavailable: {error}") from error

    from vitals_narrate import DEFAULT_MODEL

    return Narration(text=text, llm_model=DEFAULT_MODEL)
