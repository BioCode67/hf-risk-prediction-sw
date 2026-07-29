"""Tests for the dashboard API (``vitals_api``).

The real service builds a 500-patient cohort at startup, which is far too slow
for CI, so these tests swap in a small synthetic one via ``build_service``. The
contract under test is the *shape* of the payloads — that is what
``web/src/lib/types.ts`` is written against — plus the two invariants that would
be silently wrong otherwise: only test patients are exposed, and credentialed
cohorts never reach the LLM.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

import vitals_api
from vitals_api import NEWS_ALARM_THRESHOLD, risk_level
from vitals_data import (
    add_personalized_features,
    build_windows,
    generate_synthetic_cohort,
    patient_level_split,
)
from vitals_train import compute_news_scores, threshold_at_specificity, train_xgboost


def _small_service(source: str = "synthetic", keep: int = 6) -> vitals_api.Service:
    """A Service over a deliberately tiny cohort, built the same way as the real one.

    Mirrors ``build_service``: cohort-level figures are measured on the whole
    held-out set, then ``_trim_to_top_patients`` reduces what is browsable.
    ``keep`` is small so the trimming is actually exercised.
    """
    cohort = generate_synthetic_cohort(n_patients=40, seed=0)
    windowed = add_personalized_features(build_windows(cohort), cohort)
    split = patient_level_split(windowed)
    model, _ = train_xgboost(split)

    test_patients = set(np.unique(split.groups_test).tolist())
    mask = windowed.groups.isin(test_patients).to_numpy()
    risk = model.predict_proba(split.X_test)[:, 1]
    news = compute_news_scores(split.X_test)
    threshold = float(threshold_at_specificity(split.y_test, risk))

    service = vitals_api.Service(
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
        burden=[
            {
                "sensitivity": target,
                "model_alarms_per_100": 1.0,
                "model_specificity": 0.9,
                "news_alarms_per_100": 2.0,
                "news_specificity": 0.8,
            }
            for target in (0.5, 0.7, 0.9)
        ],
    )
    vitals_api._trim_to_top_patients(service, cohort, keep)
    return service


@pytest.fixture(scope="module")
def service() -> vitals_api.Service:
    return _small_service()


@pytest.fixture
def client(service, monkeypatch) -> TestClient:
    monkeypatch.setattr(vitals_api, "build_service", lambda: service)
    vitals_api.service.cache_clear()
    yield TestClient(vitals_api.app)
    vitals_api.service.cache_clear()


def test_health_does_not_build_the_cohort():
    """Liveness must answer before the (slow) startup work, so it takes no fixture."""
    assert TestClient(vitals_api.app).get("/api/health").json() == {"status": "ok"}


def test_overview_reports_burden_at_three_sensitivities(client):
    body = client.get("/api/overview").json()
    assert body["patients"] > 0
    assert body["windows"] > 0
    assert [row["sensitivity"] for row in body["burden"]] == [0.5, 0.7, 0.9]


def test_overview_separates_measured_cohort_from_browsable_subset(client, service):
    """Trimming changes what you can open, never what the figures are measured on."""
    body = client.get("/api/overview").json()
    assert body["browsable"] == len(np.unique(service.patient_ids))
    assert body["browsable"] < body["patients"]


def test_trimming_keeps_trajectories_and_baselines_for_every_kept_patient(service):
    for patient_id in np.unique(service.patient_ids):
        assert service.trajectories.get(patient_id), f"no trajectory for {patient_id}"
        assert service.baselines.get(patient_id), f"no baseline for {patient_id}"


def test_patients_are_ranked_by_risk_and_capped_by_limit(client):
    body = client.get("/api/patients?limit=5").json()
    assert 0 < len(body) <= 5
    assert [row["risk"] for row in body] == sorted((row["risk"] for row in body), reverse=True)


def test_patients_exposes_only_held_out_patients(client, service):
    body = client.get("/api/patients?limit=200").json()
    held_out = {str(patient) for patient in np.unique(service.patient_ids)}
    assert {row["patient_id"] for row in body} <= held_out


def test_patient_detail_aligns_evidence_with_the_requested_hour(client):
    patient_id = client.get("/api/patients?limit=1").json()[0]["patient_id"]
    detail = client.get(f"/api/patients/{patient_id}").json()

    assert detail["timeline"], "a patient with no windows should not be listed"
    assert [point["hour"] for point in detail["timeline"]] == sorted(
        point["hour"] for point in detail["timeline"]
    )
    assert detail["evidence"]["hour"] == detail["timeline"][-1]["hour"]

    first = detail["timeline"][0]["hour"]
    pinned = client.get(f"/api/patients/{patient_id}?hour={first}").json()
    assert pinned["evidence"]["hour"] == first


def test_patient_detail_evidence_carries_every_vital_and_three_factors(client):
    patient_id = client.get("/api/patients?limit=1").json()[0]["patient_id"]
    evidence = client.get(f"/api/patients/{patient_id}").json()["evidence"]

    assert len(evidence["top_factors"]) == 3
    assert [point["vital"] for point in evidence["vitals"]] == [
        "pulse",
        "sbp",
        "dbp",
        "temperature",
        "spo2",
        "resp_rate",
    ]
    assert evidence["text"], "the deterministic explanation is never optional"
    assert evidence["disagreement"] in {"none", "model_only", "news_only"}


def test_patients_limit_covers_the_whole_browsable_set(client, service):
    """The UI asks for every browsable patient at once; the cap must not reject it."""
    browsable = len(np.unique(service.patient_ids))
    assert client.get(f"/api/patients?limit={browsable}").status_code == 200
    assert client.get("/api/patients?limit=1000").status_code == 200


def test_unknown_patient_and_unknown_hour_are_404(client):
    assert client.get("/api/patients/does-not-exist").status_code == 404
    patient_id = client.get("/api/patients?limit=1").json()[0]["patient_id"]
    assert client.get(f"/api/patients/{patient_id}?hour=99999").status_code == 404


def test_explain_refuses_credentialed_cohorts(monkeypatch):
    """MIMIC / KHTH must never reach a third-party API — 403, not a request."""
    monkeypatch.setattr(vitals_api, "build_service", lambda: _small_service(source="mimic"))
    vitals_api.service.cache_clear()
    try:
        client = TestClient(vitals_api.app)
        patient_id = client.get("/api/patients?limit=1").json()[0]["patient_id"]
        response = client.post(f"/api/patients/{patient_id}/explain")
        assert response.status_code == 403
        assert "mimic" in response.json()["detail"]
    finally:
        vitals_api.service.cache_clear()


@pytest.mark.parametrize(
    ("risk", "threshold", "expected"),
    [
        (0.01, 0.6, "good"),
        (0.30, 0.6, "warning"),
        (0.55, 0.6, "serious"),
        (0.70, 0.6, "critical"),
        # A threshold below the serious band must not let "critical" undercut it.
        (0.55, 0.004, "critical"),
    ],
)
def test_risk_level_bands(risk, threshold, expected):
    assert risk_level(risk, threshold) == expected


def test_news_alarm_threshold_matches_the_notebook():
    """05_challenge2019 uses NEWS >= 5; the dashboard must not drift from it."""
    assert NEWS_ALARM_THRESHOLD == 5.0
