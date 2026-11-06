"""Live bedside scoring: binning, baseline gating and the API contract.

Uses the built-in synthetic cohort, so it runs on a fresh clone with no data.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from vitals_data import VITALS
from vitals_live import LiveReading, bin_readings, live_features, ready_state, sanitize

STABLE = {"pulse": 78, "sbp": 122, "dbp": 76, "temperature": 36.8, "spo2": 98, "resp_rate": 16}
CRASHING = {"pulse": 128, "sbp": 82, "dbp": 52, "temperature": 36.2, "spo2": 86, "resp_rate": 30}


def stream(seconds: int, step: float, worsen_from: int | None = None) -> list[dict]:
    """One reading per second; after ``worsen_from`` drift linearly to CRASHING."""
    rng = np.random.default_rng(0)
    readings = []
    for t in range(seconds):
        fraction = 0.0
        if worsen_from is not None and t >= worsen_from:
            fraction = min(1.0, (t - worsen_from) / max(1, seconds - worsen_from))
        reading = {"t": float(t)}
        for vital in VITALS:
            value = STABLE[vital] + fraction * (CRASHING[vital] - STABLE[vital])
            reading[vital] = float(value + rng.normal(0, 0.3))
        readings.append(reading)
    return readings


def test_sanitize_drops_sensor_glitches():
    assert math.isnan(sanitize(0, "spo2"))  # finger off the probe
    assert math.isnan(sanitize(None, "pulse"))
    assert math.isnan(sanitize("abc", "pulse"))
    assert sanitize(97, "spo2") == 97


def test_binning_keeps_silent_steps_as_missing():
    readings = [
        LiveReading(t=0, values={"pulse": 80}),
        LiveReading(t=35, values={"pulse": 90}),
    ]
    binned = bin_readings(readings, step_seconds=10)
    assert list(binned["step"]) == [0, 1, 2, 3]
    assert binned["pulse"].isna().sum() == 2


def test_baseline_gate_and_deviation():
    readings = [LiveReading(t=t, values={"pulse": 80 if t < 60 else 120}) for t in range(0, 120)]
    binned = bin_readings(readings, step_seconds=10)
    assert ready_state(binned)["baseline_ready"]
    columns = ["pulse_last", "pulse_last_dev", "pulse_mean_dev", "spo2_last_dev"]
    row, baseline = live_features(binned, columns)
    assert baseline["pulse"] == pytest.approx(80)
    assert row["pulse_last_dev"].iloc[0] == pytest.approx(40)
    assert row["spo2_last_dev"].iloc[0] == 0.0  # no baseline -> no deviation, as in training


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from vitals_api import app

    return TestClient(app)


def test_live_withholds_score_until_baseline(client):
    response = client.post("/api/live/score", json={"readings": stream(20, 10), "step_seconds": 10})
    assert response.status_code == 200
    body = response.json()
    assert body["ready"] is False
    assert body["risk"] is None
    assert len(body["vitals"]) == len(VITALS)


def test_live_deterioration_scores_higher_than_stable(client):
    stable = client.post("/api/live/score", json={"readings": stream(160, 10), "step_seconds": 10}).json()
    worse = client.post(
        "/api/live/score", json={"readings": stream(160, 10, worsen_from=70), "step_seconds": 10}
    ).json()
    assert stable["ready"] and worse["ready"]
    assert worse["risk"] > stable["risk"]
    assert worse["news_score"] > stable["news_score"]
    assert worse["top_factors"], "an explanation must accompany the score"


def test_live_rejects_bad_step(client):
    assert client.post("/api/live/score", json={"readings": [], "step_seconds": 0}).status_code == 422


def test_live_news_ignores_unmeasured_vitals(client):
    readings = [{"t": float(t), "pulse": 78.0, "spo2": 98.0} for t in range(160)]
    body = client.post("/api/live/score", json={"readings": readings, "step_seconds": 10}).json()
    assert body["news_measured"] == 2
    assert body["news_score"] == 0  # two normal vitals, not a crisis


def test_live_personal_baseline_quiets_chronic_abnormal(client):
    """The thesis in one test: chronically abnormal but stable -> NEWS alarms, model does not."""
    chronic = {"pulse": 112, "sbp": 100, "dbp": 62, "temperature": 37.4, "spo2": 92, "resp_rate": 22}
    readings = [{"t": float(t), **chronic} for t in range(160)]
    body = client.post("/api/live/score", json={"readings": readings, "step_seconds": 10}).json()
    assert body["news_alarm"] and not body["model_alarm"]
    assert body["disagreement"] == "news_only"
