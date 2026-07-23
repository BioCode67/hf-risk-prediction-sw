"""Unit and integration tests for the heart failure risk prediction pipeline."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------


def test_load_raw_data_shapes(loader):
    """Both source datasets load with non-empty, tabular shapes."""
    hf, cardio = loader._heart_failure_raw, loader._cardio_raw
    assert hf is not None and hf.shape[0] > 0
    assert "DEATH_EVENT" in hf.columns
    assert cardio is not None and cardio.shape[0] > 0


def test_preprocess_scales_continuous(loader):
    """Continuous features are standardized to ~zero mean, unit variance."""
    processed = loader.preprocess()
    from data_loader import CONTINUOUS_FEATURES

    for col in CONTINUOUS_FEATURES:
        assert col in processed.features.columns
        assert abs(processed.features[col].mean()) < 1e-6
        assert processed.features[col].std(ddof=0) == pytest.approx(1.0, abs=1e-6)


def test_stratified_split_preserves_ratio(loader):
    """Stratified split preserves the positive-class ratio within tolerance."""
    loader.preprocess()
    split = loader.get_train_test_split(test_size=0.2, random_state=42)

    total = len(split.X_train) + len(split.X_test)
    assert len(split.X_test) == pytest.approx(0.2 * total, abs=1)

    full_ratio = np.mean(loader._processed.target.to_numpy())
    train_ratio = np.mean(split.y_train)
    test_ratio = np.mean(split.y_test)
    assert abs(train_ratio - full_ratio) < 0.05
    assert abs(test_ratio - full_ratio) < 0.05


def test_split_is_deterministic(loader):
    """A fixed random_state yields reproducible splits."""
    loader.preprocess()
    a = loader.get_train_test_split(test_size=0.2, random_state=42)
    b = loader.get_train_test_split(test_size=0.2, random_state=42)
    assert np.array_equal(a.X_test, b.X_test)
    assert np.array_equal(a.y_test, b.y_test)


def test_to_omop_cdm_schema(loader):
    """OMOP CDM export produces valid Person and Measurement tables."""
    omop = loader.to_omop_cdm(loader._heart_failure_raw)

    assert {"person_id", "gender_concept_id", "year_of_birth"} <= set(omop.person.columns)
    assert {
        "person_id",
        "measurement_concept_id",
        "measurement_datetime",
        "value_as_number",
    } <= set(omop.measurement.columns)

    # OMOP standard gender concept ids.
    assert set(omop.person["gender_concept_id"].unique()) <= {8507, 8532}
    assert len(omop.person) == len(loader._heart_failure_raw)
    assert len(omop.measurement) > 0


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------


def test_scale_pos_weight():
    """scale_pos_weight equals the negative/positive ratio."""
    from train import _scale_pos_weight

    y = np.array([0, 0, 0, 1])
    assert _scale_pos_weight(y) == pytest.approx(3.0)
    # No positives -> neutral weight of 1.0.
    assert _scale_pos_weight(np.array([0, 0])) == pytest.approx(1.0)


def test_sensitivity_at_specificity_perfect():
    """A perfectly separable score achieves full sensitivity at high specificity."""
    from train import sensitivity_at_specificity

    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    assert sensitivity_at_specificity(y_true, y_prob, 0.95) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Explainability
# ---------------------------------------------------------------------------


def test_patient_risk_factors_returns_top_3(trained_artifact, split):
    """SHAP patient explanation returns exactly the requested number of drivers."""
    from explainability import patient_risk_factors

    factors = patient_risk_factors(
        trained_artifact["model"],
        split.X_test[:1],
        trained_artifact["feature_names"],
        top_k=3,
    )
    assert len(factors) == 3
    for f in factors:
        assert {"feature", "shap_value", "feature_value"} <= set(f)


# ---------------------------------------------------------------------------
# FastAPI endpoints
# ---------------------------------------------------------------------------


@pytest.fixture()
def client(trained_artifact, monkeypatch):
    """TestClient with the model artifact patched to the lightweight fixture."""
    import main

    monkeypatch.setattr(main, "load_artifact", lambda *a, **k: trained_artifact)
    return TestClient(main.app)


SAMPLE_PATIENT = {
    "age": 65,
    "anaemia": 0,
    "creatinine_phosphokinase": 146,
    "diabetes": 0,
    "ejection_fraction": 20,
    "high_blood_pressure": 1,
    "platelets": 162000,
    "serum_creatinine": 1.3,
    "serum_sodium": 129,
    "sex": 1,
    "smoking": 1,
}


def test_root_and_health(client):
    """Health and root endpoints respond successfully."""
    assert client.get("/").status_code == 200
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"


def test_predict_endpoint(client):
    """/predict returns a calibrated probability and top-3 risk factors."""
    response = client.post("/predict", json=SAMPLE_PATIENT)
    assert response.status_code == 200
    body = response.json()

    assert 0.0 <= body["risk_probability"] <= 1.0
    assert body["risk_label"] in (0, 1)
    assert len(body["top_risk_factors"]) == 3


def test_predict_validation_error(client):
    """Out-of-range input is rejected by request validation."""
    bad = dict(SAMPLE_PATIENT, ejection_fraction=250)
    assert client.post("/predict", json=bad).status_code == 422


def test_convert_omop_endpoint(client):
    """/convert-omop maps patient records to OMOP CDM tables."""
    response = client.post("/convert-omop", json=[SAMPLE_PATIENT])
    assert response.status_code == 200
    body = response.json()

    assert len(body["person"]) == 1
    assert body["person"][0]["gender_concept_id"] in (8507, 8532)
    assert len(body["measurement"]) > 0


def test_convert_omop_empty_rejected(client):
    """An empty batch is rejected with a 400."""
    assert client.post("/convert-omop", json=[]).status_code == 400
