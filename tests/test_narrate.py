"""Tests for the alarm-explanation layer (``vitals_narrate``).

Only the deterministic half is covered: ``build_evidence`` / ``format_evidence``
run on the synthetic cohort with no network and no API key. ``narrate`` itself
is exercised only for the guard that blocks credentialed cohorts, which fails
before any request is made.
"""

from __future__ import annotations

import numpy as np
import pytest

from vitals_data import (
    add_personalized_features,
    build_windows,
    generate_synthetic_cohort,
    patient_level_split,
)
from vitals_narrate import (
    build_evidence,
    describe_feature,
    format_evidence,
    narrate,
)
from vitals_train import train_xgboost


@pytest.fixture(scope="module")
def fitted():
    """A trained model plus its test windows, from the synthetic cohort."""
    cohort = generate_synthetic_cohort(n_patients=60, seed=0)
    windowed = add_personalized_features(build_windows(cohort), cohort)
    split = patient_level_split(windowed)
    model, _ = train_xgboost(split)
    return model, split.X_test


@pytest.fixture(scope="module")
def evidence(fitted):
    model, X = fitted
    riskiest = int(np.argmax(model.predict_proba(X)[:, 1]))
    return build_evidence(model, X, riskiest, source="synthetic", patient_id=7)


def test_describe_feature_covers_plain_stat_and_personal_deviation():
    assert describe_feature("resp_rate_slope") == ("호흡수 시간당 변화 추세", "회/분")
    name, unit = describe_feature("spo2_mean_dev")
    assert "개인 기저선 대비" in name
    assert unit == "%"


def test_describe_feature_passes_unknown_names_through():
    assert describe_feature("not_a_feature") == ("not_a_feature", "")


def test_build_evidence_reports_a_decision_and_ranked_factors(evidence):
    assert 0.0 <= evidence["risk"] <= 1.0
    assert evidence["model_alarm"] == (evidence["risk"] >= evidence["threshold"])
    assert evidence["news_alarm"] == (evidence["news_score"] >= evidence["news_threshold"])
    assert evidence["patient_id"] == "7"

    factors = evidence["top_factors"]
    assert len(factors) == 3
    assert [abs(f["shap"]) for f in factors] == sorted((abs(f["shap"]) for f in factors), reverse=True)
    for factor in factors:
        assert factor["effect"] == ("위험을 높임" if factor["shap"] > 0 else "위험을 낮춤")


def test_build_evidence_recovers_the_personal_baseline(fitted):
    """baseline = current - deviation, so it must match the ``_dev`` column."""
    model, X = fitted
    evidence = build_evidence(model, X, 0, source="synthetic")
    expected = float(X["pulse_last"].iloc[0]) - float(X["pulse_last_dev"].iloc[0])
    assert evidence["personal_baseline"]["맥박"] == pytest.approx(expected, abs=0.05)


def test_build_evidence_respects_the_alarm_threshold(fitted):
    model, X = fitted
    assert build_evidence(model, X, 0, threshold=0.0, source="synthetic")["model_alarm"]
    assert not build_evidence(model, X, 0, threshold=1.1, source="synthetic")["model_alarm"]


def test_format_evidence_mentions_every_top_factor(evidence):
    rendered = format_evidence(evidence)
    for factor in evidence["top_factors"]:
        assert factor["description"] in rendered
    assert "경보" in rendered


def test_narrate_refuses_credentialed_cohorts(evidence):
    """MIMIC / KHTH are under a DUA that forbids third-party sharing."""
    for source in ("mimic", "khth"):
        with pytest.raises(PermissionError, match=source):
            narrate({**evidence, "source": source})
