"""Tests for the vital-sign cardiac-arrest early-warning pipeline.

These run entirely on the synthetic cohort, so they need no external data and
stay green in CI (unlike the data-dependent heart-failure tests, which skip).
"""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Synthetic cohort & windowing
# ---------------------------------------------------------------------------


def test_synthetic_cohort_has_cases_and_controls():
    """A mixed cohort contains both arrest and control patients with vitals."""
    from vitals_data import VITALS, generate_synthetic_cohort

    cohort = generate_synthetic_cohort(n_patients=120, arrest_fraction=0.5, seed=0)
    assert set(["patient_id", "hour", *VITALS]).issubset(cohort.vitals.columns)
    arrest = cohort.events["arrest_hour"].notna()
    assert arrest.sum() > 0
    assert (~arrest).sum() > 0


def test_cohort_from_khth_adapter():
    """The KHTH adapter pivots long vitals and anchors labels to CARDT."""
    import pandas as pd

    from vitals_data import VITALS, build_windows, cohort_from_khth, feature_names

    # First measurement 08:00, arrest (CARDT) at 12:00 -> arrest_hour == 4.
    times = ["202301010800", "202301010900", "202301011000", "202301011100"]
    rows = []
    for i, stamp in enumerate(times):
        for code, value in [("HR", 80 + i), ("SBP", 120), ("DBP", 75), ("BT", 36.5), ("SPO2", 98), ("RR", 16)]:
            rows.append({"PATID": "0001", "INDD": "20230101", "VSDT": stamp, "VS_GBN": code, "VS_RSLT": str(value)})
    vital = pd.DataFrame(rows)
    pinfo = pd.DataFrame(
        [{"PATID": "0001", "INDD": "20230101", "AGE": 40, "SEX": "M", "CARDT": "202301011200"}]
    )

    cohort = cohort_from_khth(vital, pinfo)
    assert list(cohort.vitals.columns) == ["patient_id", "hour", *VITALS]
    assert cohort.vitals["hour"].tolist() == [0, 1, 2, 3]
    assert cohort.vitals["pulse"].tolist() == [80, 81, 82, 83]  # long->wide pivot of HR
    arrest_hour = cohort.events.set_index("patient_id").loc["0001_20230101", "arrest_hour"]
    assert arrest_hour == pytest.approx(4.0)

    # The adapted cohort flows through the rest of the pipeline unchanged.
    windowed = build_windows(cohort, observation_window_hours=2, min_valid_fraction=0.5)
    assert list(windowed.features.columns) == feature_names()


def test_case_only_cohort_all_arrest():
    """arrest_fraction=1.0 mirrors the competition's case-only cohort."""
    from vitals_data import generate_synthetic_cohort

    cohort = generate_synthetic_cohort(n_patients=40, arrest_fraction=1.0, seed=1)
    assert cohort.events["arrest_hour"].notna().all()


def test_build_windows_features_and_labels():
    """Windowing yields the declared feature columns and both label classes."""
    from vitals_data import build_windows, feature_names, generate_synthetic_cohort

    windowed = build_windows(generate_synthetic_cohort(n_patients=120, seed=0))
    assert list(windowed.features.columns) == feature_names()
    assert len(windowed.features) == len(windowed.labels) == len(windowed.groups)
    labels = set(windowed.labels.unique())
    assert labels == {0, 1}


def test_window_features_capture_deterioration():
    """Positive (pre-arrest) windows should show higher pulse than negatives."""
    from vitals_data import build_windows, generate_synthetic_cohort

    windowed = build_windows(generate_synthetic_cohort(n_patients=200, seed=3))
    pos = windowed.features[windowed.labels.to_numpy() == 1]["pulse_mean"].mean()
    neg = windowed.features[windowed.labels.to_numpy() == 0]["pulse_mean"].mean()
    assert pos > neg


# ---------------------------------------------------------------------------
# Patient-level split (leakage guard)
# ---------------------------------------------------------------------------


def test_patient_level_split_no_leakage():
    """No patient may appear in both train and test (temporal leakage guard)."""
    from vitals_data import build_windows, generate_synthetic_cohort, patient_level_split

    split = patient_level_split(build_windows(generate_synthetic_cohort(n_patients=120, seed=0)))
    assert set(split.groups_train).isdisjoint(set(split.groups_test))


def test_split_has_no_missing_values():
    """Imputation leaves no NaNs in either split."""
    from vitals_data import build_windows, generate_synthetic_cohort, patient_level_split

    split = patient_level_split(build_windows(generate_synthetic_cohort(n_patients=120, seed=0)))
    assert not split.X_train.isna().any().any()
    assert not split.X_test.isna().any().any()


# ---------------------------------------------------------------------------
# Metrics & NEWS baseline
# ---------------------------------------------------------------------------


def test_scale_pos_weight():
    from vitals_train import scale_pos_weight

    assert scale_pos_weight(np.array([0, 0, 0, 1])) == pytest.approx(3.0)
    assert scale_pos_weight(np.array([0, 0])) == pytest.approx(1.0)


def test_sensitivity_at_specificity_perfect():
    from vitals_train import sensitivity_at_specificity

    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])
    assert sensitivity_at_specificity(y_true, y_score, 0.95) == pytest.approx(1.0)


def test_news_scores_increase_with_deterioration():
    """NEWS assigns a higher score to a deteriorating window than a stable one."""
    import pandas as pd

    from vitals_data import feature_names
    from vitals_train import compute_news_scores

    stable = dict.fromkeys(feature_names(), 0.0)
    stable.update(
        {"resp_rate_last": 16, "spo2_last": 98, "temperature_last": 36.8, "sbp_last": 120, "pulse_last": 75}
    )
    critical = dict.fromkeys(feature_names(), 0.0)
    critical.update(
        {"resp_rate_last": 30, "spo2_last": 88, "temperature_last": 35.0, "sbp_last": 85, "pulse_last": 135}
    )
    frame = pd.DataFrame([stable, critical])[feature_names()]
    scores = compute_news_scores(frame)
    assert scores[1] > scores[0]
    assert scores[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# End-to-end training + explainability
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def trained():
    """Train a small model once for the model-dependent tests."""
    from vitals_data import build_windows, generate_synthetic_cohort, patient_level_split
    from vitals_train import train_xgboost

    split = patient_level_split(build_windows(generate_synthetic_cohort(n_patients=200, seed=7)))
    model, metrics = train_xgboost(split, n_estimators=120)
    return model, metrics, split


def test_xgboost_learns_signal(trained):
    """The model should clearly beat chance on the synthetic deterioration signal."""
    _model, metrics, _split = trained
    assert 0.0 <= metrics.auprc <= 1.0
    assert metrics.roc_auc > 0.75
    assert metrics.auprc > metrics.prevalence  # better than the base rate


def test_shap_window_factors(trained):
    """Per-window SHAP returns the requested number of drivers with expected keys."""
    from vitals_explain import window_risk_factors

    model, _metrics, split = trained
    factors = window_risk_factors(model, split.X_test.iloc[[0]], split.feature_names, top_k=3)
    assert len(factors) == 3
    for factor in factors:
        assert {"feature", "shap_value", "feature_value"} <= set(factor)
