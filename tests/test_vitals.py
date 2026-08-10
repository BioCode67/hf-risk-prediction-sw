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


def test_sanitize_vitals_drops_artifacts_and_fixes_fahrenheit():
    """Sensor artefacts become NaN; Fahrenheit temps are converted to Celsius."""
    import pandas as pd

    from vitals_data import sanitize_vitals

    df = pd.DataFrame(
        {
            "patient_id": ["a"] * 4,
            "hour": [0, 1, 2, 3],
            "pulse": [80, 0, 300, 90],  # 0 (disconnect) and 300 -> NaN
            "sbp": [120, 120, 120, 120],
            "dbp": [70, 70, 70, 70],
            "temperature": [37.0, 99.0, 36.5, 20.0],  # 99 -> °F->°C ≈37.2; 20 (< 30) -> NaN
            "spo2": [98, 29, 97, 96],  # 29 -> NaN
            "resp_rate": [16, 16, 16, 16],
        }
    )
    clean = sanitize_vitals(df)
    assert pd.isna(clean["pulse"]).tolist() == [False, True, True, False]
    assert pd.isna(clean["spo2"]).tolist() == [False, True, False, False]
    assert clean["temperature"].iloc[1] == pytest.approx((99.0 - 32.0) * 5.0 / 9.0)  # ≈37.2 °C
    assert pd.isna(clean["temperature"].iloc[3])  # 20 °C below plausible range


def test_synthetic_cohort_has_demographics():
    from vitals_data import generate_synthetic_cohort

    cohort = generate_synthetic_cohort(n_patients=30, seed=1)
    assert cohort.demographics is not None
    assert set(cohort.demographics.columns) == {"patient_id", "age", "sex"}
    assert len(cohort.demographics) == 30
    assert set(cohort.demographics["sex"].unique()) <= {0, 1}


def test_add_static_features_adds_age_sex():
    from vitals_data import add_static_features, build_windows, generate_synthetic_cohort

    cohort = generate_synthetic_cohort(n_patients=80, seed=2)
    windowed = build_windows(cohort)
    with_static = add_static_features(windowed, cohort)
    assert {"static_age", "static_sex"} <= set(with_static.features.columns)
    assert not with_static.features[["static_age", "static_sex"]].isna().any().any()
    assert len(with_static.feature_names) == len(windowed.feature_names) + 2


def test_khth_adapter_builds_demographics():
    import pandas as pd

    from vitals_data import cohort_from_khth

    vital = pd.DataFrame(
        [{"PATID": "1", "INDD": "20230101", "VSDT": "202301011400", "VS_GBN": "HR", "VS_RSLT": "80"}]
    )
    pinfo = pd.DataFrame([{"PATID": "1", "INDD": "20230101", "AGE": 70, "SEX": "M", "CARDT": "202301011600"}])
    cohort = cohort_from_khth(vital, pinfo)
    assert cohort.demographics is not None
    row = cohort.demographics.set_index("patient_id").loc["1_20230101"]
    assert row["age"] == 70 and row["sex"] == 1  # M -> 1


def test_cohort_from_mimic_adapter():
    """The MIMIC adapter pivots chartevents, converts °F→°C, and labels arrests."""
    import pandas as pd

    from vitals_data import VITALS, cohort_from_mimic

    rows = []
    for hour in range(4):
        stamp = f"2020-01-01 0{hour}:00:00"
        rows += [
            {"stay_id": 10, "charttime": stamp, "itemid": 220045, "valuenum": 80 + hour},  # HR
            {"stay_id": 10, "charttime": stamp, "itemid": 220179, "valuenum": 120},  # SBP
            {"stay_id": 10, "charttime": stamp, "itemid": 220180, "valuenum": 75},  # DBP
            {"stay_id": 10, "charttime": stamp, "itemid": 220277, "valuenum": 98},  # SpO2
            {"stay_id": 10, "charttime": stamp, "itemid": 220210, "valuenum": 16},  # RR
            {"stay_id": 10, "charttime": stamp, "itemid": 223761, "valuenum": 98.6},  # Temp °F -> 37.0 °C
        ]
    chartevents = pd.DataFrame(rows)
    arrests = pd.DataFrame([{"stay_id": 10, "arrest_time": "2020-01-01 04:00:00"}])

    cohort = cohort_from_mimic(chartevents, arrests)
    assert list(cohort.vitals.columns) == ["patient_id", "hour", *VITALS]
    assert cohort.vitals["pulse"].tolist() == [80, 81, 82, 83]
    assert cohort.vitals["temperature"].round(1).tolist() == [37.0, 37.0, 37.0, 37.0]
    arrest_hour = cohort.events.set_index("patient_id").loc["10", "arrest_hour"]
    assert arrest_hour == pytest.approx(4.0)


def test_cohort_from_mimic_controls_have_no_arrest():
    """Stays absent from arrest_events become controls (the case-only fix)."""
    import pandas as pd

    from vitals_data import cohort_from_mimic

    chartevents = pd.DataFrame(
        [{"stay_id": 7, "charttime": "2020-01-01 00:00:00", "itemid": 220045, "valuenum": 75}]
    )
    cohort = cohort_from_mimic(chartevents, arrest_events=None)
    assert cohort.events["arrest_hour"].isna().all()


def test_add_personalized_baseline_features():
    """Personalized deviation features are added and flag pre-arrest deterioration."""
    from vitals_data import VITALS, add_personalized_features, build_windows, generate_synthetic_cohort

    cohort = generate_synthetic_cohort(n_patients=150, seed=5)
    windowed = build_windows(cohort)
    personalized = add_personalized_features(windowed, cohort)

    assert len(personalized.feature_names) == len(windowed.feature_names) + 2 * len(VITALS)
    dev_cols = [c for c in personalized.feature_names if c.endswith("_dev")]
    assert "pulse_last_dev" in dev_cols
    assert not personalized.features[dev_cols].isna().any().any()

    # Pre-arrest windows deviate further from personal baseline than stable ones.
    y = personalized.labels.to_numpy()
    assert personalized.features[y == 1]["pulse_last_dev"].mean() > personalized.features[y == 0]["pulse_last_dev"].mean()


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


def test_threshold_and_alarm_burden_at_sensitivity():
    """At a matched sensitivity, alarm burden is reported at the right operating point."""
    from vitals_train import alarm_burden, threshold_at_sensitivity

    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_score = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    assert threshold_at_sensitivity(y_true, y_score, 1.0) <= 0.6  # must catch all positives

    burden = alarm_burden(y_true, y_score, target_sensitivity=1.0)
    assert burden["sensitivity"] == pytest.approx(1.0)
    assert burden["specificity"] == pytest.approx(1.0)  # perfectly separable
    assert burden["alarms_per_100_windows"] == pytest.approx(50.0)  # only the 4 positives


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


def test_tune_xgboost_returns_usable_params(trained):
    """Optuna search returns hyperparameters that train_xgboost accepts directly."""
    from vitals_train import train_xgboost, tune_xgboost

    _model, _metrics, split = trained
    best = tune_xgboost(split, n_trials=2, n_splits=2, use_gpu=False, seed=0)
    assert isinstance(best, dict)
    assert {"n_estimators", "learning_rate", "max_depth"} <= best.keys()
    # The returned params must be consumable as overrides without collisions.
    _model2, metrics = train_xgboost(split, **best)
    assert 0.0 <= metrics.auprc <= 1.0


def test_time_to_arrest_is_recorded():
    """Positive windows carry a small positive time-to-arrest; controls are NaN."""
    import numpy as np

    from vitals_data import build_windows, generate_synthetic_cohort

    windowed = build_windows(generate_synthetic_cohort(n_patients=150, seed=9))
    assert windowed.time_to_arrest is not None
    tta = windowed.time_to_arrest.to_numpy()
    positives = tta[windowed.labels.to_numpy() == 1]
    assert np.all(positives > 0) and np.all(positives <= 1)  # arrest within the 1h horizon


def test_lead_time_summary(trained):
    """Lead-time reports a plausible detection rate and non-negative lead time."""
    from vitals_train import lead_time_summary, threshold_at_specificity

    model, _metrics, split = trained
    score = model.predict_proba(split.X_test)[:, 1]
    summary = lead_time_summary(split, score, threshold_at_specificity(split.y_test, score))
    assert summary["arrest_patients"] > 0
    assert 0.0 <= summary["detection_rate"] <= 1.0
    assert summary["median_lead_time_h"] >= 0.0


# ---------------------------------------------------------------------------
# Model zoo & head-to-head comparison
# ---------------------------------------------------------------------------


def test_available_models_includes_xgboost():
    """The registry always offers XGBoost; the rest depend on what is installed."""
    from vitals_train import MODEL_NAMES, available_models

    usable = available_models()
    assert "xgboost" in usable
    assert set(usable) <= set(MODEL_NAMES)


def test_build_model_rejects_unknown_name():
    """A typo'd model name fails loudly instead of silently falling back."""
    from vitals_train import build_model

    with pytest.raises(ValueError, match="Unknown model"):
        build_model("randomforest", np.array([0, 1]))


@pytest.mark.parametrize("name", ["logistic", "random_forest"])
def test_alternative_models_learn_signal(trained, name):
    """The non-XGBoost learners fit, emit probabilities, and beat the base rate."""
    from vitals_train import train_model

    _model, _metrics, split = trained
    model, metrics = train_model(name, split)
    score = model.predict_proba(split.X_test)[:, 1]
    assert score.shape == (len(split.y_test),)
    assert np.all((score >= 0.0) & (score <= 1.0))
    assert metrics.auprc > metrics.prevalence  # better than always-alarm


def test_compare_models_ranks_by_auprc(trained):
    """Comparison returns one row per model plus NEWS, sorted best-AUPRC first."""
    from vitals_train import compare_models

    _model, _metrics, split = trained
    results = compare_models(split, models=["logistic", "random_forest"], target_sensitivity=0.9)

    names = [result.name for result in results]
    assert set(names) == {"logistic", "random_forest", "news"}
    auprcs = [result.metrics.auprc for result in results]
    assert auprcs == sorted(auprcs, reverse=True)

    for result in results:
        assert 0.0 <= result.alarm_burden["alarms_per_100_windows"] <= 100.0
        assert result.alarm_burden["sensitivity"] >= 0.9 - 1e-9  # the point is matched
        assert set(result.summary()) == {
            "model",
            "key",
            "metrics",
            "alarm_burden",
            "lead_time",
            "fit_seconds",
        }


def test_compare_models_skips_missing_packages(trained, monkeypatch):
    """A model whose package is absent is skipped, not fatal — the 안심존 case."""
    import vitals_train

    _model, _metrics, split = trained
    real_build = vitals_train.build_model

    def fake_build(name, y_train, **kwargs):
        if name == "logistic":
            raise vitals_train.ModelUnavailable("pretend scikit-learn is missing")
        return real_build(name, y_train, **kwargs)

    monkeypatch.setattr(vitals_train, "build_model", fake_build)
    results = vitals_train.compare_models(split, models=["logistic", "random_forest"])
    assert [result.name for result in results] != []
    assert "logistic" not in {result.name for result in results}


# ---------------------------------------------------------------------------
# F-score metrics
# ---------------------------------------------------------------------------


def test_f_score_matches_sklearn():
    """f_score reproduces sklearn's fbeta on the same precision/recall pair."""
    from sklearn.metrics import fbeta_score

    from vitals_train import f_score

    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 0])
    y_pred = np.array([0, 1, 1, 0, 0, 1, 0, 1])
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    precision = tp / y_pred.sum()
    recall = tp / y_true.sum()
    for beta in (1.0, 2.0):
        assert f_score(precision, recall, beta) == pytest.approx(
            fbeta_score(y_true, y_pred, beta=beta)
        )


def test_f_score_handles_no_alarms():
    """A model that never alarms scores 0, not a ZeroDivisionError."""
    from vitals_train import f_score

    assert f_score(0.0, 0.0) == 0.0


def test_best_f_score_finds_the_optimal_threshold(trained):
    """The swept best-F1 dominates the F1 read at any single fixed threshold."""
    from sklearn.metrics import f1_score

    from vitals_train import best_f_score

    model, _metrics, split = trained
    score = model.predict_proba(split.X_test)[:, 1]
    best = best_f_score(split.y_test, score)

    assert 0.0 <= best["f_score"] <= 1.0
    # It is the maximum, so no other cutoff may beat it.
    for cutoff in (0.1, 0.3, 0.5, 0.7, 0.9):
        assert f1_score(split.y_test, (score >= cutoff).astype(int)) <= best["f_score"] + 1e-9
    # And the reported threshold really does reproduce the reported score.
    at_best = f1_score(split.y_test, (score >= best["threshold"]).astype(int))
    assert at_best == pytest.approx(best["f_score"], abs=1e-9)


def test_best_f2_weights_recall_above_f1(trained):
    """F2 peaks at a lower threshold than F1 — it buys recall with precision."""
    from vitals_train import best_f_score

    model, _metrics, split = trained
    score = model.predict_proba(split.X_test)[:, 1]
    f1, f2 = best_f_score(split.y_test, score, 1.0), best_f_score(split.y_test, score, 2.0)
    assert f2["recall"] >= f1["recall"]


def test_metrics_carry_both_f1_flavours(trained):
    """evaluate() reports F1 at the operating point and the best F1 anywhere."""
    _model, metrics, _split = trained
    assert 0.0 <= metrics.f1_at_threshold <= 1.0
    # The 95%-specificity point is one cutoff among many, so it cannot beat the best.
    assert metrics.f1_at_threshold <= metrics.best_f1 + 1e-9
    # The always-alarm floor is 2p/(1+p); a useful model has to clear it.
    expected_floor = 2 * metrics.prevalence / (1 + metrics.prevalence)
    assert metrics.f1_all_alarm_baseline == pytest.approx(expected_floor)
    assert metrics.best_f1 > metrics.f1_all_alarm_baseline


def test_compare_models_can_rank_by_f1(trained):
    """--rank-by f1 sorts by best-F1; alarm burden gains a matching F1 column."""
    from vitals_train import compare_models, rank_key

    _model, _metrics, split = trained
    results = compare_models(
        split, models=["logistic", "random_forest"], rank_by="f1", target_sensitivity=0.9
    )
    scores = [result.metrics.best_f1 for result in results]
    assert scores == sorted(scores, reverse=True)
    for result in results:
        assert 0.0 <= result.alarm_burden["f1"] <= 1.0

    with pytest.raises(ValueError, match="Unknown ranking metric"):
        rank_key("accuracy")


def test_tune_xgboost_accepts_f1_objective(trained):
    """The Optuna search can maximize best-F1 instead of AUPRC."""
    from vitals_train import train_xgboost, tune_xgboost

    _model, _metrics, split = trained
    best = tune_xgboost(split, n_trials=2, n_splits=2, seed=0, metric="f1")
    _model2, metrics = train_xgboost(split, **best)
    assert 0.0 <= metrics.best_f1 <= 1.0

    with pytest.raises(ValueError, match="Unknown tuning metric"):
        tune_xgboost(split, n_trials=1, n_splits=2, metric="accuracy")


def test_shap_window_factors(trained):
    """Per-window SHAP returns the requested number of drivers with expected keys."""
    from vitals_explain import window_risk_factors

    model, _metrics, split = trained
    factors = window_risk_factors(model, split.X_test.iloc[[0]], split.feature_names, top_k=3)
    assert len(factors) == 3
    for factor in factors:
        assert {"feature", "shap_value", "feature_value"} <= set(factor)
