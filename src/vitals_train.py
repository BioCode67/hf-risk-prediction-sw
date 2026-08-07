"""Cardiac-arrest early-warning training: XGBoost vs. NEWS baseline.

The clinical thesis of the project is *false-alarm reduction*, so evaluation
does not stop at AUROC. We report AUPRC (rare positives), sensitivity at a fixed
95% specificity, and operating-point false-alarm metrics, and we compare the
cost-sensitive gradient-boosting model against NEWS — the National Early Warning
Score used at the bedside — as the clinically meaningful baseline to beat.

Training here runs on the synthetic cohort so the pipeline is self-contained;
inside the 안심존 the same functions consume real ``KHTH`` data adapted via
``vitals_data.cohort_from_khth``.
"""

from __future__ import annotations

import json
import pickle
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from xgboost import XGBClassifier
from xgboost.core import XGBoostError

from vitals_data import (
    build_windows,
    generate_synthetic_cohort,
    patient_level_split,
    PatientSplit,
)


@dataclass
class EarlyWarningMetrics:
    """Discrimination and false-alarm metrics at a fixed-specificity operating point."""

    model_name: str
    auprc: float
    roc_auc: float
    sensitivity_at_95_specificity: float
    precision_at_threshold: float
    recall_at_threshold: float
    specificity_at_threshold: float
    false_alarm_rate: float
    alarms_per_100_windows: float
    threshold: float
    prevalence: float
    confusion_matrix: list[list[int]]


def scale_pos_weight(y: np.ndarray) -> float:
    """Cost-sensitive weight for the positive class (negatives / positives)."""
    positives = int(np.sum(y == 1))
    negatives = int(np.sum(y == 0))
    if positives == 0:
        return 1.0
    return negatives / positives


def sensitivity_at_specificity(
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_specificity: float = 0.95,
) -> float:
    """Best achievable sensitivity while specificity >= the target (vectorized)."""
    y_true = np.asarray(y_true)
    if np.unique(y_true).size < 2:
        return 0.0
    fpr, tpr, _ = roc_curve(y_true, y_score)
    specificity = 1.0 - fpr
    mask = specificity >= target_specificity
    return float(tpr[mask].max()) if mask.any() else 0.0


def threshold_at_specificity(
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_specificity: float = 0.95,
) -> float:
    """Threshold meeting the target specificity while maximizing sensitivity."""
    scores = np.asarray(y_score)
    flag_nothing = float(scores.max()) + 1e-9
    if np.unique(np.asarray(y_true)).size < 2:
        return flag_nothing
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    mask = (1.0 - fpr) >= target_specificity
    if not mask.any():
        return flag_nothing
    threshold = thresholds[int(np.argmax(np.where(mask, tpr, -1.0)))]
    return float(threshold) if np.isfinite(threshold) else flag_nothing


def threshold_at_sensitivity(
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_sensitivity: float = 0.90,
) -> float:
    """Threshold meeting the target sensitivity while maximizing specificity (fewest alarms)."""
    scores = np.asarray(y_score)
    flag_everything = float(scores.min())
    if np.unique(np.asarray(y_true)).size < 2:
        return flag_everything
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    mask = tpr >= target_sensitivity
    if not mask.any():
        return flag_everything
    threshold = thresholds[int(np.argmax(np.where(mask, 1.0 - fpr, -1.0)))]
    return float(threshold) if np.isfinite(threshold) else flag_everything


def alarm_burden(
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_sensitivity: float = 0.90,
) -> dict[str, float]:
    """Operating point at a *matched sensitivity*: the alarm burden it costs.

    This is the clinical framing of the thesis — hold detection fixed and compare
    how many (false) alarms each model raises. Reports specificity, false-alarm
    rate and alarms per 100 windows at the threshold that just achieves the target
    sensitivity.
    """
    threshold = threshold_at_sensitivity(y_true, y_score, target_sensitivity)
    pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    return {
        "target_sensitivity": float(target_sensitivity),
        "threshold": float(threshold),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "false_alarm_rate": float(1.0 - precision) if (tp + fp) > 0 else 0.0,
        "alarms_per_100_windows": float(100.0 * pred.mean()),
    }


def evaluate(
    model_name: str,
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_specificity: float = 0.95,
) -> EarlyWarningMetrics:
    """Compute discrimination + false-alarm metrics at a fixed-specificity point."""
    threshold = threshold_at_specificity(y_true, y_score, target_specificity)
    pred = (y_score >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return EarlyWarningMetrics(
        model_name=model_name,
        auprc=float(average_precision_score(y_true, y_score)),
        roc_auc=float(roc_auc_score(y_true, y_score)),
        sensitivity_at_95_specificity=sensitivity_at_specificity(y_true, y_score, target_specificity),
        precision_at_threshold=float(precision),
        recall_at_threshold=float(recall),
        specificity_at_threshold=float(specificity),
        false_alarm_rate=float(1.0 - precision) if (tp + fp) > 0 else 0.0,
        alarms_per_100_windows=float(100.0 * pred.mean()),
        threshold=float(threshold),
        prevalence=float(np.mean(y_true)),
        confusion_matrix=cm.tolist(),
    )


# --- NEWS baseline --------------------------------------------------------


def _news_resp_rate(x: np.ndarray) -> np.ndarray:
    return np.select([x <= 8, x <= 11, x <= 20, x <= 24], [3, 1, 0, 2], default=3)


def _news_spo2(x: np.ndarray) -> np.ndarray:
    return np.select([x <= 91, x <= 93, x <= 95], [3, 2, 1], default=0)


def _news_temperature(x: np.ndarray) -> np.ndarray:
    return np.select([x <= 35.0, x <= 36.0, x <= 38.0, x <= 39.0], [3, 1, 0, 1], default=2)


def _news_sbp(x: np.ndarray) -> np.ndarray:
    return np.select([x <= 90, x <= 100, x <= 110, x <= 219], [3, 2, 1, 0], default=3)


def _news_pulse(x: np.ndarray) -> np.ndarray:
    return np.select([x <= 40, x <= 50, x <= 90, x <= 110, x <= 130], [3, 1, 0, 1, 2], default=3)


def compute_news_scores(X: pd.DataFrame) -> np.ndarray:
    """Aggregate NEWS score from the window's last vital values.

    Uses the five vitals available in this dataset (respiratory rate, SpO2,
    temperature, systolic BP, pulse); the consciousness and supplemental-oxygen
    components of full NEWS2 are not recorded here. Higher = sicker.
    """
    return (
        _news_resp_rate(X["resp_rate_last"].to_numpy())
        + _news_spo2(X["spo2_last"].to_numpy())
        + _news_temperature(X["temperature_last"].to_numpy())
        + _news_sbp(X["sbp_last"].to_numpy())
        + _news_pulse(X["pulse_last"].to_numpy())
    ).astype(float)


# --- Models ---------------------------------------------------------------


def train_xgboost(
    split: PatientSplit, use_gpu: bool = False, **overrides: Any
) -> tuple[XGBClassifier, EarlyWarningMetrics]:
    """Train a cost-sensitive XGBoost early-warning classifier.

    ``use_gpu=True`` runs training on an NVIDIA GPU (CUDA) — worth it at the full
    MIMIC-IV / KHTH scale and for many-trial Optuna search, not for the tiny
    synthetic dev cohort. It falls back to CPU automatically when no GPU is
    visible, so the same call works on a laptop and on the A6000 server alike.
    Explicit ``device=``/``tree_method=`` overrides win over ``use_gpu``.
    """
    params: dict[str, Any] = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "random_state": 42,
        "scale_pos_weight": scale_pos_weight(split.y_train),
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 5,
        "min_child_weight": 2,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "verbosity": 0,
    }
    if use_gpu:
        params.update({"device": "cuda", "tree_method": "hist"})
    params.update(overrides)

    try:
        model = XGBClassifier(**params)
        model.fit(split.X_train, split.y_train)
    except XGBoostError as exc:
        if params.get("device") != "cuda":
            raise  # a real failure, not a missing GPU
        print(f"[gpu] CUDA unavailable ({exc}); falling back to CPU.")
        params.update({"device": "cpu", "tree_method": "hist"})
        model = XGBClassifier(**params)
        model.fit(split.X_train, split.y_train)
    y_score = model.predict_proba(split.X_test)[:, 1]
    return model, evaluate("XGBoost", split.y_test, y_score)


def tune_xgboost(
    split: PatientSplit,
    n_trials: int = 30,
    n_splits: int = 4,
    use_gpu: bool = False,
    seed: int = 42,
) -> dict[str, Any]:
    """Optuna search for XGBoost hyperparameters, maximizing patient-level CV AUPRC.

    Folds are grouped by patient (``GroupKFold`` over ``groups_train``) so no
    patient's windows leak between train and validation — the same leakage guard
    as the outer split. The objective is **AUPRC** (not AUROC), keeping the search
    aligned with the rare-event / false-alarm focus of the project. Returns the
    best hyperparameters as a plain dict, ready to splat into ``train_xgboost`` or
    ``train`` as overrides. ``use_gpu=True`` runs each trial on CUDA (A6000) and
    falls back to a CPU search when no GPU is visible.
    """
    import optuna
    from sklearn.model_selection import GroupKFold

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    X, y, groups = split.X_train, split.y_train, split.groups_train
    spw = scale_pos_weight(y)
    device_params = {"device": "cuda", "tree_method": "hist"} if use_gpu else {}
    n_splits = max(2, min(n_splits, int(np.unique(groups).size)))

    def objective(trial: optuna.Trial) -> float:
        params: dict[str, Any] = {
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "random_state": seed,
            "scale_pos_weight": spw,
            "verbosity": 0,
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            **device_params,
        }
        fold_scores: list[float] = []
        for train_idx, val_idx in GroupKFold(n_splits=n_splits).split(X, y, groups):
            model = XGBClassifier(**params)
            model.fit(X.iloc[train_idx], y[train_idx])
            proba = model.predict_proba(X.iloc[val_idx])[:, 1]
            fold_scores.append(float(average_precision_score(y[val_idx], proba)))
        return float(np.mean(fold_scores))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    try:
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    except XGBoostError as exc:
        if not use_gpu:
            raise
        print(f"[gpu] CUDA unavailable ({exc}); re-running the search on CPU.")
        return tune_xgboost(split, n_trials=n_trials, n_splits=n_splits, use_gpu=False, seed=seed)
    print(f"Optuna: best CV AUPRC={study.best_value:.4f} over {len(study.trials)} trials")
    return dict(study.best_params)


def evaluate_news_baseline(split: PatientSplit) -> EarlyWarningMetrics:
    """Score the test windows with NEWS as the clinical baseline."""
    scores = compute_news_scores(split.X_test)
    return evaluate("NEWS", split.y_test, scores)


# --- Model zoo ------------------------------------------------------------
#
# Alternative learners scored against XGBoost on the *same* windows and the same
# patient-level split. The registry deliberately spans three inductive biases
# rather than three gradient boosters under different names:
#
#   boosting  — xgboost, lightgbm, catboost  (sequential, residual-fitting trees)
#   bagging   — random_forest                (averaged deep trees, lower variance)
#   linear    — logistic                     (no interactions unless we build them)
#
# The linear arm is the one that earns its place scientifically: if it matches
# the boosters, the hand-crafted window features already carry the signal and the
# non-linearity is not what is doing the work. Every model is made cost-sensitive
# — at a ~1% positive rate an unweighted fit finds "never alarm" near-optimal.

MODEL_NAMES: Final[tuple[str, ...]] = (
    "xgboost",
    "lightgbm",
    "catboost",
    "random_forest",
    "logistic",
)

_MODEL_LABELS: Final[dict[str, str]] = {
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "catboost": "CatBoost",
    "random_forest": "RandomForest",
    "logistic": "LogisticRegression",
}


class ModelUnavailable(RuntimeError):
    """A registered model's package is not importable in this environment.

    The 안심존 declares its packages up front, so a learner that is present on the
    dev server can be missing inside the enclave. Comparison runs report and skip
    these rather than aborting the whole evaluation.
    """


def build_model(name: str, y_train: np.ndarray, use_gpu: bool = False, **overrides: Any) -> Any:
    """Construct an unfitted, cost-sensitive classifier by registry name.

    Every estimator exposes ``fit``/``predict_proba``, so downstream evaluation is
    byte-identical across models. Imbalance is handled the way each library
    expects (``scale_pos_weight`` for the boosters, ``class_weight`` for the
    scikit-learn models). ``use_gpu`` is honoured only where a stock pip wheel
    supports it (XGBoost via CUDA, CatBoost via ``task_type``); LightGBM's GPU
    build is not pip-installable and the sklearn models are CPU-only, so those
    silently stay on CPU. ``overrides`` win over every default.
    """
    key = name.lower()
    spw = scale_pos_weight(y_train)

    if key == "xgboost":
        params: dict[str, Any] = {
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "random_state": 42,
            "scale_pos_weight": spw,
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 5,
            "min_child_weight": 2,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "verbosity": 0,
        }
        if use_gpu:
            params.update({"device": "cuda", "tree_method": "hist"})
        params.update(overrides)
        return XGBClassifier(**params)

    if key == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ModelUnavailable("lightgbm is not installed") from exc
        params = {
            "objective": "binary",
            "random_state": 42,
            "scale_pos_weight": spw,
            "n_estimators": 300,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 20,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "n_jobs": -1,
            "verbosity": -1,
        }
        params.update(overrides)
        return LGBMClassifier(**params)

    if key == "catboost":
        try:
            from catboost import CatBoostClassifier
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ModelUnavailable("catboost is not installed") from exc
        params = {
            "iterations": 300,
            "depth": 6,
            "learning_rate": 0.05,
            "loss_function": "Logloss",
            "eval_metric": "PRAUC",
            "scale_pos_weight": spw,
            "random_seed": 42,
            "verbose": 0,
            # CatBoost otherwise litters the working directory with catboost_info/.
            "allow_writing_files": False,
        }
        if use_gpu:
            params["task_type"] = "GPU"
        params.update(overrides)
        return CatBoostClassifier(**params)

    if key == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        params = {
            "n_estimators": 400,
            "min_samples_leaf": 5,
            "max_features": "sqrt",
            "class_weight": "balanced_subsample",
            "n_jobs": -1,
            "random_state": 42,
        }
        params.update(overrides)
        return RandomForestClassifier(**params)

    if key == "logistic":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        params = {
            "C": 1.0,
            "class_weight": "balanced",
            "max_iter": 2000,
            "solver": "lbfgs",
            "random_state": 42,
        }
        params.update(overrides)
        # Window features span wildly different scales (SpO2 ~98, slopes ~0.1);
        # without standardization lbfgs converges slowly and the coefficients are
        # not comparable to each other.
        return Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(**params))])

    raise ValueError(f"Unknown model {name!r}; expected one of {', '.join(MODEL_NAMES)}")


def available_models(names: Sequence[str] | None = None) -> list[str]:
    """Registry names whose backing package is importable here, order preserved."""
    probe = np.array([0, 1])
    usable: list[str] = []
    for name in names if names is not None else MODEL_NAMES:
        try:
            build_model(name, probe)
        except ModelUnavailable:
            continue
        usable.append(name.lower())
    return usable


def train_model(
    name: str, split: PatientSplit, use_gpu: bool = False, **overrides: Any
) -> tuple[Any, EarlyWarningMetrics]:
    """Fit one registry model on the split and evaluate it exactly like the others.

    ``name="xgboost"`` routes through :func:`train_xgboost` so it keeps that
    function's CUDA-to-CPU fallback; the remaining models have no GPU path to
    fall back from.
    """
    key = name.lower()
    if key == "xgboost":
        return train_xgboost(split, use_gpu=use_gpu, **overrides)
    model = build_model(key, split.y_train, use_gpu=use_gpu, **overrides)
    model.fit(split.X_train, split.y_train)
    y_score = model.predict_proba(split.X_test)[:, 1]
    return model, evaluate(_MODEL_LABELS[key], split.y_test, y_score)


def lead_time_summary(
    split: PatientSplit,
    y_score: np.ndarray,
    threshold: float,
    max_lookback_hours: float = 48.0,
) -> dict[str, float]:
    """How early the model alarms before arrest, among test arrest patients.

    For each arrest patient, the lead time is the largest time-to-arrest among
    their windows that cross ``threshold`` within ``max_lookback_hours`` (i.e. the
    earliest alarm). Reports detection rate and median/mean lead time — the metric
    that captures the whole point of *early* warning, and which (unlike raw
    specificity) does not depend on control patients.
    """
    if split.time_to_arrest_test is None:
        return {}
    frame = pd.DataFrame(
        {"patient": split.groups_test, "tta": split.time_to_arrest_test, "score": y_score}
    )
    arrest = frame[frame["tta"].notna() & (frame["tta"] >= 0)]
    n_patients = int(arrest["patient"].nunique())
    lead_times: list[float] = []
    for _, group in arrest.groupby("patient"):
        alarms = group[(group["score"] >= threshold) & (group["tta"] <= max_lookback_hours)]
        if len(alarms):
            lead_times.append(float(alarms["tta"].max()))
    detected = len(lead_times)
    return {
        "arrest_patients": float(n_patients),
        "detected": float(detected),
        "detection_rate": float(detected / n_patients) if n_patients else 0.0,
        "median_lead_time_h": float(np.median(lead_times)) if lead_times else 0.0,
        "mean_lead_time_h": float(np.mean(lead_times)) if lead_times else 0.0,
    }


# --- Head-to-head comparison ----------------------------------------------


@dataclass
class ModelResult:
    """One model's fitted estimator plus its numbers at both operating points.

    Two operating points are carried on purpose. ``metrics`` fixes *specificity*
    at 95% and reports the sensitivity that buys; ``alarm_burden`` fixes
    *sensitivity* and reports the alarms that buys. The second is the one the
    project's false-alarm thesis rests on — hold detection equal, compare cost.
    """

    name: str
    metrics: EarlyWarningMetrics
    alarm_burden: dict[str, float] = field(default_factory=dict)
    lead_time: dict[str, float] = field(default_factory=dict)
    fit_seconds: float = 0.0
    estimator: Any = None  # None for the rule-based NEWS baseline

    def summary(self) -> dict[str, Any]:
        """JSON-serializable view of this row (drops the fitted estimator)."""
        return {
            "model": self.metrics.model_name,
            "key": self.name,
            "metrics": asdict(self.metrics),
            "alarm_burden": self.alarm_burden,
            "lead_time": self.lead_time,
            "fit_seconds": round(self.fit_seconds, 3),
        }


def _score_result(
    name: str,
    label: str,
    split: PatientSplit,
    y_score: np.ndarray,
    target_sensitivity: float,
    fit_seconds: float = 0.0,
    estimator: Any = None,
) -> ModelResult:
    """Wrap one model's test-set scores into the common comparison record."""
    threshold = threshold_at_sensitivity(split.y_test, y_score, target_sensitivity)
    return ModelResult(
        name=name,
        metrics=evaluate(label, split.y_test, y_score),
        alarm_burden=alarm_burden(split.y_test, y_score, target_sensitivity),
        lead_time=lead_time_summary(split, y_score, threshold),
        fit_seconds=fit_seconds,
        estimator=estimator,
    )


def compare_models(
    split: PatientSplit,
    models: Sequence[str] | None = None,
    use_gpu: bool = False,
    target_sensitivity: float = 0.90,
    include_news: bool = True,
) -> list[ModelResult]:
    """Train several learners on one split and rank them, best AUPRC first.

    Every model sees identical windows, identical features and the identical
    patient-level split, so any difference is the learner's and nothing else's.
    NEWS is included as the bedside rule-based floor unless ``include_news`` is
    off. Models whose package is missing are reported and skipped rather than
    raising, so the same call works on the dev server and inside the 안심존.

    Every arm trains on its registry defaults on purpose: tuning one learner and
    not the others decides the ranking by search budget rather than by model.
    Tune afterwards, on whichever arm wins.
    """
    requested = list(models) if models is not None else list(MODEL_NAMES)
    results: list[ModelResult] = []

    for name in requested:
        key = name.lower()
        started = time.perf_counter()
        try:
            model, _ = train_model(key, split, use_gpu=use_gpu)
        except ModelUnavailable as exc:
            print(f"[skip] {key}: {exc}")
            continue
        elapsed = time.perf_counter() - started
        y_score = model.predict_proba(split.X_test)[:, 1]
        results.append(
            _score_result(key, _MODEL_LABELS[key], split, y_score, target_sensitivity, elapsed, model)
        )

    if include_news:
        news_score = compute_news_scores(split.X_test)
        results.append(_score_result("news", "NEWS", split, news_score, target_sensitivity))

    return sorted(results, key=lambda result: result.metrics.auprc, reverse=True)


def print_comparison(results: Sequence[ModelResult], target_sensitivity: float = 0.90) -> None:
    """Print the head-to-head table: discrimination, then the cost of equal detection."""
    percent = int(round(100 * target_sensitivity))
    header = (
        f"{'model':<20}{'AUPRC':>8}{'ROC-AUC':>9}{'sens@95spec':>13}"
        f"{'alarms/100':>12}{'FA rate':>9}{'lead-h':>8}{'fit(s)':>8}"
    )
    print(f"\n=== Model comparison — alarm burden at a matched {percent}% sensitivity ===")
    print(header)
    print("-" * len(header))
    for result in results:
        metrics = result.metrics
        print(
            f"{metrics.model_name:<20}{metrics.auprc:>8.4f}{metrics.roc_auc:>9.4f}"
            f"{metrics.sensitivity_at_95_specificity:>13.4f}"
            f"{result.alarm_burden.get('alarms_per_100_windows', 0.0):>12.2f}"
            f"{result.alarm_burden.get('false_alarm_rate', 0.0):>9.4f}"
            f"{result.lead_time.get('median_lead_time_h', 0.0):>8.2f}"
            f"{result.fit_seconds:>8.2f}"
        )
    print(
        "\nalarms/100 and FA rate are read at the threshold that just reaches the "
        f"{percent}% sensitivity above — lower is better at equal detection."
    )


def save_artifact(
    model: XGBClassifier,
    split: PatientSplit,
    metrics: EarlyWarningMetrics,
    output_path: Path,
) -> None:
    """Persist the trained model plus everything needed to score new windows."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": model,
        "feature_names": split.feature_names,
        "imputation_medians": split.imputation_medians,
        "metrics": asdict(metrics),
    }
    with output_path.open("wb") as handle:
        pickle.dump(artifact, handle)


def _print_metrics(metrics: EarlyWarningMetrics) -> None:
    print(f"=== {metrics.model_name} ===")
    print(f"  AUPRC: {metrics.auprc:.4f}")
    print(f"  ROC-AUC: {metrics.roc_auc:.4f}")
    print(f"  Sensitivity @ 95% specificity: {metrics.sensitivity_at_95_specificity:.4f}")
    print(f"  False-alarm rate @ threshold: {metrics.false_alarm_rate:.4f}")
    print(f"  Alarms per 100 windows: {metrics.alarms_per_100_windows:.2f}")
    print(f"  Confusion matrix [ [TN,FP],[FN,TP] ]: {metrics.confusion_matrix}")


def train(
    n_patients: int = 500,
    arrest_fraction: float = 0.5,
    seed: int = 42,
    model_dir: str | Path = "models",
    n_estimators: int = 300,
    use_gpu: bool = False,
    tune: bool = False,
    n_trials: int = 30,
    compare: bool = False,
    models: Sequence[str] | None = None,
    target_sensitivity: float = 0.90,
) -> dict[str, Any]:
    """Run the full synthetic early-warning pipeline and save the model.

    Returns a results dict comparing XGBoost against the NEWS baseline. With
    ``tune=True`` an Optuna search (patient-grouped CV, AUPRC objective) picks the
    hyperparameters before the final fit; otherwise the fixed defaults are used.

    ``compare=True`` additionally trains the alternative learners in
    :data:`MODEL_NAMES` on the same split and appends a ranked ``comparison``
    table to the results. The saved artifact stays the XGBoost model either way —
    ``vitals_api`` and ``vitals_explain`` load it by that name, and a comparison
    run is meant to inform the model choice, not silently change it.
    """
    project_root = Path(__file__).resolve().parent.parent
    model_path = project_root / model_dir if not Path(model_dir).is_absolute() else Path(model_dir)

    cohort = generate_synthetic_cohort(n_patients=n_patients, arrest_fraction=arrest_fraction, seed=seed)
    windowed = build_windows(cohort)
    split = patient_level_split(windowed, test_size=0.2, seed=seed)

    overrides = (
        tune_xgboost(split, n_trials=n_trials, use_gpu=use_gpu, seed=seed)
        if tune
        else {"n_estimators": n_estimators}
    )
    started = time.perf_counter()
    model, xgb_metrics = train_xgboost(split, use_gpu=use_gpu, **overrides)
    xgb_fit_seconds = time.perf_counter() - started
    news_metrics = evaluate_news_baseline(split)

    output_file = model_path / "vitals_ews_model.pkl"
    save_artifact(model, split, xgb_metrics, output_file)

    results = {
        "xgboost": asdict(xgb_metrics),
        "news_baseline": asdict(news_metrics),
        "model_path": str(output_file),
        "tuned": tune,
        "best_params": overrides if tune else None,
        "note": "Synthetic development data. Real training runs in the 안심존 on KHTH data.",
    }

    comparison: list[ModelResult] = []
    if compare:
        requested = [name.lower() for name in (models if models is not None else MODEL_NAMES)]
        # Reuse the XGBoost fit we already have instead of paying for it twice —
        # but only when it was not tuned, since comparing a searched XGBoost
        # against default-parameter rivals would rank by search budget, not model.
        reuse_xgb = "xgboost" in requested and not tune
        comparison = compare_models(
            split,
            models=[name for name in requested if not (name == "xgboost" and reuse_xgb)],
            use_gpu=use_gpu,
            target_sensitivity=target_sensitivity,
        )
        if reuse_xgb:
            xgb_score = model.predict_proba(split.X_test)[:, 1]
            comparison.append(
                _score_result(
                    "xgboost",
                    "XGBoost",
                    split,
                    xgb_score,
                    target_sensitivity,
                    fit_seconds=xgb_fit_seconds,
                    estimator=model,
                )
            )
        comparison.sort(key=lambda result: result.metrics.auprc, reverse=True)
        results["comparison"] = [result.summary() for result in comparison]
        results["matched_sensitivity"] = target_sensitivity

    (model_path / "vitals_ews_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    _print_metrics(xgb_metrics)
    print()
    _print_metrics(news_metrics)
    if comparison:
        _print_comparison(comparison, target_sensitivity)
    print(f"\nModel saved to: {output_file}")
    return results


def main() -> None:
    """CLI entry point for the synthetic training + comparison run.

        python src/vitals_train.py                      # XGBoost vs NEWS
        python src/vitals_train.py --compare            # + LightGBM/CatBoost/RF/Logistic
        python src/vitals_train.py --compare --models logistic,random_forest
        python src/vitals_train.py --tune --trials 40 --gpu
    """
    import argparse

    parser = argparse.ArgumentParser(description="Cardiac-arrest early-warning training.")
    parser.add_argument("--gpu", action="store_true", help="train on CUDA where the model supports it")
    parser.add_argument("--tune", action="store_true", help="Optuna search before the final XGBoost fit")
    parser.add_argument("--trials", type=int, default=30, help="Optuna trials (default: 30)")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="also train the alternative learners and print the ranked table",
    )
    parser.add_argument(
        "--models",
        default=None,
        help=f"comma-separated subset to compare (default: all of {','.join(MODEL_NAMES)})",
    )
    parser.add_argument(
        "--sens",
        type=float,
        default=0.90,
        help="matched sensitivity for the alarm-burden column (default: 0.90)",
    )
    args = parser.parse_args()

    selected = [name.strip() for name in args.models.split(",") if name.strip()] if args.models else None
    train(
        use_gpu=args.gpu,
        tune=args.tune,
        n_trials=args.trials,
        compare=args.compare or selected is not None,
        models=selected,
        target_sensitivity=args.sens,
    )


if __name__ == "__main__":
    main()
