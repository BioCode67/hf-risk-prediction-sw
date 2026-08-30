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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

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


def save_artifact(
    model: XGBClassifier,
    split: PatientSplit,
    metrics: EarlyWarningMetrics,
    output_path: Path,
) -> None:
    """Persist the trained model plus everything needed to score new windows.

    Two formats are written. The pickle is what the rest of this repo loads.
    The JSON pair exists for the 안심존: the carry-in declaration promises a
    reviewable text-format model (no pickle deserialization), so the booster
    goes out via XGBoost's own ``save_model`` and the preprocessing state as
    plain JSON alongside it.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": model,
        "feature_names": split.feature_names,
        "imputation_medians": split.imputation_medians,
        "metrics": asdict(metrics),
    }
    with output_path.open("wb") as handle:
        pickle.dump(artifact, handle)

    model.get_booster().save_model(str(output_path.with_suffix(".json")))
    config = {
        "feature_names": split.feature_names,
        "imputation_medians": {k: float(v) for k, v in split.imputation_medians.items()},
        "metrics": asdict(metrics),
    }
    config_path = output_path.with_name(output_path.stem + "_config.json")
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")


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
) -> dict[str, Any]:
    """Run the full synthetic early-warning pipeline and save the model.

    Returns a results dict comparing XGBoost against the NEWS baseline. With
    ``tune=True`` an Optuna search (patient-grouped CV, AUPRC objective) picks the
    hyperparameters before the final fit; otherwise the fixed defaults are used.
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
    model, xgb_metrics = train_xgboost(split, use_gpu=use_gpu, **overrides)
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
    (model_path / "vitals_ews_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    _print_metrics(xgb_metrics)
    print()
    _print_metrics(news_metrics)
    print(f"\nModel saved to: {output_file}")
    return results


def main() -> None:
    """CLI entry point. Flags: ``--gpu`` (CUDA), ``--tune`` (Optuna), ``--trials N``."""
    import sys

    argv = sys.argv
    n_trials = 30
    if "--trials" in argv and argv.index("--trials") + 1 < len(argv):
        n_trials = int(argv[argv.index("--trials") + 1])
    train(use_gpu="--gpu" in argv, tune="--tune" in argv, n_trials=n_trials)


if __name__ == "__main__":
    main()
