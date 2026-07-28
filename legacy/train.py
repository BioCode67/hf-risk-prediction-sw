"""Model training with LightGBM, XGBoost, and Optuna hyperparameter optimization."""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import optuna
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from data_loader import HeartFailureDataLoader, SplitData

optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class TrainingMetrics:
    """Evaluation metrics for a trained classifier."""

    model_name: str
    auprc: float
    roc_auc: float
    f1: float
    confusion_matrix: list[list[int]]
    sensitivity_at_95_specificity: float
    scale_pos_weight: float


def _scale_pos_weight(y: np.ndarray) -> float:
    """Compute cost-sensitive positive class weight for imbalanced data."""
    negatives = int(np.sum(y == 0))
    positives = int(np.sum(y == 1))
    if positives == 0:
        return 1.0
    return negatives / positives


def sensitivity_at_specificity(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_specificity: float = 0.95,
) -> float:
    """Return sensitivity (recall) at the given specificity threshold."""
    thresholds = np.unique(np.sort(y_prob))
    best_sensitivity = 0.0

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if specificity >= target_specificity:
            best_sensitivity = max(best_sensitivity, sensitivity)

    return float(best_sensitivity)


def evaluate_model(
    model_name: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    scale_pos_weight: float,
) -> TrainingMetrics:
    """Compute classification metrics for model predictions."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return TrainingMetrics(
        model_name=model_name,
        auprc=float(average_precision_score(y_true, y_prob)),
        roc_auc=float(roc_auc_score(y_true, y_prob)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        confusion_matrix=cm.tolist(),
        sensitivity_at_95_specificity=sensitivity_at_specificity(y_true, y_prob, 0.95),
        scale_pos_weight=scale_pos_weight,
    )


def _optuna_objective(
    trial: optuna.Trial,
    X_train: np.ndarray,
    y_train: np.ndarray,
    scale_pos_weight: float,
) -> float:
    """Optuna objective maximizing cross-validated AUPRC for LightGBM."""
    params: dict[str, Any] = {
        "objective": "binary",
        "metric": "average_precision",
        "boosting_type": "gbdt",
        "verbosity": -1,
        "random_state": 42,
        "scale_pos_weight": scale_pos_weight,
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 8, 64),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores: list[float] = []

    for train_idx, val_idx in cv.split(X_train, y_train):
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train[train_idx],
            y_train[train_idx],
            eval_set=[(X_train[val_idx], y_train[val_idx])],
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
        )
        prob = model.predict_proba(X_train[val_idx])[:, 1]
        scores.append(average_precision_score(y_train[val_idx], prob))

    return float(np.mean(scores))


def train_lightgbm(
    split: SplitData,
    n_trials: int = 20,
) -> tuple[lgb.LGBMClassifier, TrainingMetrics, dict[str, Any]]:
    """Train LightGBM with Optuna tuning and cost-sensitive weighting."""
    scale_weight = _scale_pos_weight(split.y_train)

    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: _optuna_objective(trial, split.X_train, split.y_train, scale_weight),
        n_trials=n_trials,
        show_progress_bar=False,
    )

    best_params: dict[str, Any] = {
        "objective": "binary",
        "metric": "average_precision",
        "boosting_type": "gbdt",
        "verbosity": -1,
        "random_state": 42,
        "scale_pos_weight": scale_weight,
        **study.best_params,
    }

    model = lgb.LGBMClassifier(**best_params)
    model.fit(split.X_train, split.y_train)

    y_prob = model.predict_proba(split.X_test)[:, 1]
    y_pred = model.predict(split.X_test)
    metrics = evaluate_model("LightGBM", split.y_test, y_prob, y_pred, scale_weight)
    return model, metrics, best_params


def train_xgboost(split: SplitData) -> tuple[xgb.XGBClassifier, TrainingMetrics]:
    """Train XGBoost baseline with scale_pos_weight for imbalanced classification."""
    scale_weight = _scale_pos_weight(split.y_train)

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        random_state=42,
        scale_pos_weight=scale_weight,
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        verbosity=0,
    )
    model.fit(split.X_train, split.y_train)

    y_prob = model.predict_proba(split.X_test)[:, 1]
    y_pred = model.predict(split.X_test)
    metrics = evaluate_model("XGBoost", split.y_test, y_prob, y_pred, scale_weight)
    return model, metrics


def save_model(
    model: lgb.LGBMClassifier,
    feature_names: list[str],
    scaler: Any,
    output_path: Path,
    metrics: TrainingMetrics,
    best_params: dict[str, Any],
) -> None:
    """Persist trained model and metadata to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Store metrics as a plain dict so the artifact unpickles without importing
    # the training module (avoids __main__/TrainingMetrics pickle coupling).
    artifact = {
        "model": model,
        "feature_names": feature_names,
        "scaler": scaler,
        "metrics": asdict(metrics),
        "best_params": best_params,
    }
    with output_path.open("wb") as handle:
        pickle.dump(artifact, handle)


def train(
    data_dir: str | Path = "data",
    model_dir: str | Path = "models",
    n_trials: int = 20,
) -> dict[str, Any]:
    """Run full training pipeline and save the best LightGBM model.

    Returns:
        Dictionary with metrics for both models and saved artifact path.
    """
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / data_dir if not Path(data_dir).is_absolute() else Path(data_dir)
    model_path = project_root / model_dir if not Path(model_dir).is_absolute() else Path(model_dir)

    loader = HeartFailureDataLoader(data_dir=data_path)
    loader.load_raw_data()
    loader.preprocess()
    split = loader.get_train_test_split()

    lgb_model, lgb_metrics, best_params = train_lightgbm(split, n_trials=n_trials)
    xgb_model, xgb_metrics = train_xgboost(split)

    assert loader._processed is not None
    output_file = model_path / "best_model.pkl"
    save_model(
        lgb_model,
        split.feature_names,
        loader._processed.scaler,
        output_file,
        lgb_metrics,
        best_params,
    )

    metrics_path = model_path / "training_metrics.json"
    results = {
        "lightgbm": {
            "auprc": lgb_metrics.auprc,
            "roc_auc": lgb_metrics.roc_auc,
            "f1": lgb_metrics.f1,
            "confusion_matrix": lgb_metrics.confusion_matrix,
            "sensitivity_at_95_specificity": lgb_metrics.sensitivity_at_95_specificity,
            "scale_pos_weight": lgb_metrics.scale_pos_weight,
        },
        "xgboost": {
            "auprc": xgb_metrics.auprc,
            "roc_auc": xgb_metrics.roc_auc,
            "f1": xgb_metrics.f1,
            "confusion_matrix": xgb_metrics.confusion_matrix,
            "sensitivity_at_95_specificity": xgb_metrics.sensitivity_at_95_specificity,
            "scale_pos_weight": xgb_metrics.scale_pos_weight,
        },
        "best_params": best_params,
        "model_path": str(output_file),
    }
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("=== LightGBM (Optuna-tuned) ===")
    print(f"  AUPRC: {lgb_metrics.auprc:.4f}")
    print(f"  ROC-AUC: {lgb_metrics.roc_auc:.4f}")
    print(f"  F1: {lgb_metrics.f1:.4f}")
    print(f"  Confusion Matrix: {lgb_metrics.confusion_matrix}")
    print(f"  Sensitivity@Specificity 95%: {lgb_metrics.sensitivity_at_95_specificity:.4f}")
    print(f"  scale_pos_weight: {lgb_metrics.scale_pos_weight:.4f}")

    print("\n=== XGBoost (baseline) ===")
    print(f"  AUPRC: {xgb_metrics.auprc:.4f}")
    print(f"  ROC-AUC: {xgb_metrics.roc_auc:.4f}")
    print(f"  F1: {xgb_metrics.f1:.4f}")
    print(f"  Confusion Matrix: {xgb_metrics.confusion_matrix}")
    print(f"  Sensitivity@Specificity 95%: {xgb_metrics.sensitivity_at_95_specificity:.4f}")

    print(f"\nBest model saved to: {output_file}")
    return results


def main() -> None:
    """CLI entry point for model training."""
    train(n_trials=20)


if __name__ == "__main__":
    main()
