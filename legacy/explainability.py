"""SHAP-based model explainability for heart failure risk prediction."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from data_loader import HeartFailureDataLoader


def load_model_artifact(model_path: str | Path) -> dict[str, Any]:
    """Load trained model artifact from pickle file."""
    with Path(model_path).open("rb") as handle:
        return pickle.load(handle)


def global_top_features(
    model: Any,
    X: np.ndarray,
    feature_names: list[str],
    top_k: int = 5,
) -> list[dict[str, float]]:
    """Return global top-K feature importances using mean |SHAP| values."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    mean_abs = np.abs(shap_values).mean(axis=0)
    ranked_indices = np.argsort(mean_abs)[::-1][:top_k]

    return [
        {"feature": feature_names[i], "mean_abs_shap": float(mean_abs[i])}
        for i in ranked_indices
    ]


def patient_risk_factors(
    model: Any,
    patient_features: np.ndarray,
    feature_names: list[str],
    top_k: int = 3,
) -> list[dict[str, float]]:
    """Return patient-level top-K SHAP risk contributors."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(patient_features)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    patient_shap = shap_values[0]
    ranked_indices = np.argsort(np.abs(patient_shap))[::-1][:top_k]

    return [
        {
            "feature": feature_names[i],
            "shap_value": float(patient_shap[i]),
            "feature_value": float(patient_features[0, i]),
        }
        for i in ranked_indices
    ]


def save_shap_summary_plot(
    model: Any,
    X: np.ndarray,
    feature_names: list[str],
    output_path: str | Path,
    max_display: int = 10,
) -> Path:
    """Generate and save SHAP summary plot to disk."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    X_display = pd.DataFrame(X, columns=feature_names)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_display, max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()
    return output


def generate_explanations(
    model_path: str | Path | None = None,
    data_dir: str | Path = "data",
    output_dir: str | Path = "models",
) -> dict[str, Any]:
    """Generate global and summary SHAP explanations for the trained model."""
    project_root = Path(__file__).resolve().parent.parent
    model_file = Path(model_path) if model_path else project_root / "models" / "best_model.pkl"
    output_path = project_root / output_dir if not Path(output_dir).is_absolute() else Path(output_dir)

    artifact = load_model_artifact(model_file)
    model = artifact["model"]
    feature_names: list[str] = artifact["feature_names"]

    loader = HeartFailureDataLoader(data_dir=project_root / data_dir)
    loader.load_raw_data()
    processed = loader.preprocess()
    split = loader.get_train_test_split()

    top_features = global_top_features(model, split.X_test, feature_names, top_k=5)
    sample_patient = patient_risk_factors(model, split.X_test[:1], feature_names, top_k=3)
    plot_path = save_shap_summary_plot(
        model,
        split.X_test,
        feature_names,
        output_path / "shap_summary.png",
    )

    print("=== Global Top 5 Feature Importance (SHAP) ===")
    for rank, item in enumerate(top_features, start=1):
        print(f"  {rank}. {item['feature']}: {item['mean_abs_shap']:.4f}")

    print("\n=== Sample Patient Top 3 Risk Factors ===")
    for rank, item in enumerate(sample_patient, start=1):
        print(f"  {rank}. {item['feature']}: SHAP={item['shap_value']:.4f}, value={item['feature_value']:.4f}")

    print(f"\nSHAP summary plot saved to: {plot_path}")

    return {
        "global_top_features": top_features,
        "sample_patient_factors": sample_patient,
        "shap_summary_plot": str(plot_path),
    }


def main() -> None:
    """CLI entry point for SHAP explainability generation."""
    generate_explanations()


if __name__ == "__main__":
    main()
