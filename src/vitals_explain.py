"""SHAP explainability for the cardiac-arrest early-warning model.

Explainability is a core deliverable of the project ("왜 위험한지" 근거 제시):
which vital-sign trends drive a window's risk, both globally and for a single
alarm. Uses SHAP's TreeExplainer, which is exact and fast for gradient-boosted
trees and needs no internet — suitable for the offline 안심존.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap


def _shap_values(model: Any, X: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Return a 2-D SHAP value matrix for the positive class."""
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X)
    if isinstance(values, list):  # some versions return [neg, pos]
        values = values[1]
    return np.asarray(values)


def global_top_features(
    model: Any,
    X: pd.DataFrame,
    feature_names: list[str],
    top_k: int = 5,
) -> list[dict[str, float]]:
    """Global feature importance via mean |SHAP| across windows."""
    values = _shap_values(model, X)
    mean_abs = np.abs(values).mean(axis=0)
    ranked = np.argsort(mean_abs)[::-1][:top_k]
    return [{"feature": feature_names[i], "mean_abs_shap": float(mean_abs[i])} for i in ranked]


def window_risk_factors(
    model: Any,
    window: pd.DataFrame | np.ndarray,
    feature_names: list[str],
    top_k: int = 3,
) -> list[dict[str, float]]:
    """Top-K SHAP risk drivers for a single window (first row of ``window``)."""
    values = _shap_values(model, window)
    contribution = values[0]
    feature_values = np.asarray(window)[0]
    ranked = np.argsort(np.abs(contribution))[::-1][:top_k]
    return [
        {
            "feature": feature_names[i],
            "shap_value": float(contribution[i]),
            "feature_value": float(feature_values[i]),
        }
        for i in ranked
    ]


def save_summary_plot(
    model: Any,
    X: pd.DataFrame,
    feature_names: list[str],
    output_path: str | Path,
    max_display: int = 12,
) -> Path:
    """Render and save a SHAP summary (beeswarm) plot."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    values = _shap_values(model, X)
    display_frame = pd.DataFrame(np.asarray(X), columns=feature_names)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(values, display_frame, max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()
    return output


def load_artifact(model_path: str | Path) -> dict[str, Any]:
    """Load a saved early-warning model artifact."""
    with Path(model_path).open("rb") as handle:
        return pickle.load(handle)


def generate_explanations(
    model_path: str | Path | None = None,
    output_dir: str | Path = "models",
) -> dict[str, Any]:
    """Produce global + single-window SHAP explanations for the saved model.

    Rebuilds a synthetic test split to explain against, so it runs end-to-end
    without external data.
    """
    from vitals_data import build_windows, generate_synthetic_cohort, patient_level_split

    project_root = Path(__file__).resolve().parent.parent
    model_file = Path(model_path) if model_path else project_root / "models" / "vitals_ews_model.pkl"
    output_path = project_root / output_dir if not Path(output_dir).is_absolute() else Path(output_dir)

    artifact = load_artifact(model_file)
    model = artifact["model"]
    feature_names: list[str] = artifact["feature_names"]

    split = patient_level_split(build_windows(generate_synthetic_cohort()))
    top_features = global_top_features(model, split.X_test, feature_names, top_k=5)

    # Explain the highest-risk window (most useful for a clinician).
    scores = model.predict_proba(split.X_test)[:, 1]
    riskiest = int(np.argmax(scores))
    factors = window_risk_factors(model, split.X_test.iloc[[riskiest]], feature_names, top_k=3)
    plot_path = save_summary_plot(model, split.X_test, feature_names, output_path / "vitals_shap_summary.png")

    print("=== Global top-5 drivers (mean |SHAP|) ===")
    for rank, item in enumerate(top_features, start=1):
        print(f"  {rank}. {item['feature']}: {item['mean_abs_shap']:.4f}")

    print(f"\n=== Riskiest window (p={scores[riskiest]:.3f}) top-3 factors ===")
    for rank, item in enumerate(factors, start=1):
        print(f"  {rank}. {item['feature']}: SHAP={item['shap_value']:.4f}, value={item['feature_value']:.4f}")

    print(f"\nSHAP summary plot saved to: {plot_path}")
    return {
        "global_top_features": top_features,
        "riskiest_window_factors": factors,
        "shap_summary_plot": str(plot_path),
    }


def main() -> None:
    """CLI entry point for SHAP explanation generation."""
    generate_explanations()


if __name__ == "__main__":
    main()
