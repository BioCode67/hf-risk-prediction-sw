"""Cardiac-arrest phenotype discovery from pre-arrest vital-sign trajectories.

A novelty angle that turns the case-only cohort into an asset: instead of only
*predicting* arrest, we *discover* how patients deteriorate. Each arrest patient
is summarized by how their vitals changed (vs their own baseline) in the hours
before arrest, then clustered into a few **phenotypes** — e.g. hypoxic-respiratory
vs hypotensive-circulatory vs tachycardic. This supports phenotype-tailored alarm
logic and gives a strong, interpretable figure for the 본선 report.

Lightweight (scikit-learn KMeans); matplotlib imported lazily.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vitals_data import VITALS, VitalSignsCohort

CHANGE_COLUMNS: list[str] = [f"{vital}_change" for vital in VITALS]


def patient_prearrest_signatures(
    cohort: VitalSignsCohort,
    hours_before: int = 6,
    baseline_hours: int = 6,
) -> pd.DataFrame:
    """Per arrest patient, the change of each vital (pre-arrest mean − baseline mean).

    Baseline = mean over the first ``baseline_hours``; pre-arrest = mean over the
    ``hours_before`` hours preceding the arrest. Controls and patients lacking
    either window are dropped.
    """
    events = dict(zip(cohort.events["patient_id"], cohort.events["arrest_hour"]))
    rows: list[dict[str, Any]] = []
    for patient_id, group in cohort.vitals.groupby("patient_id"):
        arrest_hour = events.get(patient_id, float("nan"))
        if arrest_hour is None or not np.isfinite(arrest_hour):
            continue
        group = group.sort_values("hour")
        baseline = group[group["hour"] < baseline_hours][list(VITALS)].mean()
        pre = group[(group["hour"] >= arrest_hour - hours_before) & (group["hour"] < arrest_hour)][list(VITALS)].mean()
        change = pre - baseline
        row: dict[str, Any] = {"patient_id": patient_id}
        for vital in VITALS:
            row[f"{vital}_change"] = float(change[vital])
        rows.append(row)
    signatures = pd.DataFrame(rows)
    return signatures.dropna(subset=CHANGE_COLUMNS).reset_index(drop=True)


def cluster_phenotypes(signatures: pd.DataFrame, k: int = 3, seed: int = 42) -> tuple[np.ndarray, pd.DataFrame]:
    """KMeans over standardized vital-change signatures; return labels + centroids.

    Centroids are reported in original units (mean change per vital per cluster),
    which is what makes each phenotype clinically readable.
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    features = signatures[CHANGE_COLUMNS].to_numpy()
    scaled = StandardScaler().fit_transform(features)
    labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(scaled)
    centroids = signatures.assign(cluster=labels).groupby("cluster")[CHANGE_COLUMNS].mean()
    return labels, centroids


def plot_phenotypes(centroids: pd.DataFrame, sizes: pd.Series, output_path: str | Path) -> Path:
    """Heatmap of each phenotype's mean vital change (rows = clusters)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = centroids.to_numpy()
    limit = float(np.max(np.abs(data))) or 1.0
    fig, ax = plt.subplots(figsize=(8, 1.4 + 0.7 * len(centroids)))
    im = ax.imshow(data, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(len(VITALS)))
    ax.set_xticklabels(list(VITALS), rotation=30, ha="right")
    ax.set_yticks(range(len(centroids)))
    ax.set_yticklabels([f"phenotype {c} (n={int(sizes.get(c, 0))})" for c in centroids.index])
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:+.1f}", ha="center", va="center", fontsize=8)
    ax.set_title("Cardiac-arrest phenotypes: mean vital change before arrest")
    fig.colorbar(im, ax=ax, label="change vs personal baseline")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output


def discover_phenotypes(
    cohort: VitalSignsCohort | None = None,
    k: int = 3,
    output_dir: str | Path = "models",
) -> dict[str, Any]:
    """Discover and render arrest phenotypes (synthetic cohort when none given)."""
    from vitals_data import generate_synthetic_cohort

    if cohort is None:
        cohort = generate_synthetic_cohort(n_patients=500, arrest_fraction=0.6)
    project_root = Path(__file__).resolve().parent.parent
    out = project_root / output_dir if not Path(output_dir).is_absolute() else Path(output_dir)

    signatures = patient_prearrest_signatures(cohort)
    labels, centroids = cluster_phenotypes(signatures, k=k)
    sizes = pd.Series(labels).value_counts().sort_index()
    plot_path = plot_phenotypes(centroids, sizes, out / "vitals_phenotypes.png")

    print(f"Arrest patients clustered: {len(signatures)} into {k} phenotypes")
    print("\n=== Phenotype centroids (mean change vs baseline) ===")
    print(centroids.round(1).to_string())
    print(f"\nSizes: {dict(sizes)}")
    print(f"Figure: {plot_path}")
    return {"n_patients": len(signatures), "centroids": centroids, "sizes": dict(sizes), "figure": str(plot_path)}


def main() -> None:
    """CLI entry point: discover phenotypes on a synthetic cohort."""
    discover_phenotypes()


if __name__ == "__main__":
    main()
