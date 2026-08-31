"""Visual report for the cardiac-arrest early-warning system.

The 본선 deliverable is a *data-visualization* report and a presentation (70% of
the final score), so the figures matter as much as the model. This module renders
the story the proposal promises, from any cohort:

1. Precision-recall curves (XGBoost vs NEWS) — the alarm-quality gap ROC-AUC hides.
2. A patient's deterioration trajectory with the arrest marked — the clinical "why".
3. The lead-time distribution — how early alarms fire before arrest.

Matplotlib only (headless Agg); imported lazily so the rest of the pipeline does
not depend on it. Figures are written under ``models/`` (git-ignored).

Labels are Korean because these exact figures go into the 본선 결과보고서; the
안심존 has no Korean system fonts, so ``plot_fonts`` registers the bundled
NanumGothic before anything is drawn.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

from vitals_data import VITALS
from vitals_narrate import VITAL_KO


def _new_axes(figsize: tuple[float, float]):
    """Create a headless matplotlib figure/axes with a Korean font (lazy import)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from plot_fonts import ensure_korean_font

    plt.rcParams["font.family"] = ensure_korean_font()
    plt.rcParams["axes.unicode_minus"] = False
    return plt, *plt.subplots(figsize=figsize)


def plot_pr_curves(y_true: np.ndarray, scores: dict[str, np.ndarray], output_path: str | Path) -> Path:
    """Precision-recall curves for each named score, with AUPRC in the legend."""
    plt, fig, ax = _new_axes((7, 6))
    prevalence = float(np.mean(y_true))
    for name, score in scores.items():
        precision, recall, _ = precision_recall_curve(y_true, score)
        auprc = average_precision_score(y_true, score)
        ax.plot(recall, precision, label=f"{name} — AUPRC {auprc:.3f}", linewidth=2)
    ax.axhline(prevalence, ls="--", c="grey", lw=1, label=f"우연 수준 (양성률={prevalence:.3f})")
    ax.set_xlabel("재현율 (민감도)")
    ax.set_ylabel("정밀도")
    ax.set_title("정밀도–재현율 곡선: 조기경보 알람 품질")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_deterioration_trajectory(
    patient_vitals: pd.DataFrame,
    arrest_hour: float | None,
    output_path: str | Path,
) -> Path:
    """Plot one patient's six vitals over time, marking the arrest hour."""
    plt, fig, axes = _new_axes((11, 6))
    fig.clf()
    axes = fig.subplots(2, 3)
    hours = patient_vitals["hour"].to_numpy()
    has_arrest = arrest_hour is not None and np.isfinite(arrest_hour)
    for ax, vital in zip(axes.ravel(), VITALS):
        ax.plot(hours, patient_vitals[vital].to_numpy(), marker=".", lw=1.2)
        if has_arrest:
            # axvline is excluded from autoscale, so a panel whose data ends
            # before the arrest hour would silently clip the marker out.
            ax.axvline(arrest_hour, c="red", ls="--", lw=1.2)
            ax.set_xlim(right=max(float(hours.max()), float(arrest_hour)) + 0.8)
        label, unit = VITAL_KO[vital]
        ax.set_title(f"{label} ({unit})")
        ax.set_xlabel("경과 시간 (h)")
    fig.suptitle("심정지 전 활력징후 궤적 (빨간 선 = 심정지 시점)")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_alarm_burden_curve(
    y_true: np.ndarray,
    scores: dict[str, np.ndarray],
    output_path: str | Path,
    sensitivities: np.ndarray | None = None,
) -> Path:
    """Alarms per 100 windows vs matched sensitivity — the false-alarm trade-off.

    Holding detection (sensitivity) fixed, a lower curve means fewer alarms for the
    same catch rate. This is the clearest picture of the false-alarm-reduction thesis.
    """
    from vitals_train import threshold_at_sensitivity

    grid = np.linspace(0.70, 0.99, 12) if sensitivities is None else sensitivities
    plt, fig, ax = _new_axes((7, 5))
    for name, score in scores.items():
        alarms = []
        for target in grid:
            threshold = threshold_at_sensitivity(y_true, score, float(target))
            alarms.append(100.0 * float((score >= threshold).mean()))
        ax.plot(grid, alarms, marker="o", lw=2, label=name)
    ax.set_xlabel("동일 민감도 (검출률)")
    ax.set_ylabel("100 윈도우당 알람 수")
    ax.set_title("동일 검출률에서의 알람 부담 (낮을수록 좋음)")
    ax.legend()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_lead_time_distribution(lead_times: list[float], output_path: str | Path) -> Path:
    """Histogram of per-patient lead time (hours before arrest of first alarm)."""
    plt, fig, ax = _new_axes((7, 5))
    values = np.asarray(lead_times, dtype=float)
    if values.size:
        ax.hist(values, bins=min(20, max(5, values.size)), color="#3b7dd8", edgecolor="white")
        ax.axvline(float(np.median(values)), c="red", ls="--", lw=1.5, label=f"중앙값 {np.median(values):.1f}시간")
        ax.legend()
    ax.set_xlabel("심정지까지 남은 시간 — 첫 알람 시점 (시간)")
    ax.set_ylabel("환자 수")
    ax.set_title("조기경보 리드타임 분포")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output


def render_report(split: Any, model: Any, cohort: Any, out_dir: str | Path) -> dict[str, Any]:
    """Render the full figure set + metrics from a trained split/model/cohort.

    Shared by the synthetic demo and the real MIMIC-IV run so both produce an
    identical report from any data source.
    """
    from vitals_train import alarm_burden, compute_news_scores, lead_time_summary, threshold_at_specificity

    out = Path(out_dir)
    xgb_score = model.predict_proba(split.X_test)[:, 1]
    news_score = compute_news_scores(split.X_test)
    threshold = threshold_at_specificity(split.y_test, xgb_score)
    scores = {"본 모델 (XGBoost)": xgb_score, "NEWS (임상 표준)": news_score}

    pr_path = plot_pr_curves(split.y_test, scores, out / "vitals_pr_curve.png")
    burden_path = plot_alarm_burden_curve(split.y_test, scores, out / "vitals_alarm_burden.png")

    # Lead-time distribution from per-patient earliest alarms.
    frame = pd.DataFrame({"patient": split.groups_test, "tta": split.time_to_arrest_test, "score": xgb_score})
    arrest = frame[frame["tta"].notna() & (frame["tta"] >= 0)]
    lead_times = [
        float(g[(g["score"] >= threshold) & (g["tta"] <= 48)]["tta"].max())
        for _, g in arrest.groupby("patient")
        if len(g[(g["score"] >= threshold) & (g["tta"] <= 48)])
    ]
    lead_path = plot_lead_time_distribution(lead_times, out / "vitals_lead_time.png")

    # Trajectory for one arrest patient.
    arrest_ids = cohort.events[cohort.events["arrest_hour"].notna()]
    traj_path = None
    if len(arrest_ids):
        pid = arrest_ids.iloc[0]["patient_id"]
        arrest_hour = float(arrest_ids.iloc[0]["arrest_hour"])
        pv = cohort.vitals[cohort.vitals["patient_id"] == pid].sort_values("hour")
        traj_path = plot_deterioration_trajectory(pv, arrest_hour, out / "vitals_trajectory.png")

    print(f"PR curve:      {pr_path}")
    print(f"Alarm burden:  {burden_path}")
    print(f"Lead-time:     {lead_path}")
    print(f"Trajectory:    {traj_path}")
    print("\n=== Alarm burden at matched 90% sensitivity ===")
    for name, score in scores.items():
        b = alarm_burden(split.y_test, score, target_sensitivity=0.90)
        print(
            f"  {name:8s} specificity={b['specificity']:.3f} "
            f"false-alarm={b['false_alarm_rate']:.3f} alarms/100={b['alarms_per_100_windows']:.1f}"
        )
    summary = lead_time_summary(split, xgb_score, threshold)
    if summary:
        print(
            f"\nDetected {int(summary['detected'])}/{int(summary['arrest_patients'])} arrests, "
            f"median lead-time {summary['median_lead_time_h']:.1f}h"
        )
    return {
        "pr_curve": str(pr_path),
        "alarm_burden": str(burden_path),
        "lead_time": str(lead_path),
        "trajectory": str(traj_path),
    }


def generate_report(output_dir: str | Path = "models", n_patients: int = 500, seed: int = 42) -> dict[str, Any]:
    """Train on a synthetic cohort and render the full figure set (no external data)."""
    from vitals_data import add_personalized_features, build_windows, generate_synthetic_cohort, patient_level_split
    from vitals_train import train_xgboost

    project_root = Path(__file__).resolve().parent.parent
    out = project_root / output_dir if not Path(output_dir).is_absolute() else Path(output_dir)

    cohort = generate_synthetic_cohort(n_patients=n_patients, seed=seed)
    split = patient_level_split(add_personalized_features(build_windows(cohort), cohort), seed=seed)
    model, _metrics = train_xgboost(split)
    return render_report(split, model, cohort, out)


def main() -> None:
    """CLI entry point: render the synthetic-data report."""
    generate_report()


if __name__ == "__main__":
    main()
