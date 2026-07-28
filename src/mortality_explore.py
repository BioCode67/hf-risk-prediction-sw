"""Run the early-warning pipeline on the open PhysioNet Challenge 2012 data.

Challenge 2012 ("Predicting Mortality of ICU Patients") is openly downloadable
with no credentialing: 4,000 ICU stays per set, the first 48 hours of each, with
36 irregularly sampled vitals and labs. It is roughly four times the patient
count of the Challenge-2019 subset we were using, which matters because the
model-vs-NEWS gap there sat inside the cross-validation noise.

The event is in-hospital death rather than cardiac arrest, and ``Survival`` is
recorded in whole days, so event times are coarse. Treat the numbers as evidence
that the pipeline works on real data, not as cardiac-arrest performance.

Download (no login):
    wget -r -N -c -np -nH --cut-dirs=4 \\
      https://physionet.org/files/challenge-2012/1.0.0/set-a/
    wget https://physionet.org/files/challenge-2012/1.0.0/Outcomes-a.txt

Run:
    python src/mortality_explore.py /path/to/set-a [--outcomes=PATH] [--horizon=H]
        [--window=W] [--gap=G] [--gpu] [--tune] [--trials=N] [--max-files=N]
"""

from __future__ import annotations

import sys
from pathlib import Path

from vitals_data import (
    GAP_HOURS,
    OBSERVATION_WINDOW_HOURS,
    PREDICTION_HORIZON_HOURS,
    add_personalized_features,
    build_windows,
    cohort_from_challenge2012,
    patient_level_split,
)


def run(
    data_dir: str,
    outcomes_path: str | None = None,
    use_gpu: bool = False,
    tune: bool = False,
    n_trials: int = 30,
    max_files: int | None = None,
    horizon_hours: int = 6,
    window_hours: int = OBSERVATION_WINDOW_HOURS,
    gap_hours: int = GAP_HOURS,
) -> None:
    """Load Challenge-2012 records and run XGBoost vs NEWS + the full report."""
    from vitals_phenotype import discover_phenotypes
    from vitals_report import render_report
    from vitals_train import (
        evaluate_news_baseline,
        lead_time_summary,
        threshold_at_specificity,
        train_xgboost,
        tune_xgboost,
    )

    print(
        f"Loading Challenge-2012 records from {data_dir} (max_files={max_files}) | "
        f"window={window_hours}h horizon={horizon_hours}h gap={gap_hours}h ..."
    )
    cohort = cohort_from_challenge2012(data_dir, outcomes_path=outcomes_path, max_files=max_files)
    windowed = add_personalized_features(
        build_windows(
            cohort,
            observation_window_hours=window_hours,
            prediction_horizon_hours=horizon_hours,
            gap_hours=gap_hours,
        ),
        cohort,
    )

    positives = int(windowed.labels.sum())
    n_event = int(cohort.events["arrest_hour"].notna().sum())
    n_patients = cohort.vitals["patient_id"].nunique()
    print(
        f"Patients: {n_patients} ({n_event} with a dated in-hospital death) | "
        f"windows: {len(windowed.labels)} | positive: {positives} "
        f"({windowed.labels.mean():.2%})"
    )
    if positives < 10:
        print(
            "Too few positive windows to learn from. Most deaths fall after the "
            "48-hour record, so widen --horizon or read more files."
        )
        return

    split = patient_level_split(windowed)
    best = tune_xgboost(split, n_trials=n_trials, use_gpu=use_gpu) if tune else {}
    model, xgb = train_xgboost(split, use_gpu=use_gpu, **best)
    news = evaluate_news_baseline(split)

    base_rate = float(split.y_test.mean())
    print(f"\nAUPRC baseline (positive rate) = {base_rate:.4f} — read AUPRC against this.")
    for m in (xgb, news):
        print(
            f"{m.model_name:8s} AUPRC={m.auprc:.3f} ({m.auprc / base_rate:.1f}x baseline) "
            f"ROC={m.roc_auc:.3f} sens@95spec={m.sensitivity_at_95_specificity:.3f} "
            f"falseAlarm={m.false_alarm_rate:.3f}"
        )

    score = model.predict_proba(split.X_test)[:, 1]
    lead = lead_time_summary(split, score, threshold_at_specificity(split.y_test, score))
    if lead:
        print(
            f"Lead-time: detected {int(lead['detected'])}/{int(lead['arrest_patients'])} events, "
            f"median {lead['median_lead_time_h']:.1f}h before death (@95% specificity)"
        )

    models_dir = Path(__file__).resolve().parent.parent / "models"
    print()
    render_report(split, model, cohort, models_dir)
    try:
        discover_phenotypes(cohort, output_dir=models_dir)
    except Exception as exc:
        print(f"(phenotype step skipped: {exc})")


def main() -> None:
    """CLI entry point."""
    positional = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not positional:
        print(__doc__)
        return

    def _int(name: str, default: int | None) -> int | None:
        return next((int(a.split("=", 1)[1]) for a in sys.argv if a.startswith(f"--{name}=")), default)

    outcomes = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--outcomes=")), None)

    run(
        positional[0],
        outcomes_path=outcomes,
        use_gpu="--gpu" in sys.argv,
        tune="--tune" in sys.argv,
        n_trials=_int("trials", 30),
        max_files=_int("max-files", None),
        horizon_hours=_int("horizon", 6),
        window_hours=_int("window", OBSERVATION_WINDOW_HOURS),
        gap_hours=_int("gap", GAP_HOURS),
    )


if __name__ == "__main__":
    main()
