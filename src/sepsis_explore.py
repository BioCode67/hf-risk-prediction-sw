"""Run the early-warning pipeline on the open PhysioNet Challenge 2019 (sepsis) data.

The best *openly downloadable, no-credentialing* dataset with real positives AND
controls for "vitals time series → imminent deterioration event". Not cardiac
arrest (it is sepsis), but the same problem shape, so it validates the whole
method end-to-end on real data while credentialed cardiac-arrest data is pending.

Download (no login):
    https://physionet.org/content/challenge-2019/1.0.0/
    # training_setA/ and training_setB/ contain p*.psv (one patient per file)

Run:
    python src/sepsis_explore.py /path/to/training_setA [--gpu] [--tune] [--trials=N] [--max-files=N]
"""

from __future__ import annotations

import sys
from pathlib import Path

from vitals_data import add_personalized_features, build_windows, cohort_from_challenge2019, patient_level_split


def run(psv_dir: str, use_gpu: bool = False, tune: bool = False, n_trials: int = 30, max_files: int | None = None) -> None:
    """Load Challenge-2019 vitals and run XGBoost vs NEWS + the full report."""
    from vitals_phenotype import discover_phenotypes
    from vitals_report import render_report
    from vitals_train import (
        evaluate_news_baseline,
        lead_time_summary,
        threshold_at_specificity,
        train_xgboost,
        tune_xgboost,
    )

    print(f"Loading Challenge-2019 PSVs from {psv_dir} (max_files={max_files}) ...")
    cohort = cohort_from_challenge2019(psv_dir, max_files=max_files)
    windowed = add_personalized_features(build_windows(cohort), cohort)
    positives = int(windowed.labels.sum())
    n_event = int(cohort.events["arrest_hour"].notna().sum())
    print(f"Patients: {cohort.vitals['patient_id'].nunique()} ({n_event} with event) | "
          f"windows: {len(windowed.labels)} | positive: {positives}")
    if positives < 10:
        print("Too few positive windows — increase --max-files.")
        return

    split = patient_level_split(windowed)
    best = tune_xgboost(split, n_trials=n_trials, use_gpu=use_gpu) if tune else {}
    model, xgb = train_xgboost(split, use_gpu=use_gpu, **best)
    news = evaluate_news_baseline(split)
    for m in (xgb, news):
        print(
            f"{m.model_name:8s} AUPRC={m.auprc:.3f} ROC={m.roc_auc:.3f} "
            f"sens@95spec={m.sensitivity_at_95_specificity:.3f} falseAlarm={m.false_alarm_rate:.3f}"
        )

    score = model.predict_proba(split.X_test)[:, 1]
    lead = lead_time_summary(split, score, threshold_at_specificity(split.y_test, score))
    if lead:
        print(f"Lead-time: detected {int(lead['detected'])}/{int(lead['arrest_patients'])} events, "
              f"median {lead['median_lead_time_h']:.1f}h before onset (@95% specificity)")

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
    trials = next((int(a.split("=", 1)[1]) for a in sys.argv if a.startswith("--trials=")), 30)
    max_files = next((int(a.split("=", 1)[1]) for a in sys.argv if a.startswith("--max-files=")), None)
    run(positional[0], use_gpu="--gpu" in sys.argv, tune="--tune" in sys.argv, n_trials=trials, max_files=max_files)


if __name__ == "__main__":
    main()
