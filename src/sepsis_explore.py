"""Run the early-warning pipeline on the open PhysioNet Challenge 2019 (sepsis) data.

The best *openly downloadable, no-credentialing* dataset with real positives AND
controls for "vitals time series → imminent deterioration event". Not cardiac
arrest (it is sepsis), but the same problem shape, so it validates the whole
method end-to-end on real data while credentialed cardiac-arrest data is pending.

Download (no login):
    https://physionet.org/content/challenge-2019/1.0.0/
    # training_setA/ and training_setB/ contain p*.psv (one patient per file)

Run:
    python src/sepsis_explore.py /path/to/training_setA [--gpu] [--tune] [--trials=N]
        [--max-files=N] [--horizon=H] [--window=W] [--gap=G] [--seed=S]

``--seed`` drives the patient-level split, the Optuna sampler and XGBoost alike, so
re-running with a different seed re-draws the test patients. A margin over NEWS that
moves when the seed does is split noise, not a result — always check more than one.

The prediction horizon defaults to 1 hour, which is deliberately strict: with one
event per patient and a 1-hour horizon only ~one window per patient is positive, so
the positive rate is ~0.2% and precision-based metrics (AUPRC) collapse to the base
rate even though ROC shows weak signal. A clinically standard early-warning horizon
(``--horizon=6``) yields several positive windows per patient and a learnable task.
"""

from __future__ import annotations

import sys
from pathlib import Path

from vitals_data import (
    OBSERVATION_WINDOW_HOURS,
    PREDICTION_HORIZON_HOURS,
    GAP_HOURS,
    add_personalized_features,
    build_windows,
    cohort_from_challenge2019,
    patient_level_split,
)


def run(
    psv_dir: str,
    use_gpu: bool = False,
    tune: bool = False,
    n_trials: int = 30,
    max_files: int | None = None,
    horizon_hours: int = PREDICTION_HORIZON_HOURS,
    window_hours: int = OBSERVATION_WINDOW_HOURS,
    gap_hours: int = GAP_HOURS,
    seed: int = 42,
) -> None:
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

    print(
        f"Loading Challenge-2019 PSVs from {psv_dir} (max_files={max_files}) | "
        f"window={window_hours}h horizon={horizon_hours}h gap={gap_hours}h seed={seed} ..."
    )
    cohort = cohort_from_challenge2019(psv_dir, max_files=max_files)
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
    print(f"Patients: {cohort.vitals['patient_id'].nunique()} ({n_event} with event) | "
          f"windows: {len(windowed.labels)} | positive: {positives}")
    if positives < 10:
        print("Too few positive windows — increase --max-files.")
        return

    split = patient_level_split(windowed, seed=seed)
    best = tune_xgboost(split, n_trials=n_trials, use_gpu=use_gpu, seed=seed) if tune else {}
    model, xgb = train_xgboost(split, use_gpu=use_gpu, random_state=seed, **best)
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
    def _opt(name: str, default: int | None) -> int | None:
        return next((int(a.split("=", 1)[1]) for a in sys.argv if a.startswith(f"--{name}=")), default)

    run(
        positional[0],
        use_gpu="--gpu" in sys.argv,
        tune="--tune" in sys.argv,
        n_trials=_opt("trials", 30),
        max_files=_opt("max-files", None),
        horizon_hours=_opt("horizon", PREDICTION_HORIZON_HOURS),
        window_hours=_opt("window", OBSERVATION_WINDOW_HOURS),
        gap_hours=_opt("gap", GAP_HOURS),
        seed=_opt("seed", 42),
    )


if __name__ == "__main__":
    main()
