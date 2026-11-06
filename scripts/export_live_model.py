"""Export the model the dashboard's live bedside page runs in the browser.

    python scripts/export_live_model.py            # -> web/public/live/model.json

The live page must work on a laptop with no server and no network (a
presentation hall), so the model ships as a static JSON file and is evaluated in
TypeScript (``web/src/lib/live-model.ts``). This script:

1. builds the built-in **synthetic** cohort — no patient data, nothing that
   cannot be redistributed;
2. trains the same pipeline as everywhere else (sliding windows + personal
   baseline features + cost-sensitive XGBoost), with a smaller tree budget so
   the score rises gradually instead of jumping from 0 to 1;
3. writes the trees, feature order, threshold and a few parity cases that the
   TypeScript port is checked against (``web/scripts/check_live_parity.mjs``).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vitals_data import (  # noqa: E402
    OBSERVATION_WINDOW_HOURS,
    PERSONAL_BASELINE_HOURS,
    VITALS,
    add_personalized_features,
    build_windows,
    generate_synthetic_cohort,
    patient_level_split,
)
from vitals_live import LiveReading, bin_readings, live_features  # noqa: E402
from vitals_train import threshold_at_specificity, train_xgboost  # noqa: E402

OUT = ROOT / "web" / "public" / "live" / "model.json"

# Smaller and shallower than the analysis model: the synthetic cohort is easy,
# and a full-size model saturates to 0/1, which reads as a switch, not a score.
DEMO_PARAMS = {"n_estimators": 80, "max_depth": 3, "learning_rate": 0.08, "min_child_weight": 8}


def scenario(kind: str, seconds: int = 240, seed: int = 0) -> list[dict]:
    """The same three scenarios the page offers, for parity and sanity checks."""
    rng = np.random.default_rng(seed)
    stable = {"pulse": 78, "sbp": 122, "dbp": 76, "temperature": 36.8, "spo2": 98, "resp_rate": 16}
    crash = {"pulse": 128, "sbp": 82, "dbp": 52, "temperature": 36.2, "spo2": 86, "resp_rate": 30}
    chronic = {"pulse": 112, "sbp": 100, "dbp": 62, "temperature": 37.4, "spo2": 92, "resp_rate": 22}
    out = []
    for t in range(seconds):
        if kind == "chronic":
            base = chronic
            frac = 0.0
        else:
            base = stable
            frac = 0.0 if kind == "stable" else max(0.0, min(1.0, (t - 80) / 140))
        row = {"t": float(t)}
        for vital in VITALS:
            value = base[vital] + frac * (crash[vital] - base[vital])
            row[vital] = round(float(value + rng.normal(0, 0.4)), 2)
        out.append(row)
    return out


def main() -> None:
    cohort = generate_synthetic_cohort()
    windowed = add_personalized_features(build_windows(cohort, prediction_horizon_hours=1), cohort)
    split = patient_level_split(windowed)
    model, metrics = train_xgboost(split, **DEMO_PARAMS)
    risk = model.predict_proba(split.X_test)[:, 1]
    threshold = float(threshold_at_specificity(split.y_test, risk))
    columns = list(split.X_test.columns)

    booster = model.get_booster()
    raw = json.loads(booster.save_raw("json").decode())
    learner = raw["learner"]
    trees = []
    for tree in learner["gradient_booster"]["model"]["trees"]:
        trees.append(
            {
                "left": tree["left_children"],
                "right": tree["right_children"],
                "feature": tree["split_indices"],
                "split": tree["split_conditions"],
                "default_left": tree["default_left"],
                "weight": tree["base_weights"],
                "cover": tree["sum_hessian"],
            }
        )
    base_score = float(str(learner["learner_model_param"]["base_score"]).strip("[]"))

    # Parity cases: the TypeScript port must reproduce these to the 4th decimal.
    cases = []
    for kind in ("stable", "worsening", "chronic"):
        readings = scenario(kind)
        binned = bin_readings(
            [LiveReading(t=r["t"], values={v: r[v] for v in VITALS}) for r in readings], 10.0
        )
        row, _ = live_features(binned, columns)
        cases.append(
            {
                "kind": kind,
                "readings": readings,
                "features": [None if not np.isfinite(x) else float(x) for x in row.iloc[0]],
                "risk": float(model.predict_proba(row)[0, 1]),
            }
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "about": "Demo model trained on the built-in SYNTHETIC cohort (no patient data). "
        "Same pipeline as the analysis: windows + personal-baseline deviation features + cost-sensitive XGBoost.",
        "source": "synthetic",
        "features": columns,
        "window_steps": OBSERVATION_WINDOW_HOURS,
        "baseline_steps": PERSONAL_BASELINE_HOURS,
        "threshold": round(threshold, 4),
        "base_score": base_score,
        "test_auprc": round(float(metrics.auprc), 4),
        "trees": trees,
        "parity": cases,
    }
    OUT.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB) · trees={len(trees)} · threshold={threshold:.3f}")
    for case in cases:
        print(f"  {case['kind']:<10} risk={case['risk']:.3f}")

    # Trajectory of the worsening scenario, step by step, to eyeball smoothness.
    readings = scenario("worsening")
    line = []
    for upto in range(60, 241, 20):
        sub = [LiveReading(t=r["t"], values={v: r[v] for v in VITALS}) for r in readings[:upto]]
        row, _ = live_features(bin_readings(sub, 10.0), columns)
        line.append(f"{upto}s:{model.predict_proba(row)[0, 1]:.2f}")
    print("  worsening over time:", " ".join(line))


if __name__ == "__main__":
    pd.options.mode.chained_assignment = None
    main()
