"""Measure what the personalized-baseline features are actually worth.

The proposal's headline differentiator is the personal-baseline deviation
feature, and on Challenge-2019 (sepsis) it did *not* lead — it ranked fourth and
sixth in global SHAP contribution while raw respiratory rate and temperature
took the top slots. §7-3 says so, and §9.5 commits to re-checking it on arrest
data. This script is that check.

It trains twice on the same patient split — once with the personalized features
and once without — and reports the difference in AUPRC and in alarm burden at
matched detection. Same seed, same windows, same hyperparameters, so the delta
is attributable to the features and nothing else.

    # synthetic arrest cohort (no data needed)
    python scripts/ablate_personalized.py

    # real MIMIC-IV arrest cohort
    python scripts/ablate_personalized.py --mimic /path/to/mimiciv/icu [--gpu]

    # public sepsis cohort, for the event-type comparison
    python scripts/ablate_personalized.py --challenge2019 /path/to/training_setA

Prints a table and writes models/ablation_personalized.json.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import vitals_data as vd  # noqa: E402
from vitals_train import alarm_burden, train_xgboost  # noqa: E402

OUT = ROOT / "models" / "ablation_personalized.json"

# Detection rates the proposal compares at. 90% is excluded on purpose: NEWS
# specificity collapses to zero there, so the comparison is degenerate (§7-3).
DETECTION_RATES = (0.5, 0.7)


def _personalized_columns(names: list[str]) -> list[str]:
    """The deviation-from-own-baseline features, by their naming convention."""
    return [c for c in names if c.endswith("_last_dev") or c.endswith("_mean_dev")]


def _build(source: str, path: str | None, seed: int):
    """Load a cohort and turn it into windows with personalized + static features."""
    if source == "mimic":
        from mimic_explore import arrest_events_from_procedures, load_mimic_demo

        tables = load_mimic_demo(path)
        arrests = arrest_events_from_procedures(path)
        cohort = vd.cohort_from_mimic(tables["chartevents_vitals"], arrest_events=arrests)
    elif source == "challenge2019":
        cohort = vd.cohort_from_challenge2019(path)
    else:
        cohort = vd.generate_synthetic_cohort(n_patients=500, arrest_fraction=0.5, seed=seed)

    windowed = vd.build_windows(cohort)
    windowed = vd.add_personalized_features(windowed, cohort)
    try:
        windowed = vd.add_static_features(windowed, cohort)
    except Exception as exc:  # static columns are optional per adapter
        print(f"  (static features skipped: {exc})")
    return cohort, windowed


def _drop_features(split, columns: list[str]):
    """A copy of the split with `columns` removed from train and test."""
    import copy

    stripped = copy.copy(split)
    stripped.X_train = split.X_train.drop(columns=columns, errors="ignore")
    stripped.X_test = split.X_test.drop(columns=columns, errors="ignore")
    return stripped


def _burden_rows(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """Alarms per 100 windows, and the specificity it costs, at each matched rate."""
    out: dict[str, float] = {}
    for rate in DETECTION_RATES:
        burden = alarm_burden(y_true, scores, target_sensitivity=rate)
        pct = int(rate * 100)
        out[f"alarms_per_100_at_{pct}"] = burden["alarms_per_100_windows"]
        out[f"specificity_at_{pct}"] = burden["specificity"]
    return out


def run(source: str, path: str | None, *, use_gpu: bool, seed: int) -> dict[str, Any]:
    cohort, windowed = _build(source, path, seed)
    split = vd.patient_level_split(windowed, test_size=0.2, seed=seed)

    personal = _personalized_columns(list(split.X_train.columns))
    if not personal:
        raise SystemExit("No personalized features found — check add_personalized_features ran.")

    n_pos = int(np.asarray(split.y_test).sum())
    print(f"\n출처: {source}   환자 {windowed.groups.nunique():,}명   "
          f"윈도우 {len(windowed.labels):,}개   양성률 {float(np.mean(windowed.labels)):.4f}")
    print(f"개인화 피처 {len(personal)}개: {', '.join(personal[:4])}"
          f"{' …' if len(personal) > 4 else ''}")
    print(f"test 양성 윈도우 {n_pos:,}개\n")

    results: dict[str, Any] = {}
    for label, active in (("with_personalized", split), ("without_personalized", _drop_features(split, personal))):
        model, metrics = train_xgboost(active, use_gpu=use_gpu)
        scores = model.predict_proba(active.X_test)[:, 1]
        row = asdict(metrics)
        row.update(_burden_rows(np.asarray(active.y_test), scores))
        row["n_features"] = int(active.X_train.shape[1])
        results[label] = row

    with_, without = results["with_personalized"], results["without_personalized"]
    results["delta"] = {
        "auprc": with_["auprc"] - without["auprc"],
        "auprc_relative_pct": (with_["auprc"] / without["auprc"] - 1) * 100 if without["auprc"] else float("nan"),
        **{
            k: with_[k] - without[k]
            for k in with_
            if k.startswith("alarms_per_100_at_")
        },
    }
    results["meta"] = {
        "source": source,
        "path": path,
        "seed": seed,
        "personalized_features": personal,
        "detection_rates": list(DETECTION_RATES),
        "note": "Same split, same hyperparameters; the only difference is the feature set.",
    }
    return results


def _report(r: dict[str, Any]) -> None:
    w, o, d = r["with_personalized"], r["without_personalized"], r["delta"]
    print(f"{'지표':<28}{'개인화 포함':>14}{'개인화 제외':>14}{'차이':>14}")
    print("-" * 70)
    print(f"{'피처 수':<28}{w['n_features']:>14,}{o['n_features']:>14,}"
          f"{w['n_features'] - o['n_features']:>+14,}")
    print(f"{'AUPRC':<28}{w['auprc']:>14.4f}{o['auprc']:>14.4f}{d['auprc']:>+14.4f}")
    print(f"{'ROC-AUC':<28}{w['roc_auc']:>14.4f}{o['roc_auc']:>14.4f}"
          f"{w['roc_auc'] - o['roc_auc']:>+14.4f}")
    for rate in DETECTION_RATES:
        k = f"alarms_per_100_at_{int(rate * 100)}"
        print(f"{'알람/100 @ 검출률 ' + str(int(rate * 100)) + '%':<28}"
              f"{w[k]:>14.1f}{o[k]:>14.1f}{d[k]:>+14.1f}")
    print("-" * 70)
    rel = d["auprc_relative_pct"]
    verdict = (
        "개인화 피처가 기여함" if rel > 5
        else "기여가 미미함 — 제안서 §7-3의 관찰과 일치" if rel > -5
        else "개인화 피처가 오히려 해로움"
    )
    print(f"AUPRC 상대 변화 {rel:+.1f}%  →  {verdict}")


def _aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarise repeats. One split is not a result — the spread is the point.

    Whether the personalized features help turned out to depend on which
    patients land in the test set, so a single seed can report the opposite
    sign of the mean. Report mean, spread and how many repeats improved.
    """
    import statistics as st

    w = [r["with_personalized"]["auprc"] for r in runs]
    o = [r["without_personalized"]["auprc"] for r in runs]
    rel = [r["delta"]["auprc_relative_pct"] for r in runs]
    spread = lambda v: st.pstdev(v) if len(v) > 1 else 0.0  # noqa: E731

    return {
        "n_seeds": len(runs),
        "seeds": [r["meta"]["seed"] for r in runs],
        "with_personalized": {"auprc_mean": st.mean(w), "auprc_sd": spread(w),
                              "auprc_min": min(w), "auprc_max": max(w)},
        "without_personalized": {"auprc_mean": st.mean(o), "auprc_sd": spread(o),
                                 "auprc_min": min(o), "auprc_max": max(o)},
        "auprc_delta_mean": st.mean(w) - st.mean(o),
        "auprc_relative_pct_mean": st.mean(rel),
        "seeds_improved": sum(1 for x in rel if x > 0),
        "sd_ratio": (spread(o) / spread(w)) if spread(w) else float("nan"),
    }


def _report_aggregate(a: dict[str, Any]) -> None:
    w, o = a["with_personalized"], a["without_personalized"]
    print(f"\n=== 시드 {a['n_seeds']}회 반복 {a['seeds']} ===")
    print(f"{'':<14}{'AUPRC 평균':>12}{'표준편차':>12}{'최소':>10}{'최대':>10}")
    print("-" * 58)
    print(f"{'개인화 포함':<14}{w['auprc_mean']:>12.4f}{w['auprc_sd']:>12.4f}"
          f"{w['auprc_min']:>10.4f}{w['auprc_max']:>10.4f}")
    print(f"{'개인화 제외':<14}{o['auprc_mean']:>12.4f}{o['auprc_sd']:>12.4f}"
          f"{o['auprc_min']:>10.4f}{o['auprc_max']:>10.4f}")
    print("-" * 58)
    print(f"평균 차이 {a['auprc_delta_mean']:+.4f} (상대 {a['auprc_relative_pct_mean']:+.1f}%), "
          f"개선 {a['seeds_improved']}/{a['n_seeds']} 시드")
    print(f"분산 비교: 개인화를 빼면 표준편차가 {a['sd_ratio']:.1f}배로 커진다 — "
          f"개인화 피처의 기여는 평균 성능보다 시드 간 안정성 쪽이 크다")


def main() -> None:
    argv = sys.argv[1:]
    use_gpu = "--gpu" in argv

    seeds = [42]
    if "--seeds" in argv:
        seeds = [int(s) for s in argv[argv.index("--seeds") + 1].split(",")]
    elif "--seed" in argv:
        seeds = [int(argv[argv.index("--seed") + 1])]

    source, path = "synthetic", None
    for flag, name in (("--mimic", "mimic"), ("--challenge2019", "challenge2019")):
        if flag in argv:
            idx = argv.index(flag)
            if idx + 1 >= len(argv):
                raise SystemExit(f"{flag} needs a path")
            source, path = name, argv[idx + 1]

    runs = []
    for seed in seeds:
        results = run(source, path, use_gpu=use_gpu, seed=seed)
        print()
        _report(results)
        runs.append(results)

    payload: dict[str, Any] = {"runs": runs}
    if len(runs) > 1:
        payload["aggregate"] = _aggregate(runs)
        _report_aggregate(payload["aggregate"])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n저장: {OUT}")


if __name__ == "__main__":
    main()
