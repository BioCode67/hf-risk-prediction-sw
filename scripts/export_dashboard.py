"""Bake the dashboard into static JSON so deployment needs no ML runtime.

The dashboard only ever *reads back* an analysis of a fixed retrospective
cohort: which held-out patients were flagged, when, and why. Nothing about that
changes at request time, so every answer can be computed once, here, and served
as files.

That removes the whole reason the service was heavy. Scoring a window needs
xgboost (a 226 MB compiled library, most of it CUDA kernels) and explaining one
needs shap (which drags in numba and a 171 MB LLVM). Neither is needed to hand
back a number that was already computed. What ships instead is JSON that any
static host serves for free.

    python scripts/export_dashboard.py \\
        --artifact models/dashboard_challenge2019.pkl \\
        --out web/public/data

Writes ``index.json`` (cohort figures + the patient list) and one
``patients/<id>.json`` per patient. The per-patient payload matches the shapes
in ``web/src/lib/types.ts`` exactly, so the frontend reads files instead of
endpoints and nothing else changes.

Re-run this whenever the model, the cohort or the operating threshold changes —
the output is a snapshot, not a live view.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402  (path setup must precede project imports)

import vitals_api  # noqa: E402
from vitals_data import VITALS  # noqa: E402
from vitals_narrate import VITAL_KO  # noqa: E402


def _json_default(value: Any) -> Any:
    """Make numpy scalars serialisable; anything else is a bug worth raising on."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    raise TypeError(f"cannot serialise {type(value)!r}")


def _write(path: Path, payload: Any) -> int:
    """Write compact JSON and return the byte count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=_json_default)
    path.write_text(text, encoding="utf-8")
    return len(text.encode("utf-8"))


def export(artifact: Path, out_dir: Path) -> None:
    """Serialise one prebuilt Service into the static file tree.

    Goes through ``vitals_api``'s own loader and route functions rather than
    re-deriving anything, so the files cannot drift from what the live API
    returns — the frontend has to read both against one set of types.
    """
    os.environ["EWS_ARTIFACT"] = str(artifact)
    vitals_api.service.cache_clear()
    svc = vitals_api.service()

    patient_ids = sorted(set(svc.patient_ids.astype(str)))
    print(f"artifact : {artifact}  ({artifact.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"patients : {len(patient_ids)}   windows: {len(svc.risk):,}")

    summaries = [
        json.loads(summary.model_dump_json())
        for summary in vitals_api.patients(limit=len(patient_ids))
    ]

    index = {
        "source": svc.source,
        "threshold": round(svc.threshold, 3),
        "news_threshold": vitals_api.NEWS_ALARM_THRESHOLD,
        "cohort": {
            "patients": svc.cohort_patients,
            "windows": svc.cohort_windows,
            "positive_rate": svc.cohort_positive_rate,
            "alarming": svc.cohort_alarming,
        },
        "browsable": len(patient_ids),
        "burden": svc.burden,
        # Shared once instead of repeated in every hour's payload.
        "vitals": [
            {"vital": vital, "label": VITAL_KO[vital][0], "unit": VITAL_KO[vital][1]}
            for vital in VITALS
        ],
        "patients": summaries,
    }
    total = _write(out_dir / "index.json", index)
    print(f"index.json written ({total / 1024:.0f} KB)")

    started = time.time()
    for position, patient_id in enumerate(patient_ids, start=1):
        rows = svc.rows_for(patient_id)
        demographics = svc.demographics.get(patient_id, {})
        last = int(rows[-1])
        arrest = svc.hours[last] + svc.time_to_arrest[last]

        payload = {
            "patient_id": patient_id,
            "age": demographics.get("age"),
            "sex": demographics.get("sex"),
            "hours_observed": len(rows),
            "threshold": round(svc.threshold, 3),
            "news_threshold": vitals_api.NEWS_ALARM_THRESHOLD,
            "arrest_hour": float(arrest) if np.isfinite(arrest) else None,
            "baseline": svc.baselines.get(patient_id, {}),
            "trajectory": svc.trajectories.get(patient_id, []),
            "timeline": [
                {
                    "hour": int(svc.hours[row]),
                    "risk": round(float(svc.risk[row]), 3),
                    "news_score": float(svc.news[row]),
                }
                for row in rows
            ],
            # Every hour the user can click, explained ahead of time.
            "evidence": {
                str(int(svc.hours[row])): json.loads(
                    vitals_api._evidence(svc, patient_id, int(row)).model_dump_json()
                )
                for row in rows
            },
        }
        total += _write(out_dir / "patients" / f"{patient_id}.json", payload)

        if position % 50 == 0 or position == len(patient_ids):
            elapsed = time.time() - started
            rate = position / elapsed
            print(
                f"  {position:>4}/{len(patient_ids)}  "
                f"{total / 1024 / 1024:>5.1f} MB  "
                f"{elapsed:>4.0f}s  (eta {(len(patient_ids) - position) / rate:>4.0f}s)"
            )

    print(f"\ntotal {total / 1024 / 1024:.1f} MB in {out_dir}")
    print("no xgboost, no shap, no python needed to serve this")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        type=Path,
        default=REPO_ROOT / "models" / "dashboard_challenge2019.pkl",
        help="prebuilt Service pickle (see docs/deploy.md)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "web" / "public" / "data",
        help="output directory, served as static files",
    )
    args = parser.parse_args()

    if not args.artifact.exists():
        raise SystemExit(f"artifact not found: {args.artifact}\nBuild it first — see docs/deploy.md")
    export(args.artifact, args.out)


if __name__ == "__main__":
    main()
