"""Explore the MIMIC-IV Clinical Database **Demo** for the early-warning track.

MIMIC-IV is the real, public ICU database we use to *develop and validate* the
cardiac-arrest early-warning pipeline before the offline 안심존. The **Demo**
(100 patients) needs **no credentialing** — download it directly and unzip:

    https://physionet.org/content/mimic-iv-demo/    (Open Data license)
    # e.g.
    wget -r -N -c -np https://physionet.org/files/mimic-iv-demo/2.2/
    #   or grab the single zip and unzip it, then point this script at the folder

Then:

    python src/mimic_explore.py /path/to/mimic-iv-clinical-database-demo-2.2

It summarizes the ICU tables, shows which vital-sign ``itemid``s are present, and
feeds them straight into ``vitals_data.cohort_from_mimic`` so you can see the real
data flow through the exact pipeline used on the competition data. The full
MIMIC-IV (with enough cardiac-arrest events to *model*) needs free CITI
credentialing; the Demo is for understanding structure and validating the loader.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from vitals_data import MIMIC_ITEMID_MAP, VITALS, build_windows, cohort_from_mimic


def _read_table(icu_dir: Path, name: str, **kwargs) -> pd.DataFrame:
    """Read an ICU table as ``.csv`` or ``.csv.gz`` (whichever exists)."""
    for suffix in (".csv.gz", ".csv"):
        path = icu_dir / f"{name}{suffix}"
        if path.exists():
            return pd.read_csv(path, **kwargs)
    raise FileNotFoundError(f"MIMIC ICU table '{name}' not found under {icu_dir}")


def load_mimic_demo(root: str | Path) -> dict[str, pd.DataFrame]:
    """Load the ICU module tables needed for vital-sign early warning.

    Returns a dict with ``icustays``, ``d_items`` and the vital-sign subset of
    ``chartevents`` (only the itemids in :data:`MIMIC_ITEMID_MAP`).
    """
    icu_dir = Path(root) / "icu"
    if not icu_dir.exists():
        icu_dir = Path(root)  # allow pointing directly at the icu folder
    d_items = _read_table(icu_dir, "d_items")
    icustays = _read_table(icu_dir, "icustays")

    vital_ids = set(MIMIC_ITEMID_MAP)
    chunks = _read_table(
        icu_dir,
        "chartevents",
        usecols=["stay_id", "charttime", "itemid", "valuenum"],
        chunksize=500_000,
    )
    vitals = pd.concat([chunk[chunk["itemid"].isin(vital_ids)] for chunk in chunks], ignore_index=True)
    return {"icustays": icustays, "d_items": d_items, "chartevents_vitals": vitals}


def vital_itemid_coverage(d_items: pd.DataFrame) -> pd.DataFrame:
    """Show which modelled vitals are present, with their MIMIC labels."""
    labels = dict(zip(d_items["itemid"], d_items["label"]))
    rows = [
        {"itemid": itemid, "vital": vital, "mimic_label": labels.get(itemid, "<absent>")}
        for itemid, vital in MIMIC_ITEMID_MAP.items()
    ]
    return pd.DataFrame(rows).sort_values("vital").reset_index(drop=True)


# Keywords that flag a candidate in-hospital cardiac-arrest / resuscitation item.
ARREST_KEYWORDS: tuple[str, ...] = (
    "arrest",
    "cpr",
    "resuscitat",
    "compression",
    "defibrill",
    "code blue",
    "epineph",
)


def scan_arrest_items(root: str | Path) -> pd.DataFrame:
    """Search ``d_items`` for candidate cardiac-arrest / resuscitation markers.

    IHCA is not a single flag in MIMIC-IV, so we surface every item whose label
    matches an arrest/CPR keyword (with its ``linksto`` table) and let a human
    pick the definition. Run this on your real demo and share the result.
    """
    icu_dir = Path(root) / "icu"
    if not icu_dir.exists():
        icu_dir = Path(root)
    d_items = _read_table(icu_dir, "d_items")
    label = d_items["label"].fillna("").str.lower()
    mask = pd.Series(False, index=d_items.index)
    for keyword in ARREST_KEYWORDS:
        mask = mask | label.str.contains(keyword, regex=False)
    columns = [c for c in ("itemid", "label", "abbreviation", "linksto", "category") if c in d_items.columns]
    return d_items.loc[mask, columns].reset_index(drop=True)


def mimic_arrest_events(
    events: pd.DataFrame,
    arrest_itemids: list[int],
    *,
    id_col: str = "stay_id",
    time_col: str = "charttime",
    item_col: str = "itemid",
) -> pd.DataFrame:
    """Build ``[stay_id, arrest_time]`` from the chosen arrest-marker itemids.

    Takes the *earliest* matching event per stay as the arrest time. Feed the
    result to ``vitals_data.cohort_from_mimic(chartevents, arrest_events=...)``.
    """
    hits = events[events[item_col].isin(set(arrest_itemids))].copy()
    hits[time_col] = pd.to_datetime(hits[time_col], errors="coerce")
    arrest = hits.groupby(id_col)[time_col].min().rename("arrest_time").reset_index()
    return arrest


def main(root: str | None = None) -> None:
    """Summarize the MIMIC-IV Demo and run it through the early-warning loader."""
    import sys

    positional = [a for a in sys.argv[1:] if not a.startswith("--")]
    root = root or (positional[0] if positional else None)
    if root is None:
        print(__doc__)
        print("Usage: python src/mimic_explore.py /path/to/mimic-iv-clinical-database-demo-2.2 [--scan-arrest]")
        return

    if "--scan-arrest" in sys.argv:
        print("=== Candidate cardiac-arrest / resuscitation items (from d_items) ===")
        found = scan_arrest_items(root)
        print(found.to_string(index=False) if len(found) else "(no keyword matches — full MIMIC-IV likely needed)")
        return

    tables = load_mimic_demo(root)
    stays, d_items, vitals = tables["icustays"], tables["d_items"], tables["chartevents_vitals"]
    print(f"ICU stays: {len(stays):,} | vital chartevents rows: {len(vitals):,}")

    print("\n=== Vital-sign itemid coverage ===")
    print(vital_itemid_coverage(d_items).to_string(index=False))

    # Flow the real vitals through the exact competition pipeline (no arrest
    # labels in the demo -> all treated as controls; full MIMIC-IV adds events).
    cohort = cohort_from_mimic(vitals)
    print(f"\nCohort: {cohort.vitals['patient_id'].nunique():,} ICU stays, {len(cohort.vitals):,} hourly rows")

    print("\n=== Vital value distribution (hourly, after bucketing) ===")
    stats = cohort.vitals[list(VITALS)].describe(percentiles=[0.5]).T[["count", "mean", "std", "min", "50%", "max"]]
    stats["missing_%"] = (1 - cohort.vitals[list(VITALS)].notna().mean()) * 100
    print(stats.round(2).to_string())

    windowed = build_windows(cohort)
    print(f"\nWindows: {len(windowed.features):,} x {len(windowed.feature_names)} features")

    sample = cohort.vitals["patient_id"].iloc[0]
    print(f"\n=== Sample stay {sample} (first hours) ===")
    print(cohort.vitals[cohort.vitals["patient_id"] == sample][["hour", *VITALS]].head(8).to_string(index=False))


if __name__ == "__main__":
    main()
