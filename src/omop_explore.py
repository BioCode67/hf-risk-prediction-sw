"""Hands-on OMOP CDM explorer for the K-Health 심부전(HF) competition track.

The competition's real heart-failure data lives only inside the offline 안심존,
and the 예선 sample is Synthea-synthetic OMOP CDM. To learn — and prototype —
the exact workflow beforehand, this module works on *any* OMOP CDM CSV folder:

- it can download a public OMOP sample (OHDSI Eunomia, e.g. ``GiBleed`` — 2,694
  synthetic patients with full condition/drug/measurement tables), and
- it exercises the core moves you repeat in the 안심존: resolve integer
  ``*_concept_id``s to names via the ``concept`` table, build a condition cohort
  (with descendant concepts), and reconstruct a patient's journey across the
  clinical tables.

On the real 안심존 data you change nothing but the folder path (and the anchor
concept, e.g. Heart failure = 316139). Standard library + pandas only — no
network dependency at import time, nothing that needs the closed network.
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Final, Iterable

import pandas as pd

EUNOMIA_BASE: Final[str] = "https://raw.githubusercontent.com/OHDSI/EunomiaDatasets/main/datasets"

# A few standard OMOP concept ids handy for a heart-failure cohort.
HEART_FAILURE_CONCEPT_ID: Final[int] = 316139
GENDER_CONCEPTS: Final[dict[int, str]] = {8507: "Male", 8532: "Female"}


def download_eunomia(dataset: str = "GiBleed", cdm_version: str = "5.3", dest_dir: str | Path = "data/omop") -> Path:
    """Download and unzip a public OHDSI Eunomia OMOP dataset.

    Returns the folder containing the CDM CSV files. Downloaded data is
    git-ignored (see ``.gitignore``); on a machine with normal internet this
    just works, no credentials required.
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    url = f"{EUNOMIA_BASE}/{dataset}/{dataset}_{cdm_version}.zip"
    print(f"Downloading {url} ...")
    with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310 (trusted OHDSI host)
        payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        archive.extractall(dest)
    # Eunomia zips nest the CSVs under a ``<dataset>_<version>`` folder.
    csv_dir = next((p.parent for p in dest.rglob("PERSON.csv")), dest)
    print(f"Extracted OMOP CSVs to {csv_dir}")
    return csv_dir


@dataclass
class OmopDataset:
    """Lazy reader over a folder of OMOP CDM v5.x CSV tables (case-insensitive)."""

    csv_dir: str | Path

    def _load(self, table: str) -> pd.DataFrame:
        """Load a CDM table by name (tries upper/lowercase filenames)."""
        directory = Path(self.csv_dir)
        for name in (f"{table.upper()}.csv", f"{table.lower()}.csv"):
            path = directory / name
            if path.exists():
                frame = pd.read_csv(path, low_memory=False)
                frame.columns = frame.columns.str.lower()
                return frame
        raise FileNotFoundError(f"OMOP table '{table}' not found in {directory}")

    @cached_property
    def concept(self) -> pd.DataFrame:
        return self._load("concept")

    @cached_property
    def _concept_name(self) -> dict[int, str]:
        return dict(zip(self.concept["concept_id"], self.concept["concept_name"]))

    def name_of(self, concept_id: int) -> str:
        """Resolve a ``*_concept_id`` integer to its human-readable name."""
        return self._concept_name.get(int(concept_id), f"<{concept_id}>")

    def table(self, name: str) -> pd.DataFrame:
        """Public accessor for any CDM table (lowercased columns)."""
        return self._load(name)

    def descendants(self, concept_ids: Iterable[int]) -> set[int]:
        """Expand concept ids to include descendants via ``concept_ancestor``."""
        anchors = {int(c) for c in concept_ids}
        try:
            ancestor = self._load("concept_ancestor")
        except FileNotFoundError:
            return anchors
        kids = ancestor[ancestor["ancestor_concept_id"].isin(anchors)]["descendant_concept_id"]
        return anchors | set(kids.astype(int))

    def condition_cohort(self, anchor_concept_id: int, include_descendants: bool = True) -> set[int]:
        """Return person_ids diagnosed with a condition (optionally + subtypes)."""
        targets = self.descendants([anchor_concept_id]) if include_descendants else {int(anchor_concept_id)}
        cond = self._load("condition_occurrence")
        return set(cond[cond["condition_concept_id"].isin(targets)]["person_id"].astype(int))

    def patient_journey(self, person_id: int) -> pd.DataFrame:
        """Reconstruct one patient's timeline across the clinical tables."""
        events: list[pd.DataFrame] = []
        specs = [
            ("condition_occurrence", "condition_concept_id", "condition_start_date", "Condition"),
            ("drug_exposure", "drug_concept_id", "drug_exposure_start_date", "Drug"),
            ("procedure_occurrence", "procedure_concept_id", "procedure_date", "Procedure"),
            ("measurement", "measurement_concept_id", "measurement_date", "Measurement"),
        ]
        for table, concept_col, date_col, domain in specs:
            try:
                frame = self._load(table)
            except FileNotFoundError:
                continue
            rows = frame[frame["person_id"].astype(int) == int(person_id)]
            if rows.empty or date_col not in rows.columns:
                continue
            events.append(
                pd.DataFrame(
                    {
                        "date": rows[date_col],
                        "domain": domain,
                        "event": rows[concept_col].map(self.name_of),
                    }
                )
            )
        if not events:
            return pd.DataFrame(columns=["date", "domain", "event"])
        journey = pd.concat(events, ignore_index=True)
        journey["date"] = pd.to_datetime(journey["date"], errors="coerce")
        return journey.sort_values("date").reset_index(drop=True)

    def summary(self) -> dict[str, int]:
        """Row counts for the main clinical tables (a quick sanity check)."""
        out: dict[str, int] = {}
        for table in ("person", "visit_occurrence", "condition_occurrence", "drug_exposure", "measurement"):
            try:
                out[table] = len(self._load(table))
            except FileNotFoundError:
                out[table] = 0
        return out

    def top_conditions(self, n: int = 10) -> pd.DataFrame:
        """Most common conditions by number of distinct patients."""
        cond = self._load("condition_occurrence")
        counts = cond.groupby("condition_concept_id")["person_id"].nunique().sort_values(ascending=False).head(n)
        return pd.DataFrame(
            {"condition": [self.name_of(c) for c in counts.index], "patients": counts.to_numpy()}
        )


def main() -> None:
    """Download GiBleed and demonstrate the OMOP workflow end-to-end."""
    project_root = Path(__file__).resolve().parent.parent
    csv_dir = download_eunomia(dataset="GiBleed", cdm_version="5.3", dest_dir=project_root / "data" / "omop")
    omop = OmopDataset(csv_dir)

    print("\n=== Table row counts ===")
    for table, count in omop.summary().items():
        print(f"  {table}: {count:,}")

    print("\n=== Top conditions (by #patients) ===")
    print(omop.top_conditions(8).to_string(index=False))

    # Pick the most common condition as an example cohort anchor. On the real
    # 안심존 data, swap in Heart failure (316139) instead.
    top_concept = int(omop.table("condition_occurrence")["condition_concept_id"].value_counts().index[0])
    cohort = omop.condition_cohort(top_concept)
    print(f"\n=== Example cohort: '{omop.name_of(top_concept)}' -> {len(cohort):,} patients ===")
    print("(On the competition data, use condition_concept_id=316139 'Heart failure'.)")

    sample_id = sorted(cohort)[0]
    journey = omop.patient_journey(sample_id)
    print(f"\n=== Patient {sample_id} journey (first 12 events) ===")
    print(journey.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
