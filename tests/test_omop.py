"""Tests for the OMOP CDM explorer (network-free: tiny in-memory CDM tables)."""

from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture()
def omop_dir(tmp_path):
    """Write a minimal OMOP CDM v5.3 CSV folder for testing."""
    pd.DataFrame(
        {"concept_id": [316139, 4229440, 1308216, 3004249, 8507],
         "concept_name": ["Heart failure", "Chronic congestive heart failure", "Lisinopril", "Systolic blood pressure", "Male"],
         "domain_id": ["Condition", "Condition", "Drug", "Measurement", "Gender"],
         "vocabulary_id": ["SNOMED", "SNOMED", "RxNorm", "LOINC", "Gender"]}
    ).to_csv(tmp_path / "CONCEPT.csv", index=False)
    # 4229440 is a descendant (subtype) of Heart failure 316139.
    pd.DataFrame(
        {"ancestor_concept_id": [316139, 316139], "descendant_concept_id": [316139, 4229440],
         "min_levels_of_separation": [0, 1], "max_levels_of_separation": [0, 1]}
    ).to_csv(tmp_path / "CONCEPT_ANCESTOR.csv", index=False)
    pd.DataFrame({"person_id": [1, 2], "gender_concept_id": [8507, 8507], "year_of_birth": [1950, 1961]}).to_csv(
        tmp_path / "PERSON.csv", index=False
    )
    pd.DataFrame(
        {"person_id": [1, 2], "condition_concept_id": [4229440, 320128],
         "condition_start_date": ["2020-03-05", "2019-01-01"]}
    ).to_csv(tmp_path / "CONDITION_OCCURRENCE.csv", index=False)
    pd.DataFrame(
        {"person_id": [1], "drug_concept_id": [1308216], "drug_exposure_start_date": ["2020-03-06"]}
    ).to_csv(tmp_path / "DRUG_EXPOSURE.csv", index=False)
    pd.DataFrame(
        {"person_id": [1], "measurement_concept_id": [3004249], "measurement_date": ["2020-03-04"], "value_as_number": [95.0]}
    ).to_csv(tmp_path / "MEASUREMENT.csv", index=False)
    return tmp_path


def test_name_resolution(omop_dir):
    from omop_explore import OmopDataset

    omop = OmopDataset(omop_dir)
    assert omop.name_of(316139) == "Heart failure"
    assert omop.name_of(999999).startswith("<")  # unknown -> placeholder


def test_condition_cohort_includes_descendants(omop_dir):
    """A Heart-failure cohort must include patients coded with a subtype concept."""
    from omop_explore import OmopDataset

    omop = OmopDataset(omop_dir)
    cohort = omop.condition_cohort(316139, include_descendants=True)
    assert cohort == {1}  # person 1 has the subtype 4229440; person 2 does not

    # Without descendant expansion, the exact concept is absent -> empty cohort.
    assert omop.condition_cohort(316139, include_descendants=False) == set()


def test_patient_journey_is_ordered_and_named(omop_dir):
    from omop_explore import OmopDataset

    omop = OmopDataset(omop_dir)
    journey = omop.patient_journey(1)
    assert list(journey["domain"]) == ["Measurement", "Condition", "Drug"]  # sorted by date
    assert "Chronic congestive heart failure" in set(journey["event"])
    assert journey["date"].is_monotonic_increasing


def test_summary_counts(omop_dir):
    from omop_explore import OmopDataset

    summary = OmopDataset(omop_dir).summary()
    assert summary["person"] == 2
    assert summary["condition_occurrence"] == 2
