"""Tests for the MIMIC-IV Demo explorer (network-free: tiny fake ICU tables)."""

from __future__ import annotations

import pandas as pd


def _write_demo(root):
    """Write a minimal MIMIC-IV Demo ICU folder for testing."""
    icu = root / "icu"
    icu.mkdir()
    pd.DataFrame(
        {"itemid": [220045, 220179, 220180, 220277, 220210, 223762, 999999, 225466],
         "label": ["Heart Rate", "NBP systolic", "NBP diastolic", "SpO2", "Resp Rate", "Temp C", "Glucose", "Cardiac Arrest"],
         "linksto": ["chartevents"] * 7 + ["procedureevents"]}
    ).to_csv(icu / "d_items.csv", index=False)
    pd.DataFrame({"subject_id": [1], "stay_id": [30001], "intime": ["2150-01-01 00:00:00"]}).to_csv(
        icu / "icustays.csv", index=False
    )
    rows = []
    for hour in range(3):
        stamp = f"2150-01-01 0{hour}:00:00"
        rows += [
            {"stay_id": 30001, "charttime": stamp, "itemid": 220045, "valuenum": 80 + hour},
            {"stay_id": 30001, "charttime": stamp, "itemid": 220277, "valuenum": 98},
            {"stay_id": 30001, "charttime": stamp, "itemid": 999999, "valuenum": 140},  # non-vital, filtered out
        ]
    pd.DataFrame(rows).to_csv(icu / "chartevents.csv", index=False)
    return root


def test_load_mimic_demo_filters_to_vitals(tmp_path):
    from mimic_explore import load_mimic_demo

    tables = load_mimic_demo(_write_demo(tmp_path))
    vitals = tables["chartevents_vitals"]
    # Only vital itemids survive (the 999999 glucose rows are dropped).
    assert set(vitals["itemid"]) == {220045, 220277}
    assert len(tables["icustays"]) == 1


def test_vital_itemid_coverage_reports_labels(tmp_path):
    from mimic_explore import load_mimic_demo, vital_itemid_coverage

    tables = load_mimic_demo(_write_demo(tmp_path))
    coverage = vital_itemid_coverage(tables["d_items"])
    present = dict(zip(coverage["itemid"], coverage["mimic_label"]))
    assert present[220045] == "Heart Rate"
    assert present[220051] == "<absent>"  # DBP arterial line not in this tiny d_items


def test_scan_arrest_items_finds_markers(tmp_path):
    from mimic_explore import scan_arrest_items

    found = scan_arrest_items(_write_demo(tmp_path))
    assert 225466 in set(found["itemid"])  # 'Cardiac Arrest' matched
    assert 220045 not in set(found["itemid"])  # 'Heart Rate' not an arrest marker


def test_mimic_arrest_events_takes_earliest_per_stay(tmp_path):
    from mimic_explore import mimic_arrest_events

    events = pd.DataFrame(
        {
            "stay_id": [30001, 30001, 30002],
            "charttime": ["2150-01-01 05:00:00", "2150-01-01 03:00:00", "2150-01-02 10:00:00"],
            "itemid": [225466, 225466, 225466],
        }
    )
    arrests = mimic_arrest_events(events, [225466])
    assert dict(zip(arrests["stay_id"], arrests["arrest_time"].astype(str)))[30001] == "2150-01-01 03:00:00"
    assert len(arrests) == 2


def test_mimic_vitals_flow_into_cohort(tmp_path):
    """Real-shaped MIMIC vitals adapt into a cohort the pipeline can window."""
    from mimic_explore import load_mimic_demo
    from vitals_data import VITALS, cohort_from_mimic

    tables = load_mimic_demo(_write_demo(tmp_path))
    cohort = cohort_from_mimic(tables["chartevents_vitals"])
    assert list(cohort.vitals.columns) == ["patient_id", "hour", *VITALS]
    assert cohort.vitals["pulse"].tolist() == [80, 81, 82]
    assert cohort.events["arrest_hour"].isna().all()  # demo stays are controls
