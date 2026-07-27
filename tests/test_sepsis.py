"""Tests for the PhysioNet Challenge 2019 adapter (network-free tiny PSVs)."""

from __future__ import annotations

import pandas as pd


def _write_psv(directory, name, rows):
    pd.DataFrame(rows).to_csv(directory / f"{name}.psv", sep="|", index=False)


def test_cohort_from_challenge2019(tmp_path):
    from vitals_data import VITALS, cohort_from_challenge2019

    # Patient p1: sepsis onset (SepsisLabel 0,0,1,1) -> event at hour index 2.
    _write_psv(
        tmp_path, "p000001",
        [
            {"HR": 80, "O2Sat": 98, "Temp": 36.8, "SBP": 120, "DBP": 75, "Resp": 16, "Age": 70, "Gender": 1, "ICULOS": 1, "SepsisLabel": 0},
            {"HR": 88, "O2Sat": 96, "Temp": 37.0, "SBP": 112, "DBP": 70, "Resp": 20, "Age": 70, "Gender": 1, "ICULOS": 2, "SepsisLabel": 0},
            {"HR": 110, "O2Sat": 90, "Temp": 37.6, "SBP": 95, "DBP": 60, "Resp": 28, "Age": 70, "Gender": 1, "ICULOS": 3, "SepsisLabel": 1},
            {"HR": 120, "O2Sat": 88, "Temp": 38.0, "SBP": 90, "DBP": 55, "Resp": 30, "Age": 70, "Gender": 1, "ICULOS": 4, "SepsisLabel": 1},
        ],
    )
    # Patient p2: never septic -> control.
    _write_psv(
        tmp_path, "p000002",
        [{"HR": 78, "O2Sat": 98, "Temp": 36.7, "SBP": 122, "DBP": 76, "Resp": 15, "Age": 55, "Gender": 0, "ICULOS": 1, "SepsisLabel": 0}],
    )

    cohort = cohort_from_challenge2019(tmp_path)
    assert list(cohort.vitals.columns) == ["patient_id", "hour", *VITALS]
    assert cohort.vitals["hour"].min() == 0  # ICULOS normalized to start at 0

    events = cohort.events.set_index("patient_id")["arrest_hour"]
    assert events["p000001"] == 2.0  # first SepsisLabel==1 (ICULOS 3 -> hour index 2)
    assert pd.isna(events["p000002"])  # control

    # HR maps to pulse; Gender -> sex demographic carried through.
    assert cohort.vitals[cohort.vitals["patient_id"] == "p000001"]["pulse"].iloc[0] == 80
    assert cohort.demographics.set_index("patient_id").loc["p000001", "sex"] == 1


def test_cohort_from_challenge2019_max_files(tmp_path):
    from vitals_data import cohort_from_challenge2019

    for i in range(5):
        _write_psv(tmp_path, f"p{i:06d}", [{"HR": 80, "SBP": 120, "DBP": 75, "Temp": 37, "O2Sat": 98, "Resp": 16, "ICULOS": 1, "SepsisLabel": 0}])
    cohort = cohort_from_challenge2019(tmp_path, max_files=3)
    assert cohort.vitals["patient_id"].nunique() == 3
