"""Tests for the PhysioNet Challenge 2012 adapter (no network, no real data)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vitals_data import build_windows, cohort_from_challenge2012

# One record in the documented Challenge-2012 layout: descriptors at 00:00 (with
# -1 marking "not recorded"), then irregular Time,Parameter,Value observations.
RECORD_TEMPLATE = """Time,Parameter,Value
00:00,RecordID,{record_id}
00:00,Age,{age}
00:00,Gender,{gender}
00:00,Height,-1
00:00,ICUType,4
00:00,Weight,-1
00:07,HR,{hr0}
00:07,SysABP,{sbp0}
00:07,DiasABP,70
00:07,Temp,37.0
00:07,SaO2,98
00:07,RespRate,16
00:37,HR,{hr1}
01:07,HR,{hr2}
01:07,NISysABP,{nisbp}
01:07,Temp,37.2
02:07,HR,{hr3}
02:07,SysABP,{sbp2}
02:07,SaO2,95
02:07,RespRate,22
"""

OUTCOMES = """RecordID,SAPS-I,SOFA,Length_of_stay,Survival,In-hospital_death
1001,6,1,5,1,1
1002,10,4,8,-1,1
1003,8,2,6,-1,0
"""


def _write_cohort(tmp_path):
    """Three records: a dated death, an undated death, and a survivor."""
    for i, record_id in enumerate(("1001", "1002", "1003")):
        (tmp_path / f"{record_id}.txt").write_text(
            RECORD_TEMPLATE.format(
                record_id=record_id,
                age=60 + i,
                gender=i % 2,
                hr0=80 + i, hr1=85 + i, hr2=90 + i, hr3=95 + i,
                sbp0=120 - i, sbp2=110 - i, nisbp=118 - i,
            )
        )
    (tmp_path / "Outcomes-a.txt").write_text(OUTCOMES)
    return tmp_path


def test_parses_records_into_hourly_vitals(tmp_path):
    cohort = cohort_from_challenge2012(_write_cohort(tmp_path))

    assert cohort.vitals["patient_id"].nunique() == 3
    # 00:07 and 00:37 collapse into hour 0; 01:07 -> 1; 02:07 -> 2.
    hours = sorted(cohort.vitals.loc[cohort.vitals.patient_id == "1001", "hour"].unique())
    assert hours == [0, 1, 2]

    for vital in ("pulse", "sbp", "dbp", "temperature", "spo2", "resp_rate"):
        assert vital in cohort.vitals.columns

    # Repeated same-hour readings are averaged: HR 80 and 85 in hour 0 -> 82.5
    hour0 = cohort.vitals[(cohort.vitals.patient_id == "1001") & (cohort.vitals.hour == 0)]
    assert hour0["pulse"].iloc[0] == pytest.approx(82.5)


def test_invasive_bp_preferred_over_cuff(tmp_path):
    """SysABP wins where present; NISysABP only fills the gap."""
    cohort = cohort_from_challenge2012(_write_cohort(tmp_path))
    rows = cohort.vitals[cohort.vitals.patient_id == "1001"].set_index("hour")

    assert rows.loc[0, "sbp"] == pytest.approx(120.0)  # SysABP
    assert rows.loc[1, "sbp"] == pytest.approx(118.0)  # only NISysABP at 01:07
    assert rows.loc[2, "sbp"] == pytest.approx(110.0)  # SysABP again


def test_event_times_come_from_outcomes(tmp_path):
    cohort = cohort_from_challenge2012(_write_cohort(tmp_path))
    events = cohort.events.set_index("patient_id")["arrest_hour"]

    assert events["1001"] == pytest.approx(24.0)  # died, Survival=1 day -> hour 24
    assert np.isnan(events["1002"])  # died but Survival unknown -> control
    assert np.isnan(events["1003"])  # survived -> control


def test_missing_descriptors_are_nan_not_negative_one(tmp_path):
    """Height/Weight are -1 in the raw file; -1 must not leak in as a real value."""
    cohort = cohort_from_challenge2012(_write_cohort(tmp_path))
    numeric = cohort.vitals.select_dtypes(include=[np.number])
    assert not (numeric < 0).any().any()

    demographics = cohort.demographics.set_index("patient_id")
    assert demographics.loc["1001", "age"] == pytest.approx(60.0)


def test_runs_without_outcomes_file(tmp_path):
    """Records alone still load; everyone becomes a control."""
    _write_cohort(tmp_path)
    (tmp_path / "Outcomes-a.txt").unlink()

    cohort = cohort_from_challenge2012(tmp_path)
    assert cohort.events["arrest_hour"].isna().all()


def test_outcomes_file_is_not_read_as_a_patient(tmp_path):
    cohort = cohort_from_challenge2012(_write_cohort(tmp_path))
    assert not any(pid.lower().startswith("outcomes") for pid in cohort.events["patient_id"])


def test_windows_label_from_the_event_time(tmp_path):
    """A death at hour 24 is only imminent for windows close enough to it."""
    tmp_path = _write_cohort(tmp_path)
    # Give 1001 a long enough record for hour-24 to be reachable.
    long_record = ["Time,Parameter,Value", "00:00,RecordID,1001", "00:00,Age,60", "00:00,Gender,0"]
    for hour in range(24):
        long_record.append(f"{hour:02d}:00,HR,{80 + hour % 10}")
        long_record.append(f"{hour:02d}:00,SysABP,{120 - hour % 8}")
    (tmp_path / "1001.txt").write_text("\n".join(long_record) + "\n")

    cohort = cohort_from_challenge2012(tmp_path)
    windowed = build_windows(cohort, observation_window_hours=4, prediction_horizon_hours=6)

    labels = pd.DataFrame({"label": windowed.labels, "group": windowed.groups})
    assert labels.loc[labels.group == "1001", "label"].sum() > 0  # some window is positive
    assert labels.loc[labels.group != "1001", "label"].sum() == 0  # controls stay negative


def test_max_files_limits_records(tmp_path):
    cohort = cohort_from_challenge2012(_write_cohort(tmp_path), max_files=2)
    assert cohort.events["patient_id"].nunique() == 2


def test_missing_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        cohort_from_challenge2012(tmp_path / "nope")
