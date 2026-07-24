"""Tests for the neural sequence-model prototype (src/vitals_seq.py).

The sequence-construction tests are torch-free and always run. The training test
skips automatically when PyTorch (the optional GPU-track dependency) is absent, so
CI stays green on the lightweight package set.
"""

from __future__ import annotations

import numpy as np
import pytest


def test_build_sequences_shape_and_labels():
    """Sequences are dense (T, C) and share build_windows' label rule."""
    from vitals_data import OBSERVATION_WINDOW_HOURS, VITALS, build_windows, generate_synthetic_cohort
    from vitals_seq import build_sequences

    cohort = generate_synthetic_cohort(n_patients=120, seed=0)
    data = build_sequences(cohort)

    assert data.X.ndim == 3
    assert data.X.shape[1] == OBSERVATION_WINDOW_HOURS
    assert data.X.shape[2] == len(VITALS)
    assert not np.isnan(data.X).any()  # fully imputed
    assert set(np.unique(data.y)).issubset({0, 1})
    assert data.y.sum() > 0  # some positive (pre-arrest) windows exist

    # Same windows/labels as the tabular builder → identical window count & positives.
    windowed = build_windows(cohort)
    assert data.X.shape[0] == len(windowed.labels)
    assert int(data.y.sum()) == int(windowed.labels.sum())


def test_patient_level_sequence_split_no_leakage():
    """No patient appears in both train and test sequence sets."""
    from vitals_data import generate_synthetic_cohort
    from vitals_seq import build_sequences, patient_level_sequence_split

    data = build_sequences(generate_synthetic_cohort(n_patients=120, seed=1))
    split = patient_level_sequence_split(data, seed=1)
    test_patients = set(split["groups_test"].tolist())
    # Reconstruct train patient ids from the mask complement.
    all_patients = set(np.unique(data.groups).tolist())
    train_patients = all_patients - test_patients
    assert train_patients and test_patients
    assert train_patients.isdisjoint(test_patients)


def test_sequence_model_learns_signal():
    """The GRU should beat the base rate on the synthetic deterioration signal."""
    pytest.importorskip("torch")
    from vitals_data import generate_synthetic_cohort
    from vitals_seq import build_sequences, patient_level_sequence_split, train_sequence_model

    data = build_sequences(generate_synthetic_cohort(n_patients=200, seed=7))
    split = patient_level_sequence_split(data, seed=7)
    _model, metrics = train_sequence_model(split, epochs=8, use_gpu=False, seed=7)

    assert 0.0 <= metrics.auprc <= 1.0
    assert metrics.roc_auc > 0.7
    assert metrics.auprc > metrics.prevalence  # better than predicting the base rate
