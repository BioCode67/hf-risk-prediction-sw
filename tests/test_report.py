"""Tests for the visual report module (renders headless PNGs to a temp dir)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("matplotlib")


def test_plot_pr_curves_creates_file(tmp_path):
    from vitals_report import plot_pr_curves

    y = np.array([0, 0, 1, 1, 0, 1])
    scores = {"XGBoost": np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7]), "NEWS": np.array([1.0, 2, 3, 4, 1, 2])}
    out = plot_pr_curves(y, scores, tmp_path / "pr.png")
    assert out.exists() and out.stat().st_size > 0


def test_plot_lead_time_distribution_creates_file(tmp_path):
    from vitals_report import plot_lead_time_distribution

    out = plot_lead_time_distribution([1.0, 2.0, 3.0, 2.0], tmp_path / "lt.png")
    assert out.exists() and out.stat().st_size > 0


def test_plot_deterioration_trajectory_creates_file(tmp_path):
    from vitals_data import VITALS
    from vitals_report import plot_deterioration_trajectory

    frame = pd.DataFrame({"hour": [0, 1, 2, 3], **{v: [1.0, 2.0, 3.0, 4.0] for v in VITALS}})
    out = plot_deterioration_trajectory(frame, arrest_hour=3.0, output_path=tmp_path / "traj.png")
    assert out.exists() and out.stat().st_size > 0
