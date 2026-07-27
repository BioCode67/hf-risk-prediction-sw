"""Preprocessing utilities for the vital-sign deep-learning (LSTM/GRU) track.

Kept deliberately framework-light (numpy/pandas only) so preprocessing is
testable without torch and reusable across the XGBoost and RNN pipelines.
The canonical unit is a per-patient array of shape ``[T, F]`` (time × feature),
possibly containing NaNs, which we forward-fill, mean/zero-impute, then z-score.

Swap ``load_patient_sequences`` in for real MIMIC-IV CSV/Parquet by pointing it
at a directory — only the path changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

# Vital-sign features for the RNN track (MIMIC-IV / Challenge-2019 style).
FEATURES: Final[tuple[str, ...]] = ("HR", "SBP", "DBP", "MAP", "RespRate", "SpO2", "Temp")


@dataclass
class Normalizer:
    """Per-feature z-score normalizer, fit on training sequences only."""

    mean: np.ndarray
    std: np.ndarray

    def transform(self, array: np.ndarray) -> np.ndarray:
        """Z-score an ``[T, F]`` array with the fitted per-feature mean/std."""
        std = np.where(self.std < 1e-8, 1.0, self.std)
        return (array - self.mean) / std


def forward_fill(array: np.ndarray) -> np.ndarray:
    """Forward-fill NaNs down each feature column (last observation carried forward)."""
    return pd.DataFrame(array).ffill().to_numpy()


def impute_remaining(array: np.ndarray, fill_values: np.ndarray | None) -> np.ndarray:
    """Fill NaNs left after forward-fill with per-feature ``fill_values`` (else 0)."""
    filled = array.copy()
    if fill_values is None:
        return np.nan_to_num(filled, nan=0.0)
    idx = np.where(np.isnan(filled))
    filled[idx] = np.take(fill_values, idx[1])
    return filled


def fit_normalizer(sequences: list[np.ndarray]) -> Normalizer:
    """Fit per-feature mean/std over all timesteps of all (training) sequences."""
    stacked = np.concatenate([np.asarray(s, dtype=float) for s in sequences], axis=0)
    mean = np.nanmean(stacked, axis=0)
    std = np.nanstd(stacked, axis=0)
    return Normalizer(mean=np.nan_to_num(mean, nan=0.0), std=np.nan_to_num(std, nan=1.0))


def preprocess_sequence(
    array: np.ndarray,
    normalizer: Normalizer,
    impute: str = "mean",
) -> np.ndarray:
    """Forward-fill → impute remaining (mean/zero) → z-score. Returns float32 ``[T, F]``.

    ``impute='mean'`` uses the normalizer's per-feature mean; ``'zero'`` fills 0.
    """
    array = np.asarray(array, dtype=float)
    array = forward_fill(array)
    fill = normalizer.mean if impute == "mean" else None
    array = impute_remaining(array, fill)
    array = normalizer.transform(array)
    return array.astype(np.float32)


def load_patient_sequences(
    data_dir: str | Path,
    features: tuple[str, ...] = FEATURES,
    label_col: str = "label",
    pattern: str = "*",
) -> tuple[list[np.ndarray], list[int]]:
    """Load one time series per patient file from a directory of CSV/Parquet.

    Each file is one patient: rows = timesteps (hourly), columns include the
    ``features`` and a per-file ``label_col`` (0/1, taken from the max over rows —
    i.e. the patient's outcome). This is the single point to swap for real
    MIMIC-IV: point ``data_dir`` at the exported per-patient files.
    """
    directory = Path(data_dir)
    sequences: list[np.ndarray] = []
    labels: list[int] = []
    paths = sorted(p for p in directory.glob(pattern) if p.suffix in (".csv", ".parquet", ".psv"))
    for path in paths:
        if path.suffix == ".parquet":
            frame = pd.read_parquet(path)
        else:
            frame = pd.read_csv(path, sep="|" if path.suffix == ".psv" else ",")
        cols = [c for c in features if c in frame.columns]
        sequences.append(frame[cols].to_numpy(dtype=float))
        labels.append(int(frame[label_col].max()) if label_col in frame.columns else 0)
    return sequences, labels
