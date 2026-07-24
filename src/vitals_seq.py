"""Neural **sequence** model for cardiac-arrest early warning (GPU-track prototype).

The core pipeline (``vitals_train``) turns each observation window into
hand-crafted statistics (mean/std/slope/…) and feeds them to XGBoost. This module
is an alternative that keeps the *raw temporal shape*: each window is the
``(T hours × C vitals)`` sequence itself, fed to a small **GRU** that learns the
deterioration dynamics directly. It is the natural way to exploit the A6000 and a
strong 본선 differentiator beyond the tabular baseline.

It reuses the exact windowing/labelling of :func:`vitals_data.build_windows`
(same 8-hour observation window, 1-hour horizon, patient-level grouping), so the
sequence model is an apples-to-apples comparison with the XGBoost baseline on
identical windows and labels.

**Isolation.** ``torch`` is an *optional* dependency (see ``requirements-seq.txt``)
for the GPU dev server — it is deliberately **not** in the offline 안심존 package
set. This module imports fine without torch; only the training/model calls require
it, and the test skips when it is absent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from vitals_data import (
    GAP_HOURS,
    OBSERVATION_WINDOW_HOURS,
    PREDICTION_HORIZON_HOURS,
    STEP_HOURS,
    VITALS,
    VitalSignsCohort,
)

try:  # torch is optional (GPU-track only); the module stays importable without it
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    _HAS_TORCH = True
except ModuleNotFoundError:  # pragma: no cover - exercised only in torch-free envs
    _HAS_TORCH = False


def _require_torch() -> None:
    if not _HAS_TORCH:
        raise ModuleNotFoundError(
            "The sequence model needs PyTorch. Install the optional GPU-track "
            "dependency:  pip install -r requirements-seq.txt"
        )


# --- Sequence construction -------------------------------------------------


@dataclass
class SequenceDataset:
    """Raw per-window vital sequences, labels and patient grouping.

    Attributes:
        X: ``float32`` array ``(n_windows, T, C)`` — T = observation hours,
            C = number of vitals (channels).
        y: ``int`` label per window (1 = arrest within the horizon).
        groups: patient id per window (for patient-level splitting).
        time_to_arrest: hours from window end to arrest (NaN for control windows).
        channels: the vital name of each channel, in order.
    """

    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    time_to_arrest: np.ndarray
    channels: list[str]


def build_sequences(
    cohort: VitalSignsCohort,
    observation_window_hours: int = OBSERVATION_WINDOW_HOURS,
    prediction_horizon_hours: int = PREDICTION_HORIZON_HOURS,
    gap_hours: int = GAP_HOURS,
    step_hours: int = STEP_HOURS,
    min_valid_fraction: float = 0.5,
) -> SequenceDataset:
    """Slide a window over each patient and emit the raw ``(T, C)`` vital sequence.

    Mirrors :func:`vitals_data.build_windows` exactly (same window, horizon, gap,
    step and label rule) but keeps the per-hour values instead of aggregating them.
    Missing hours inside a window are forward/back-filled, then any remaining gaps
    filled with that channel's window mean (0.0 if the whole channel is absent), so
    every emitted sequence is dense ``(T, C)``. Normalization is intentionally left
    to :func:`train_sequence_model`, which standardizes with train-only statistics.
    """
    channels = list(VITALS)
    n_channels = len(channels)
    min_rows = max(2, int(np.ceil(min_valid_fraction * observation_window_hours)))
    events = dict(zip(cohort.events["patient_id"], cohort.events["arrest_hour"]))

    seqs: list[np.ndarray] = []
    labels: list[int] = []
    groups: list[int] = []
    ttas: list[float] = []

    for patient_id, group in cohort.vitals.groupby("patient_id"):
        group = group.sort_values("hour")
        arrest_hour = events.get(patient_id, float("nan"))
        has_arrest = arrest_hour is not None and np.isfinite(arrest_hour)
        max_hour = int(group["hour"].max())
        by_hour = group.set_index("hour")

        for end in range(observation_window_hours - 1, max_hour + 1, step_hours):
            if has_arrest and end >= arrest_hour:
                continue  # cannot observe at/after the arrest
            low = end - observation_window_hours + 1
            observation = group[(group["hour"] >= low) & (group["hour"] <= end)]
            if len(observation) < min_rows:
                continue

            # Dense (T, C): reindex to every hour in the window, fill gaps.
            hours = list(range(low, end + 1))
            frame = by_hour.reindex(hours)[channels]
            frame = frame.ffill().bfill()
            frame = frame.fillna(frame.mean()).fillna(0.0)
            seq = frame.to_numpy(dtype=np.float32)  # (T, C)

            if has_arrest and (end + gap_hours) < arrest_hour <= (end + gap_hours + prediction_horizon_hours):
                label = 1
            else:
                label = 0

            seqs.append(seq)
            labels.append(label)
            groups.append(patient_id)
            ttas.append(float(arrest_hour - end) if has_arrest else float("nan"))

    X = np.stack(seqs).astype(np.float32) if seqs else np.empty((0, observation_window_hours, n_channels), np.float32)
    return SequenceDataset(
        X=X,
        y=np.asarray(labels, dtype=np.int64),
        groups=np.asarray(groups),
        time_to_arrest=np.asarray(ttas, dtype=np.float64),
        channels=channels,
    )


def patient_level_sequence_split(
    data: SequenceDataset, test_size: float = 0.2, seed: int = 42
) -> dict[str, Any]:
    """Split sequences by patient so no patient appears in both train and test."""
    rng = np.random.default_rng(seed)
    patients = np.unique(data.groups)
    rng.shuffle(patients)
    n_test = max(1, int(round(len(patients) * test_size)))
    test_patients = set(patients[:n_test].tolist())

    test_mask = np.array([g in test_patients for g in data.groups])
    train_mask = ~test_mask
    return {
        "X_train": data.X[train_mask],
        "y_train": data.y[train_mask],
        "X_test": data.X[test_mask],
        "y_test": data.y[test_mask],
        "groups_test": data.groups[test_mask],
        "tta_test": data.time_to_arrest[test_mask],
        "channels": data.channels,
    }


# --- Model -----------------------------------------------------------------

if _HAS_TORCH:

    class GRUClassifier(nn.Module):
        """Small GRU over the ``(T, C)`` vital sequence → arrest-risk logit."""

        def __init__(self, n_channels: int, hidden_size: int = 48, num_layers: int = 1, dropout: float = 0.0):
            super().__init__()
            self.gru = nn.GRU(
                input_size=n_channels,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.head = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, 1))

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":  # noqa: F821
            _out, h_n = self.gru(x)
            return self.head(h_n[-1]).squeeze(-1)  # (batch,) logit


def _resolve_device(use_gpu: bool) -> str:
    """Pick 'cuda' when requested and available, else 'cpu'."""
    if use_gpu and torch.cuda.is_available():
        return "cuda"
    if use_gpu:
        print("[gpu] CUDA not available; training the sequence model on CPU.")
    return "cpu"


@dataclass
class SequenceMetrics:
    model_name: str
    auprc: float
    roc_auc: float
    prevalence: float
    n_test: int


def train_sequence_model(
    split: dict[str, Any],
    hidden_size: int = 48,
    num_layers: int = 1,
    epochs: int = 15,
    batch_size: int = 256,
    lr: float = 1e-3,
    use_gpu: bool = False,
    seed: int = 42,
) -> tuple[Any, SequenceMetrics]:
    """Train the GRU with a cost-sensitive loss; evaluate AUPRC/ROC on the test set.

    The positive class is up-weighted by ``negatives/positives`` (the same
    cost-sensitive framing as the XGBoost ``scale_pos_weight``), because arrests
    are rare. Standardization uses **train-only** channel statistics to avoid
    leakage. Returns the trained model and its metrics.
    """
    _require_torch()
    torch.manual_seed(seed)
    device = _resolve_device(use_gpu)

    X_train = split["X_train"].astype(np.float32)
    X_test = split["X_test"].astype(np.float32)
    y_train, y_test = split["y_train"], split["y_test"]

    # Standardize per channel with train-only stats (broadcast over time).
    mean = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0)
    std = X_train.reshape(-1, X_train.shape[-1]).std(axis=0)
    std[std == 0] = 1.0
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    positives = int((y_train == 1).sum())
    negatives = int((y_train == 0).sum())
    pos_weight = torch.tensor([negatives / positives if positives else 1.0], device=device)

    model = GRUClassifier(n_channels=X_train.shape[-1], hidden_size=hidden_size, num_layers=num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train.astype(np.float32))),
        batch_size=batch_size,
        shuffle=True,
    )

    model.train()
    for _epoch in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X_test).to(device)).cpu().numpy()
    scores = 1.0 / (1.0 + np.exp(-logits))  # sigmoid → probability

    metrics = SequenceMetrics(
        model_name="GRU",
        auprc=float(average_precision_score(y_test, scores)) if len(np.unique(y_test)) > 1 else float("nan"),
        roc_auc=float(roc_auc_score(y_test, scores)) if len(np.unique(y_test)) > 1 else float("nan"),
        prevalence=float(np.mean(y_test)),
        n_test=int(len(y_test)),
    )
    return model, metrics


def run_demo(n_patients: int = 600, epochs: int = 15, use_gpu: bool = False, seed: int = 42) -> SequenceMetrics:
    """End-to-end synthetic demo: sequences → GRU → AUPRC/ROC vs the XGBoost baseline."""
    from vitals_data import build_windows, generate_synthetic_cohort, patient_level_split
    from vitals_train import train_xgboost

    cohort = generate_synthetic_cohort(n_patients=n_patients, seed=seed)

    # Sequence (GRU) track.
    data = build_sequences(cohort)
    split = patient_level_sequence_split(data, seed=seed)
    print(
        f"Sequences: {data.X.shape[0]:,} windows x {data.X.shape[1]}h x {data.X.shape[2]} vitals "
        f"| train {len(split['y_train']):,} / test {len(split['y_test']):,} "
        f"| test prevalence {np.mean(split['y_test']):.3f}"
    )
    _model, seq_metrics = train_sequence_model(split, epochs=epochs, use_gpu=use_gpu, seed=seed)

    # Tabular (XGBoost) baseline on the same cohort, for a like-for-like number.
    tab_split = patient_level_split(build_windows(cohort), seed=seed)
    _xgb, xgb_metrics = train_xgboost(tab_split)

    print("\n=== Sequence model vs tabular baseline (same synthetic cohort) ===")
    print(f"  GRU      AUPRC={seq_metrics.auprc:.3f}  ROC={seq_metrics.roc_auc:.3f}")
    print(f"  XGBoost  AUPRC={xgb_metrics.auprc:.3f}  ROC={xgb_metrics.roc_auc:.3f}")
    return seq_metrics


def main() -> None:
    """CLI entry point. Flags: ``--gpu`` (CUDA), ``--epochs N``."""
    import sys

    argv = sys.argv
    epochs = 15
    if "--epochs" in argv and argv.index("--epochs") + 1 < len(argv):
        epochs = int(argv[argv.index("--epochs") + 1])
    run_demo(epochs=epochs, use_gpu="--gpu" in argv)


if __name__ == "__main__":
    main()
