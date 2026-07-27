"""PyTorch Dataset / DataLoader for variable-length vital-sign sequences.

Handles the two things RNNs need for ragged clinical time series: **padding**
(to a common length within a batch) and **masking** (so padded steps don't count).
Sequences are preprocessed (forward-fill → impute → z-score) via ``utils`` before
batching, so a batch yields padded features, a boolean valid-step mask, true
lengths, and the label.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from utils import FEATURES, Normalizer, fit_normalizer, preprocess_sequence


class VitalSignsDataset(Dataset):
    """Per-patient vital-sign sequences with binary labels.

    Args:
        sequences: list of raw ``[T_i, F]`` arrays (may contain NaN).
        labels: list of 0/1 patient labels.
        normalizer: fitted :class:`utils.Normalizer` (fit on the training split).
        impute: ``'mean'`` or ``'zero'`` for NaNs remaining after forward-fill.
    """

    def __init__(
        self,
        sequences: list[np.ndarray],
        labels: list[int],
        normalizer: Normalizer,
        impute: str = "mean",
    ) -> None:
        self.sequences = [preprocess_sequence(s, normalizer, impute) for s in sequences]
        self.labels = [float(y) for y in labels]

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        x = torch.from_numpy(self.sequences[idx])  # [T, F] float32
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y, x.shape[0]


def collate_pad(batch: list[tuple[torch.Tensor, torch.Tensor, int]]) -> dict[str, torch.Tensor]:
    """Pad a batch to its max length and build a valid-step mask.

    Returns dict with:
        x: ``[B, T_max, F]`` padded features
        mask: ``[B, T_max]`` bool (True = real timestep)
        lengths: ``[B]`` true lengths
        y: ``[B]`` labels
    """
    xs, ys, lengths = zip(*batch)
    padded = pad_sequence(list(xs), batch_first=True)  # [B, T_max, F]
    lengths_t = torch.tensor(lengths, dtype=torch.long)
    max_len = padded.shape[1]
    mask = torch.arange(max_len)[None, :] < lengths_t[:, None]
    return {"x": padded, "mask": mask, "lengths": lengths_t, "y": torch.stack(ys)}


def make_dataloader(
    dataset: VitalSignsDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Wrap a :class:`VitalSignsDataset` in a padding-aware DataLoader."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_pad,
    )


def build_datasets(
    sequences: list[np.ndarray],
    labels: list[int],
    val_fraction: float = 0.2,
    seed: int = 42,
    impute: str = "mean",
    n_features: int | None = None,
) -> tuple[VitalSignsDataset, VitalSignsDataset]:
    """Patient-level train/val split with the normalizer fit on **train only**."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(sequences))
    n_val = max(1, int(len(sequences) * val_fraction))
    val_idx, train_idx = set(order[:n_val].tolist()), order[n_val:].tolist()

    train_seq = [sequences[i] for i in train_idx]
    train_lab = [labels[i] for i in train_idx]
    val_seq = [sequences[i] for i in range(len(sequences)) if i in val_idx]
    val_lab = [labels[i] for i in range(len(sequences)) if i in val_idx]

    normalizer = fit_normalizer(train_seq)
    return (
        VitalSignsDataset(train_seq, train_lab, normalizer, impute),
        VitalSignsDataset(val_seq, val_lab, normalizer, impute),
    )


N_FEATURES: int = len(FEATURES)
