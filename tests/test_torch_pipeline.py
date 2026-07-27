"""Tests for the PyTorch LSTM/GRU track. Skipped when torch is not installed."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

import torch  # noqa: E402


def _dummy(n=20, f=7, seed=0):
    from train_dl import make_dummy_sequences

    return make_dummy_sequences(n_patients=n, n_features=f, seed=seed)


def test_collate_pads_and_masks():
    from dataset import VitalSignsDataset, collate_pad
    from utils import fit_normalizer

    seqs = [np.random.rand(t, 7) for t in (5, 9, 3)]
    labels = [0, 1, 1]
    ds = VitalSignsDataset(seqs, labels, fit_normalizer(seqs))
    batch = collate_pad([ds[i] for i in range(3)])

    assert batch["x"].shape == (3, 9, 7)  # padded to max length 9
    assert batch["mask"].shape == (3, 9)
    assert batch["lengths"].tolist() == [5, 9, 3]
    # mask marks exactly the real timesteps
    assert batch["mask"].sum(dim=1).tolist() == [5, 9, 3]
    assert batch["y"].shape == (3,)


def test_model_forward_shape():
    from model import build_model

    model = build_model(input_size=7, rnn_type="gru", hidden_size=16)
    x = torch.randn(4, 10, 7)
    lengths = torch.tensor([10, 7, 4, 2])
    logits = model(x, lengths)
    assert logits.shape == (4,)  # one logit per patient


def test_training_step_reduces_loss():
    """A few optimizer steps on learnable dummy data should reduce BCE loss."""
    from dataset import build_datasets, make_dataloader
    from model import build_model

    seqs, labels = _dummy(n=60, f=7)
    train_ds, _ = build_datasets(seqs, labels, seed=1)
    loader = make_dataloader(train_ds, batch_size=16, shuffle=True)

    model = build_model(7, rnn_type="lstm", hidden_size=16)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    first, last = None, None
    for _ in range(6):
        for batch in loader:
            opt.zero_grad()
            loss = loss_fn(model(batch["x"], batch["lengths"]), batch["y"])
            loss.backward()
            opt.step()
            last = loss.item()
            if first is None:
                first = last
    assert last < first  # learning happened
