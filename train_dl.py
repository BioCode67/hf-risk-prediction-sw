"""Train/verify the LSTM/GRU vital-sign early-warning baseline (deep-learning track).

Runs on **dummy** variable-length sequences by default so the whole PyTorch
pipeline (preprocess → pad/mask Dataset → RNN → BCE) is verified end-to-end with
no data. Point ``--data-dir`` at a folder of per-patient CSV/Parquet/PSV files
(columns = vital features + a ``label`` column) to train on real data — only the
path changes.

    python train_dl.py                                  # dummy smoke test
    python train_dl.py --rnn gru --epochs 15 --device cuda
    python train_dl.py --data-dir /path/to/patients --rnn lstm

NOTE: this is the GPU-server benchmark arm (vs XGBoost). The 안심존 competition
entry stays on the lightweight XGBoost pipeline (see docs/competition-strategy.md).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def make_dummy_sequences(
    n_patients: int = 400,
    n_features: int = 7,
    positive_fraction: float = 0.5,
    min_len: int = 12,
    max_len: int = 48,
    missing_rate: float = 0.1,
    seed: int = 42,
) -> tuple[list[np.ndarray], list[int]]:
    """Synthetic ragged sequences; positives drift upward near the end (learnable)."""
    rng = np.random.default_rng(seed)
    sequences: list[np.ndarray] = []
    labels: list[int] = []
    for _ in range(n_patients):
        length = int(rng.integers(min_len, max_len + 1))
        is_pos = rng.random() < positive_fraction
        base = rng.normal(0.0, 1.0, size=n_features)
        seq = base[None, :] + rng.normal(0.0, 0.3, size=(length, n_features))
        if is_pos:
            ramp = np.linspace(0.0, 1.0, length)[:, None]
            seq = seq + ramp * rng.uniform(1.5, 3.0, size=n_features)[None, :]
        mask = rng.random(seq.shape) < missing_rate
        seq[mask] = np.nan
        sequences.append(seq)
        labels.append(int(is_pos))
    return sequences, labels


def evaluate(model, loader, device) -> dict[str, float]:
    """Return val loss / AUROC / AUPRC over a loader."""
    import torch
    from sklearn.metrics import average_precision_score, roc_auc_score

    model.eval()
    loss_fn = torch.nn.BCEWithLogitsLoss()
    losses, ys, ps = [], [], []
    with torch.no_grad():
        for batch in loader:
            x, lengths, y = batch["x"].to(device), batch["lengths"], batch["y"].to(device)
            logits = model(x, lengths)
            losses.append(float(loss_fn(logits, y)))
            ys.append(y.cpu().numpy())
            ps.append(torch.sigmoid(logits).cpu().numpy())
    y_true, y_prob = np.concatenate(ys), np.concatenate(ps)
    out = {"loss": float(np.mean(losses))}
    if len(np.unique(y_true)) > 1:
        out["auroc"] = float(roc_auc_score(y_true, y_prob))
        out["auprc"] = float(average_precision_score(y_true, y_prob))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="LSTM/GRU vital-sign early-warning baseline")
    parser.add_argument("--data-dir", default=None, help="dir of per-patient CSV/Parquet/PSV (default: dummy)")
    parser.add_argument("--rnn", choices=["lstm", "gru"], default="lstm")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    import torch

    from dataset import build_datasets, make_dataloader
    from model import build_model
    from utils import FEATURES, load_patient_sequences

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device

    if args.data_dir:
        sequences, labels = load_patient_sequences(args.data_dir)
        n_features = sequences[0].shape[1]
        print(f"Loaded {len(sequences)} patients from {args.data_dir} ({n_features} features)")
    else:
        sequences, labels = make_dummy_sequences(n_features=len(FEATURES))
        n_features = len(FEATURES)
        print(f"Dummy data: {len(sequences)} patients, {n_features} features, "
              f"prevalence={np.mean(labels):.2f}")

    train_ds, val_ds = build_datasets(sequences, labels, seed=42)
    train_loader = make_dataloader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = make_dataloader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = build_model(n_features, rnn_type=args.rnn, hidden_size=args.hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    print(f"Model: {args.rnn.upper()} | device: {device} | params: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        for batch in train_loader:
            x, lengths, y = batch["x"].to(device), batch["lengths"], batch["y"].to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x, lengths), y)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        val = evaluate(model, val_loader, device)
        msg = f"epoch {epoch:2d} | train_loss {np.mean(epoch_losses):.4f} | val_loss {val['loss']:.4f}"
        if "auroc" in val:
            msg += f" | val_AUROC {val['auroc']:.3f} | val_AUPRC {val['auprc']:.3f}"
        print(msg)

    print("\nPipeline OK — preprocess → pad/mask → RNN → BCE ran end-to-end.")


if __name__ == "__main__":
    main()
