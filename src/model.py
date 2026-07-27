"""Baseline recurrent classifier (LSTM/GRU) for vital-sign early warning.

A deliberately simple, masking-aware sequence classifier used as the **deep-
learning benchmark** against the XGBoost pipeline (per the mentor's request to
compare non-tree models). Emits a single logit per patient for
``BCEWithLogitsLoss``. GPU-ready — just ``.to(device)``.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class RNNClassifier(nn.Module):
    """LSTM/GRU → last hidden state → linear → single logit.

    Args:
        input_size: number of vital features (F).
        hidden_size: RNN hidden units.
        num_layers: stacked RNN layers.
        rnn_type: ``'lstm'`` or ``'gru'``.
        bidirectional: run the RNN in both directions.
        dropout: dropout between RNN layers (and before the head).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        rnn_type: str = "lstm",
        bidirectional: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU}[rnn_type.lower()]
        self.rnn = rnn_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        directions = 2 if bidirectional else 1
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * directions, 1),
        )
        self.bidirectional = bidirectional
        self.num_layers = num_layers

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """x: ``[B, T, F]``, lengths: ``[B]`` → logits ``[B]`` (padding-aware)."""
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        output = self.rnn(packed)[1]
        hidden = output[0] if isinstance(output, tuple) else output  # LSTM returns (h_n, c_n)
        # hidden: [num_layers * directions, B, H] -> take the last layer.
        if self.bidirectional:
            last = torch.cat([hidden[-2], hidden[-1]], dim=1)  # [B, 2H]
        else:
            last = hidden[-1]  # [B, H]
        return self.head(last).squeeze(-1)  # [B]


def build_model(input_size: int, rnn_type: str = "lstm", hidden_size: int = 64, **kwargs) -> RNNClassifier:
    """Convenience constructor mirroring the CLI options."""
    return RNNClassifier(input_size=input_size, hidden_size=hidden_size, rnn_type=rnn_type, **kwargs)
