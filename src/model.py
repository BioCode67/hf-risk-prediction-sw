"""Masking-aware sequence classifiers (LSTM/GRU/Transformer) for vital signs.

Deliberately simple architectures used as the **deep-learning benchmark**
against the XGBoost pipeline. All emit a single logit per patient for
``BCEWithLogitsLoss`` and share the ``forward(x, lengths)`` signature, so the
training loop is architecture-agnostic. GPU-ready — just ``.to(device)``.
"""

from __future__ import annotations

import math

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


class TransformerClassifier(nn.Module):
    """Transformer encoder → masked mean-pool → linear → single logit.

    Padding positions are excluded both from attention (``src_key_padding_mask``)
    and from the pooled representation, so variable-length stays exact. Sized
    for hundreds-of-patients cohorts: small d_model, pre-norm, GELU.

    Args:
        input_size: number of vital features (F).
        d_model: encoder width.
        nhead: attention heads (must divide d_model).
        num_layers: stacked encoder layers.
        dim_feedforward: FFN width inside each layer.
        dropout: dropout in the encoder and before the head.
        max_len: longest supported sequence (positional-encoding table size).
    """

    def __init__(
        self,
        input_size: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        max_len: int = 512,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("positional_encoding", pe, persistent=False)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, 1))

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """x: ``[B, T, F]``, lengths: ``[B]`` → logits ``[B]`` (padding-aware)."""
        batch, steps, _ = x.shape
        h = self.input_proj(x) + self.positional_encoding[:steps].unsqueeze(0)
        pad_mask = torch.arange(steps, device=x.device).unsqueeze(0) >= lengths.to(x.device).unsqueeze(1)
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        valid = (~pad_mask).unsqueeze(-1).to(h.dtype)  # [B, T, 1]
        pooled = (h * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        return self.head(pooled).squeeze(-1)  # [B]


def build_model(input_size: int, rnn_type: str = "lstm", hidden_size: int = 64, **kwargs) -> nn.Module:
    """Convenience constructor mirroring the CLI options.

    ``rnn_type`` picks the architecture: ``'lstm'`` / ``'gru'`` build an
    :class:`RNNClassifier`; ``'transformer'`` builds a
    :class:`TransformerClassifier` with ``hidden_size`` as ``d_model``.
    """
    if rnn_type.lower() == "transformer":
        return TransformerClassifier(input_size=input_size, d_model=hidden_size, **kwargs)
    return RNNClassifier(input_size=input_size, hidden_size=hidden_size, rnn_type=rnn_type, **kwargs)
