"""Transformer based model for candle forecasting and trading decisions."""
from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - simple wrapper
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class CandleTransformer(nn.Module):
    """A compact yet expressive Transformer for timeseries prediction."""

    def __init__(
        self,
        input_dim: int = 6,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 8),
        )

    def forward(self, src: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning predictions and raw logits.

        Returns
        -------
        tuple
            (ohlcv_pred, direction_logits, trade_logits)
        """

        x = self.input_proj(src)
        x = self.pos_encoder(x)
        encoded = self.transformer_encoder(x)
        pooled = encoded[:, -1]
        output = self.head(pooled)
        ohlcv = output[:, :5]
        direction = output[:, 5:6]
        trade = output[:, 6:7]
        volatility = output[:, 7:]
        combined = torch.cat([ohlcv, volatility], dim=-1)
        return combined, direction, trade
