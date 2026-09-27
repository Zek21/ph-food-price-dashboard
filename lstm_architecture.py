"""Dependency-minimal LSTM architecture shared by release training and ONNX export."""

from __future__ import annotations

import torch
import torch.nn as nn

SEQ_LEN = 12
N_FEATURES = 6
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.2
BATCH_SIZE = 64
DEFAULT_LR = 0.001
FEATURE_NAMES = ["price_norm", "month_sin", "month_cos", "year_norm", "region_enc", "pt_enc"]


class PriceLSTM(nn.Module):
    """Architecture-compatible replacement for the legacy lstm_model.PriceLSTM."""

    def __init__(
        self,
        input_size: int = N_FEATURES,
        hidden_size: int = HIDDEN_SIZE,
        num_layers: int = NUM_LAYERS,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        return self.head(output[:, -1, :]).squeeze(-1)
