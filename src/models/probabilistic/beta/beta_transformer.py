from __future__ import annotations

from typing import Optional, Tuple

import math
import torch
from torch import nn
import torch.nn.functional as F


class PositionalEncodingBatchFirst(nn.Module):
    """
    Positional encoding for batch-first tensors.
    Input/Output: x [B, L, D]
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # pe: [1, max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]

        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        L = x.size(1)
        x = x + self.pe[:, :L, :]
        return self.dropout(x)


class BetaTimeSeriesTransformer(nn.Module):
    """
    Time series Transformer that outputs Beta distribution parameters.

    - Input:  x [B, L, input_dim]
    - Output:
        if num_components == 1:
            alpha, beta: [B, 1] each
        else:
            alpha, beta: [B, K] each

    This version uses (mu, kappa) parameterization:
        mu in (0,1), kappa > 0
        alpha = mu * kappa
        beta  = (1 - mu) * kappa
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        num_components: int = 1,
        eps: float = 1e-4,
        kappa_max: Optional[float] = None,  # e.g. 500.0 if you want to clamp
        max_len: int = 5000,
    ):
        super().__init__()
        if num_components < 1:
            raise ValueError(f"num_components must be >= 1, got {num_components}")

        self.input_dim = input_dim
        self.d_model = d_model
        self.num_components = num_components
        self.eps = float(eps)
        self.kappa_max = kappa_max

        # 1) projection to d_model
        self.input_proj = nn.Linear(input_dim, d_model)

        # 2) positional encoding (batch-first)
        self.pos_encoder = PositionalEncodingBatchFirst(d_model=d_model, dropout=dropout, max_len=max_len)

        # 3) Transformer Encoder (batch_first=True avoids transpose and warning)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 4) output head: 2 * K (mu_raw, kappa_raw)
        self.fc_out = nn.Linear(d_model, 2 * num_components)

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, L, input_dim]
        src_key_padding_mask (optional): [B, L] where True indicates padding positions.

        returns:
            alpha, beta:
              if num_components == 1: [B, 1]
              else: [B, K]
        """
        if x.dim() != 3:
            raise ValueError(f"Expected x to be 3D [B,L,D], got shape {tuple(x.shape)}")

        B, L, _ = x.shape

        # project + positional encoding
        h = self.input_proj(x)          # [B, L, d_model]
        h = self.pos_encoder(h)         # [B, L, d_model]

        # transformer encoder
        mem = self.transformer_encoder(h, src_key_padding_mask=src_key_padding_mask)  # [B, L, d_model]

        # last time step representation
        # If you use padding and variable length, you should pool with mask instead of taking -1.
        last_hidden = mem[:, -1, :]     # [B, d_model]

        raw = self.fc_out(last_hidden)  # [B, 2K]
        raw = raw.view(B, self.num_components, 2)  # [B, K, 2]

        mu_raw = raw[..., 0]            # [B, K]
        kappa_raw = raw[..., 1]         # [B, K]

        # mu in (0,1)
        mu = torch.sigmoid(mu_raw)
        mu = torch.clamp(mu, self.eps, 1.0 - self.eps)

        # kappa > 0
        kappa = F.softplus(kappa_raw) + self.eps
        if self.kappa_max is not None:
            kappa = torch.clamp(kappa, max=float(self.kappa_max))

        alpha = mu * kappa
        beta = (1.0 - mu) * kappa

        # return [B,1] if K==1 to match pipeline expectations nicely
        if self.num_components == 1:
            alpha = alpha.view(B, 1)
            beta = beta.view(B, 1)

        return alpha, beta