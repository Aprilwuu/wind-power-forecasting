# src/models/quantile_transformer.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn


class QuantileTimeSeriesTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        quantiles: List[float],
        keep_zone: bool = False,
        num_zones: Optional[int] = None,
        zone_emb_dim: int = 0,
    ):
        super().__init__()
        self.quantiles = torch.tensor(quantiles, dtype=torch.float32)
        self.K = len(quantiles)

        # ---- Copy from Beta Transformer backbone ----
        self.in_proj = nn.Linear(input_dim, d_model)

        if keep_zone:
            assert num_zones is not None and zone_emb_dim > 0
            self.zone_emb = nn.Embedding(num_zones, zone_emb_dim)
            self.zone_proj = nn.Linear(d_model + zone_emb_dim, d_model)
        else:
            self.zone_emb = None
            self.zone_proj = None

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # important
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # Simple pooling: take last token (or mean pooling)
        self.pool = "last"

        # ---- Quantile head (replace beta head) ----
        self.head = nn.Linear(d_model, self.K)

    def forward(self, x_seq: torch.Tensor, zone_id: Optional[torch.Tensor] = None):
        """
        x_seq: (B, L, input_dim)
        zone_id: (B,) optional
        """
        h = self.in_proj(x_seq)  # (B, L, d_model)

        if self.zone_emb is not None and zone_id is not None:
            z = self.zone_emb(zone_id)  # (B, zone_emb_dim)
            z = z.unsqueeze(1).expand(-1, h.size(1), -1)  # (B, L, zone_emb_dim)
            h = torch.cat([h, z], dim=-1)
            h = self.zone_proj(h)

        h = self.encoder(h)  # (B, L, d_model)

        if self.pool == "last":
            h_last = h[:, -1, :]  # (B, d_model)
        else:
            h_last = h.mean(dim=1)

        q = self.head(h_last)  # (B, K)
        return q