from __future__ import annotations

import torch
from torch import nn

from src.pipelines.forecast_base import ForecastBasePipeline
from src.pipelines.utils_data import DataArtifacts
from src.models.backbones.transformer import TimeSeriesTransformer


class TransformerWithZoneEmbedding(nn.Module):
    """
    Zone embedding wrapper:
      - input:  x [B, L, F], zone id z [B]
      - output: base([x, emb(z)]) -> [B, H]
    """
    def __init__(self, base: nn.Module, n_zones_with_unk: int, emb_dim: int):
        super().__init__()
        self.base = base
        self.emb = nn.Embedding(int(n_zones_with_unk), int(emb_dim))

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        z = z.long()
        ze = self.emb(z)                                # [B, E]
        ze = ze.unsqueeze(1).expand(-1, x.size(1), -1)   # [B, L, E]
        x_aug = torch.cat([x, ze], dim=-1)               # [B, L, F+E]
        return self.base(x_aug)


class TransformerDetPipeline(ForecastBasePipeline):
    """
    Deterministic Transformer pipeline.

    - Supports zone embedding (keep_zone)
    - No VAE (removed for simplicity and reproducibility)
    """

    def build_model(self, input_dim: int, output_dim: int, data_art: DataArtifacts) -> nn.Module:
        cfg = self.cfg

        keep_zone = bool(data_art.meta.get("keep_zone", False))
        zone_emb_dim = int(cfg.get("zone_emb_dim", 8))

        d_model = int(cfg.get("d_model", 64))
        nhead = int(cfg.get("nhead", 4))
        num_layers = int(cfg.get("num_layers", 2))
        dim_ff = int(cfg.get("dim_feedforward", 128))
        dropout = float(cfg.get("dropout", 0.1))

        # input dim (+ zone embedding if used)
        in_dim = int(input_dim + (zone_emb_dim if keep_zone else 0))

        base = TimeSeriesTransformer(
            input_dim=in_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_ff,
            dropout=dropout,
            output_dim=int(output_dim),
        )

        if not keep_zone:
            return base

        zone_mapping = data_art.zone_mapping
        if zone_mapping is None:
            raise ValueError("keep_zone=True but zone_mapping is None.")

        n_train_zones = int(len(zone_mapping))
        return TransformerWithZoneEmbedding(
            base=base,
            n_zones_with_unk=n_train_zones + 1,
            emb_dim=zone_emb_dim,
        )