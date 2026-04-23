from __future__ import annotations

from typing import Optional, Tuple, List

import numpy as np
import torch
from torch import nn

from src.pipelines.forecast_base import ForecastBasePipeline
from src.pipelines.utils_data import DataArtifacts
from src.models.probabilistic.quantile_transformer import QuantileTimeSeriesTransformer


def pinball_loss_torch(y: torch.Tensor, q: torch.Tensor, quantiles: List[float]) -> torch.Tensor:
    """
    y: [B] or [B,1]
    q: [B,K] or [B,1,K]
    """
    if y.dim() == 1:
        y = y.unsqueeze(-1)  # [B,1]

    if q.dim() == 3:
        # horizon=1: [B,1,K] -> [B,K]
        q = q[:, 0, :]

    taus = torch.tensor(quantiles, device=q.device, dtype=q.dtype).view(1, -1)  # [1,K]
    diff = y - q  # [B,K]
    loss = torch.maximum(taus * diff, (taus - 1.0) * diff)
    return loss.mean()


def _nearest_idx(values: List[float], target: float) -> int:
    return int(np.argmin([abs(v - target) for v in values]))


def _ensure_b1(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 1:
        return x.unsqueeze(-1)
    if x.dim() == 2 and x.size(-1) != 1:
        return x[:, :1]
    return x


def _enforce_monotone(q: torch.Tensor) -> torch.Tensor:
    """
    q: [B,K] or [B,1,K]
    """
    return torch.cummax(q, dim=-1).values


class TransformerQRPipeline(ForecastBasePipeline):
    """
    Quantile Transformer pipeline:
      - model outputs q (multi-quantiles)
      - train with pinball loss
      - point prediction uses q at 0.5
      - interval uses q at tail and 1-tail
      - supports keep_zone embedding concat (same philosophy as Beta pipeline)
    """

    def build_model(self, input_dim: int, output_dim: int, data_art: DataArtifacts) -> nn.Module:
        cfg = self.cfg

        if int(output_dim) != 1:
            raise ValueError(f"Quantile transformer currently supports horizon=1 only, got output_dim={output_dim}.")

        # IMPORTANT: trust dataloader meta for keep_zone（同 beta 的做法）
        keep_zone = bool(data_art.meta.get("keep_zone", False))
        zone_emb_dim = int(cfg.get("zone_emb_dim", 8))

        # Transformer hparams（与 beta 对齐）
        d_model = int(cfg.get("d_model", 64))
        nhead = int(cfg.get("nhead", 4))
        num_layers = int(cfg.get("num_layers", 2))
        dim_ff = int(cfg.get("dim_feedforward", 128))
        dropout = float(cfg.get("dropout", 0.1))

        quantiles = list(cfg["probabilistic"]["quantiles"])
        if len(quantiles) < 2:
            raise ValueError("Need at least 2 quantiles for QR.")

        in_dim = int(input_dim + (zone_emb_dim if keep_zone else 0))

        qr_core = QuantileTimeSeriesTransformer(
            input_dim=in_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_ff,
            dropout=dropout,
            quantiles=quantiles,
        )

   
        if not keep_zone:
            return qr_core

        zone_mapping = data_art.zone_mapping
        if zone_mapping is None:
            raise ValueError("keep_zone=True but data_art.zone_mapping is None.")
        n_train_zones = int(len(zone_mapping))

        return _QRTransformerWithZoneEmbedding(
            base=qr_core,
            n_zones_with_unk=n_train_zones + 1,
            emb_dim=zone_emb_dim,
        )

    def model_forward(self, model: nn.Module, batch):
        """
        Return raw q.
        Supports batches (x, y) or (x, y, z).
        """
        if len(batch) == 2:
            x, _y = batch
            z = None
        else:
            x, _y, z = batch

        x = x.to(self.device)
        if z is not None:
            z = z.to(self.device).long()

        try:
            return model(x, z) if z is not None else model(x)
        except TypeError:
            return model(x)

    def compute_loss(self, loss_fn: nn.Module, pred, y: torch.Tensor) -> torch.Tensor:
        """
        Pinball loss on y (same scale as q).
        pred: q
        """
        y = y.to(self.device)
        quantiles = list(self.cfg["probabilistic"]["quantiles"])
        return pinball_loss_torch(y, pred, quantiles)

    def postprocess_pred(self, pred) -> torch.Tensor:
        """
        Point prediction via median quantile (0.5). Output: [B,1].
        """
        q = _enforce_monotone(pred)
        qs = list(self.cfg["probabilistic"]["quantiles"])
        mid = _nearest_idx(qs, 0.5)

        if q.dim() == 3:
            mu = q[:, 0, mid]
        else:
            mu = q[:, mid]

        mu = _ensure_b1(mu)

        y_min = float(self.cfg.get("y_min", 0.0))
        y_max = float(self.cfg.get("y_max", 1.0))
        if (y_min, y_max) != (0.0, 1.0):
            mu = _inverse_scale_from_unit(mu, y_min=y_min, y_max=y_max)
        return mu
    
    def postprocess_quantiles(self, pred_raw) -> torch.Tensor:
        """
        Return quantile predictions for pinball eval.
        Shape: [B,K] or [B,1,K] (we keep it as model outputs, but enforce monotone).
        """
        q = _enforce_monotone(pred_raw)

        # If you used y_min/y_max scaling (unit interval training), inverse it here too
        y_min = float(self.cfg.get("y_min", 0.0))
        y_max = float(self.cfg.get("y_max", 1.0))
        if (y_min, y_max) != (0.0, 1.0):
            q = _inverse_scale_from_unit(q, y_min=y_min, y_max=y_max)

        return q
        
    @torch.no_grad()
    def postprocess_interval(self, pred, coverage: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Interval from predicted quantiles. Returns lower/upper: [B,1].
        """
        q = _enforce_monotone(pred)

        if coverage is None:
            coverage = float(self.cfg.get("interval_coverage", 0.9))
        tail = (1.0 - float(coverage)) / 2.0

        qs = list(self.cfg["probabilistic"]["quantiles"])
        lo = _nearest_idx(qs, tail)
        hi = _nearest_idx(qs, 1.0 - tail)

        if q.dim() == 3:
            lower = q[:, 0, lo]
            upper = q[:, 0, hi]
        else:
            lower = q[:, lo]
            upper = q[:, hi]

        lower = _ensure_b1(lower)
        upper = _ensure_b1(upper)

        y_min = float(self.cfg.get("y_min", 0.0))
        y_max = float(self.cfg.get("y_max", 1.0))
        if (y_min, y_max) != (0.0, 1.0):
            lower = _inverse_scale_from_unit(lower, y_min=y_min, y_max=y_max)
            upper = _inverse_scale_from_unit(upper, y_min=y_min, y_max=y_max)

        return lower, upper


class _QRTransformerWithZoneEmbedding(nn.Module):
    """
    Same idea as BetaTransformerWithZoneEmbedding:
      x: [B,L,F]
      z: [B]
    """
    def __init__(self, base: nn.Module, n_zones_with_unk: int, emb_dim: int):
        super().__init__()
        self.base = base
        self.emb = nn.Embedding(int(n_zones_with_unk), int(emb_dim))

    def forward(self, x: torch.Tensor, z: torch.Tensor):
        z = z.long()
        ze = self.emb(z)                                # [B,E]
        ze = ze.unsqueeze(1).expand(-1, x.size(1), -1)   # [B,L,E]
        x_aug = torch.cat([x, ze], dim=-1)               # [B,L,F+E]
        return self.base(x_aug)


def _inverse_scale_from_unit(u01: torch.Tensor, y_min: float, y_max: float) -> torch.Tensor:
    return u01 * (y_max - y_min) + y_min

