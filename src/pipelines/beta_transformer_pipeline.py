# src/pipelines/beta_transformer_pipeline.py
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn

from src.pipelines.forecast_base import ForecastBasePipeline
from src.pipelines.utils_data import DataArtifacts
from src.models.probabilistic.beta.beta_transformer import BetaTimeSeriesTransformer
from src.models.representation.vae import SequenceVAE
from src.models.representation.vae_transformer_wrappers import VAETransformerBetaForecast


def _scale_to_unit(y: torch.Tensor, y_min: float, y_max: float, eps: float = 1e-4) -> torch.Tensor:
    """Scale y from [y_min, y_max] to (0, 1) and clamp for numerical stability."""
    denom = (y_max - y_min)
    if denom <= 0:
        raise ValueError(f"Invalid y_min/y_max: y_min={y_min}, y_max={y_max}")
    y01 = (y - y_min) / denom
    return torch.clamp(y01, eps, 1.0 - eps)


def _inverse_scale_from_unit(u01: torch.Tensor, y_min: float, y_max: float) -> torch.Tensor:
    """Inverse scale from (0, 1) back to [y_min, y_max]."""
    return u01 * (y_max - y_min) + y_min


def beta_nll(y01: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """
    Negative log-likelihood of Beta(alpha, beta) on targets y01 in (0,1).
    Supports alpha/beta of shape [B,1] or [B,K].
    """
    if y01.dim() == 2 and y01.size(-1) == 1:
        y01 = y01.squeeze(-1)

    if alpha.dim() == 2 and alpha.size(-1) == 1:
        alpha = alpha.squeeze(-1)
        beta = beta.squeeze(-1)

    logB = torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta)
    logp = (alpha - 1.0) * torch.log(y01) + (beta - 1.0) * torch.log(1.0 - y01) - logB
    return (-logp).mean()


class BetaTransformerWithZoneEmbedding(nn.Module):
    """
    Concatenate a trainable zone embedding to each time step feature.
    Base model expects x only; this wrapper makes it accept (x, z).

      x: [B, L, F]
      z: [B]
    """
    def __init__(self, base: nn.Module, n_zones_with_unk: int, emb_dim: int):
        super().__init__()
        self.base = base
        self.emb = nn.Embedding(int(n_zones_with_unk), int(emb_dim))

    def forward(self, x: torch.Tensor, z: torch.Tensor):
        z = z.long()
        ze = self.emb(z)                                  # [B, E]
        ze = ze.unsqueeze(1).expand(-1, x.size(1), -1)   # [B, L, E]
        x_aug = torch.cat([x, ze], dim=-1)               # [B, L, F+E]
        return self.base(x_aug)


class TransformerBetaPipeline(ForecastBasePipeline):
    """
    Beta Transformer pipeline:
      - model outputs (alpha, beta)
      - train with Beta NLL on y scaled to (0,1)
      - point prediction uses Beta mean mapped back to original scale
      - interval uses SciPy beta.ppf
      - supports keep_zone embedding concat
      - supports VAE in two modes:
          (1) vae_mode="e2e": end-to-end VAE encoder runs inside the model
          (2) vae_mode="cache": dataloader provides precomputed latent sequences
    """

    def build_model(self, input_dim: int, output_dim: int, data_art: DataArtifacts) -> nn.Module:
        cfg = self.cfg

        if int(output_dim) != 1:
            raise ValueError(f"Beta transformer currently supports horizon=1 only, got output_dim={output_dim}.")

        keep_zone = bool(data_art.meta.get("keep_zone", False))
        zone_emb_dim = int(cfg.get("zone_emb_dim", 8))

        if bool(cfg.get("keep_zone", keep_zone)) and not keep_zone:
            pass

        d_model = int(cfg.get("d_model", 64))
        nhead = int(cfg.get("nhead", 4))
        num_layers = int(cfg.get("num_layers", 2))
        dim_ff = int(cfg.get("dim_feedforward", 128))
        dropout = float(cfg.get("dropout", 0.1))

        eps = float(cfg.get("eps", 1e-4))
        kappa_max = cfg.get("kappa_max", None)

        num_components = int(cfg.get("num_components", 1))
        if num_components != 1:
            raise ValueError("TransformerBetaPipeline currently supports num_components=1 only.")

        use_vae = bool(cfg.get("use_vae", False))

        def _load_ckpt(model: nn.Module, ckpt_path: str) -> None:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
                state = {k.replace("module.", "", 1): v for k, v in state.items()}
            model.load_state_dict(state, strict=False)

        if use_vae:
            vae_mode = str(cfg.get("vae_mode", "e2e")).lower().strip()
            if vae_mode not in ("e2e", "cache"):
                raise ValueError("cfg['vae_mode'] must be one of: 'e2e', 'cache'.")

            if vae_mode == "cache":
                if int(input_dim) <= 0:
                    raise ValueError(
                        "vae_mode='cache' expects input_dim (=latent_dim) > 0. "
                        "This usually means your latent cache was not loaded correctly. "
                        "Verify your cache loading replaces Xtr/Xva/Xte and updates DataArtifacts.input_dim."
                    )

                in_dim = int(input_dim + (zone_emb_dim if keep_zone else 0))

                beta_core = BetaTimeSeriesTransformer(
                    input_dim=in_dim,
                    d_model=d_model,
                    nhead=nhead,
                    num_layers=num_layers,
                    dim_feedforward=dim_ff,
                    dropout=dropout,
                    num_components=1,
                    eps=eps,
                    kappa_max=kappa_max,
                )

                if not keep_zone:
                    return beta_core

                zone_mapping = data_art.zone_mapping
                if zone_mapping is None:
                    raise ValueError("keep_zone=True but data_art.zone_mapping is None.")
                n_train_zones = int(len(zone_mapping))
                return BetaTransformerWithZoneEmbedding(
                    base=beta_core, n_zones_with_unk=n_train_zones + 1, emb_dim=zone_emb_dim
                )

            vae_cfg = cfg.get("vae", {})
            if not isinstance(vae_cfg, dict):
                raise ValueError("cfg['vae'] must be a dict of SequenceVAE init kwargs.")
            vae = SequenceVAE(**vae_cfg)

            vae_ckpt = cfg.get("vae_ckpt_path", None)
            if vae_ckpt:
                _load_ckpt(vae, vae_ckpt)

            latent_dim = None
            for attr in ("latent_dim", "z_dim", "dim_latent"):
                if hasattr(vae, attr):
                    latent_dim = int(getattr(vae, attr))
                    break
            if latent_dim is None:
                latent_dim = int(cfg.get("vae_latent_dim", 0))
            if latent_dim <= 0:
                raise ValueError("Could not infer VAE latent dim. Set cfg['vae_latent_dim'].")

            in_dim = int(latent_dim + (zone_emb_dim if keep_zone else 0))

            beta_core = BetaTimeSeriesTransformer(
                input_dim=in_dim,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                dim_feedforward=dim_ff,
                dropout=dropout,
                num_components=1,
                eps=eps,
                kappa_max=kappa_max,
            )

            if keep_zone:
                zone_mapping = data_art.zone_mapping
                if zone_mapping is None:
                    raise ValueError("keep_zone=True but data_art.zone_mapping is None.")
                n_train_zones = int(len(zone_mapping))
                beta_model = BetaTransformerWithZoneEmbedding(
                    base=beta_core, n_zones_with_unk=n_train_zones + 1, emb_dim=zone_emb_dim
                )
            else:
                beta_model = beta_core

            return VAETransformerBetaForecast(
                vae=vae,
                beta_transformer=beta_model,
                freeze_vae=bool(cfg.get("freeze_vae", True)),
                use_sample_z=bool(cfg.get("use_sample_z", False)),
                detach_z_when_frozen=True,
                return_aux=False,
            )

        in_dim = int(input_dim + (zone_emb_dim if keep_zone else 0))

        beta_core = BetaTimeSeriesTransformer(
            input_dim=in_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_ff,
            dropout=dropout,
            num_components=1,
            eps=eps,
            kappa_max=kappa_max,
        )

        if not keep_zone:
            return beta_core

        zone_mapping = data_art.zone_mapping
        if zone_mapping is None:
            raise ValueError("keep_zone=True but data_art.zone_mapping is None.")
        n_train_zones = int(len(zone_mapping))
        return BetaTransformerWithZoneEmbedding(
            base=beta_core, n_zones_with_unk=n_train_zones + 1, emb_dim=zone_emb_dim
        )

    def model_forward(self, model: nn.Module, batch):
        """
        Return raw (alpha, beta).
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
        """Beta NLL on y scaled to (0,1)."""
        alpha, beta = pred
        y = y.to(self.device)
        if y.dim() == 1:
            y = y.unsqueeze(-1)

        y_min = float(self.cfg.get("y_min", 0.0))
        y_max = float(self.cfg.get("y_max", 1.0))
        eps = float(self.cfg.get("eps", 1e-4))

        y01 = _scale_to_unit(y, y_min=y_min, y_max=y_max, eps=eps)
        return beta_nll(y01, alpha, beta)

    def _beta_mu01(self, pred) -> torch.Tensor:
        """
        Return Beta mean on unit scale, shape [B,1].
        """
        alpha, beta = pred
        mu01 = alpha / (alpha + beta)

        if mu01.dim() == 1:
            mu01 = mu01.unsqueeze(-1)
        elif mu01.dim() == 2 and mu01.size(-1) != 1:
            mu01 = mu01[:, :1]

        return mu01

    def postprocess_mu(self, pred) -> torch.Tensor:
        """
        Return Beta mean (mu) on original target scale, shape [B,1].
        """
        mu01 = self._beta_mu01(pred)
        y_min = float(self.cfg.get("y_min", 0.0))
        y_max = float(self.cfg.get("y_max", 1.0))
        return _inverse_scale_from_unit(mu01, y_min=y_min, y_max=y_max)

    def postprocess_pred(self, pred) -> torch.Tensor:
        """
        Keep point prediction behavior unchanged.
        Here point prediction == mu in original scale.
        """
        return self.postprocess_mu(pred)

    @torch.no_grad()
    def postprocess_interval(
        self, pred, coverage: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build prediction interval from Beta(alpha, beta) using SciPy beta.ppf.

        Returns:
            lower, upper: [B,1] in ORIGINAL scale
        """
        alpha, beta = pred

        if coverage is None:
            coverage = float(self.cfg.get("interval_coverage", 0.9))

        eps = float(self.cfg.get("eps", 1e-4))
        tail = (1.0 - coverage) / 2.0
        tail = max(tail, eps)
        upper_p = 1.0 - tail

        if alpha.dim() == 1:
            alpha_ = alpha.unsqueeze(-1)
            beta_ = beta.unsqueeze(-1)
        else:
            alpha_ = alpha[:, :1] if alpha.size(-1) != 1 else alpha
            beta_ = beta[:, :1] if beta.size(-1) != 1 else beta

        alpha_ = torch.clamp(alpha_, min=eps)
        beta_ = torch.clamp(beta_, min=eps)

        try:
            from scipy.stats import beta as sp_beta
        except Exception as e:
            raise ImportError(
                "SciPy is required for Beta intervals when torch Beta.cdf/icdf is unavailable. "
                "Install with: pip install -U scipy"
            ) from e

        a_np = alpha_.detach().cpu().numpy().astype(np.float64, copy=False)
        b_np = beta_.detach().cpu().numpy().astype(np.float64, copy=False)

        p_low_np = np.full_like(a_np, tail, dtype=np.float64)
        p_high_np = np.full_like(a_np, upper_p, dtype=np.float64)

        q_low01_np = sp_beta.ppf(p_low_np, a_np, b_np)
        q_high01_np = sp_beta.ppf(p_high_np, a_np, b_np)

        q_low01_np = np.clip(q_low01_np, eps, 1.0 - eps)
        q_high01_np = np.clip(q_high01_np, eps, 1.0 - eps)

        q_low01 = torch.from_numpy(q_low01_np).to(device=alpha.device, dtype=alpha.dtype)
        q_high01 = torch.from_numpy(q_high01_np).to(device=alpha.device, dtype=alpha.dtype)

        y_min = float(self.cfg.get("y_min", 0.0))
        y_max = float(self.cfg.get("y_max", 1.0))

        lower = _inverse_scale_from_unit(q_low01, y_min=y_min, y_max=y_max)
        upper = _inverse_scale_from_unit(q_high01, y_min=y_min, y_max=y_max)
        return lower, upper