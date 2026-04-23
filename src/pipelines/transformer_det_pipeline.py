# src/pipelines/transformer_pipeline.py
from __future__ import annotations

import torch
from torch import nn

from src.pipelines.forecast_base import ForecastBasePipeline
from src.pipelines.utils_data import DataArtifacts
from src.models.deterministic.transformer import TimeSeriesTransformer
from src.models.representation.vae import SequenceVAE
from src.models.representation.vae_transformer_wrappers import VAETransformerDetForecast


class TransformerWithZoneEmbedding(nn.Module):
    """
    Zone embedding wrapper:
      - input:  x [B, L, F], zone id z [B]
      - output: base([x, emb(z)]) -> [B, H] (typically H=1)
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
    Deterministic Transformer pipeline:
      - supports keep_zone via TransformerWithZoneEmbedding
      - supports VAE in two modes:
          (1) vae_mode="e2e": end-to-end, VAE encoder runs inside the model (existing behavior)
          (2) vae_mode="cache": VAE latents are precomputed and fed by the dataloader
      - training/early-stop/eval is handled by ForecastBasePipeline
    """

    def build_model(self, input_dim: int, output_dim: int, data_art: DataArtifacts) -> nn.Module:
        cfg = self.cfg

        # IMPORTANT:
        # Always trust the dataloader meta for keep_zone.
        # For Track2 LOFO, your build_seq_dataloaders() explicitly forces keep_zone=False
        # to support "unknown zone" generalization. That should override YAML.
        keep_zone = bool(data_art.meta.get("keep_zone", False))
        zone_emb_dim = int(cfg.get("zone_emb_dim", 8))

        # If YAML sets keep_zone=True but loader disables it (Track2), keep behavior stable.
        if bool(cfg.get("keep_zone", keep_zone)) and not keep_zone:
            pass

        d_model = int(cfg.get("d_model", 64))
        nhead = int(cfg.get("nhead", 4))
        num_layers = int(cfg.get("num_layers", 2))
        dim_ff = int(cfg.get("dim_feedforward", 128))
        dropout = float(cfg.get("dropout", 0.1))

        use_vae = bool(cfg.get("use_vae", False))

        # ----------------------------
        # No VAE path (raw features)
        # ----------------------------
        if not use_vae:
            # input_dim here is the raw feature dim from dataloader: x is [B, L, input_dim]
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
                raise ValueError("keep_zone=True but data_art.zone_mapping is None. Check build_seq_dataloaders.")
            n_train_zones = int(len(zone_mapping))
            return TransformerWithZoneEmbedding(base=base, n_zones_with_unk=n_train_zones + 1, emb_dim=zone_emb_dim)

        # ----------------------------
        # VAE path (two modes)
        # ----------------------------
        # Default is "e2e" to preserve your existing experiments.
        vae_mode = str(cfg.get("vae_mode", "e2e")).lower().strip()
        if vae_mode not in ("e2e", "cache"):
            raise ValueError("cfg['vae_mode'] must be one of: 'e2e', 'cache'.")

        # ============================================================
        # Mode A: cache (recommended for fair comparisons & speed)
        #   - Dataloader provides latent sequences directly:
        #       x is [B, L, latent_dim]
        #   - Therefore input_dim == latent_dim here.
        #   - No SequenceVAE is built and no wrapper is used.
        # ============================================================
        if vae_mode == "cache":
            if int(input_dim) <= 0:
                raise ValueError(
                    "vae_mode='cache' expects input_dim (=latent_dim) > 0. "
                    "This usually means your latent cache was not loaded correctly. "
                    "Verify your cache loading replaces Xtr/Xva/Xte and updates DataArtifacts.input_dim."
                )

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
                raise ValueError("keep_zone=True but data_art.zone_mapping is None. Check build_seq_dataloaders.")
            n_train_zones = int(len(zone_mapping))
            return TransformerWithZoneEmbedding(base=base, n_zones_with_unk=n_train_zones + 1, emb_dim=zone_emb_dim)

        # ============================================================
        # Mode B: e2e (existing behavior)
        #   - Dataloader provides raw sequences: x is [B, L, F_raw]
        #   - VAE encoder runs inside the model to produce latent z
        #   - Transformer consumes z (plus zone embedding if enabled)
        # ============================================================
        vae_cfg = cfg.get("vae", {})
        if not isinstance(vae_cfg, dict):
            raise ValueError("cfg['vae'] must be a dict of SequenceVAE init kwargs.")

        vae = SequenceVAE(**vae_cfg)

        # Optional: load pretrained VAE encoder weights
        vae_ckpt = cfg.get("vae_ckpt_path", None)
        if vae_ckpt:
            ckpt = torch.load(vae_ckpt, map_location="cpu")
            state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            # Handle DataParallel "module." prefix
            if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
                state = {k.replace("module.", "", 1): v for k, v in state.items()}
            vae.load_state_dict(state, strict=False)

        # Infer latent_dim (prefer attribute, else cfg fallback)
        latent_dim = None
        for attr in ("latent_dim", "z_dim", "dim_latent"):
            if hasattr(vae, attr):
                latent_dim = int(getattr(vae, attr))
                break
        if latent_dim is None:
            latent_dim = int(cfg.get("vae_latent_dim", 0))
        if latent_dim <= 0:
            raise ValueError(
                "Could not infer latent_dim from VAE. "
                "Set cfg['vae_latent_dim'] or ensure SequenceVAE has attribute latent_dim/z_dim."
            )

        # Transformer expects latent sequences as input:
        #   z has dim latent_dim, so transformer input_dim = latent_dim (+ zone_emb_dim if keep_zone)
        in_dim = int(latent_dim + (zone_emb_dim if keep_zone else 0))

        base = TimeSeriesTransformer(
            input_dim=in_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_ff,
            dropout=dropout,
            output_dim=int(output_dim),
        )

        if keep_zone:
            zone_mapping = data_art.zone_mapping
            if zone_mapping is None:
                raise ValueError("keep_zone=True but data_art.zone_mapping is None. Check build_seq_dataloaders.")
            n_train_zones = int(len(zone_mapping))
            core = TransformerWithZoneEmbedding(base=base, n_zones_with_unk=n_train_zones + 1, emb_dim=zone_emb_dim)
        else:
            core = base

        freeze_vae = bool(cfg.get("freeze_vae", True))
        use_sample_z = bool(cfg.get("use_sample_z", False))

        # Wrapper: raw x -> VAE encoder -> latent z -> (core transformer or zone wrapper)
        model = VAETransformerDetForecast(
            vae=vae,
            transformer=core,
            freeze_vae=freeze_vae,
            use_sample_z=use_sample_z,
            detach_z_when_frozen=True,
        )
        return model
