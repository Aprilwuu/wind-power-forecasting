from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from src.models.representation.vae import SequenceVAE
from src.models.deterministic.transformer import TimeSeriesTransformer


class VAETransformerForecast(nn.Module):
    """
    Wrapper: VAE encoder -> latent sequence -> Transformer forecast.

    Assumptions:
      - vae.encode(x, **kwargs) returns (mu, logvar), each [B, L, latent_dim] (or compatible)
      - optionally vae.reparameterize(mu, logvar) exists
      - transformer(z, **kwargs) accepts z [B, L, latent_dim]
    """

    def __init__(
        self,
        vae: SequenceVAE,
        transformer: TimeSeriesTransformer,
        freeze_vae: bool = True,
        use_sample_z: bool = False,
        detach_z_when_frozen: bool = True,
        return_aux: bool = False,  # if True, return (out, {"mu":..., "logvar":...})
    ):
        super().__init__()
        self.vae = vae
        self.transformer = transformer
        self.freeze_vae = freeze_vae
        self.use_sample_z = use_sample_z
        self.detach_z_when_frozen = detach_z_when_frozen
        self.return_aux = return_aux

        if self.freeze_vae:
            for p in self.vae.parameters():
                p.requires_grad = False

    def forward(
        self,
        x: torch.Tensor,
        *,
        vae_kwargs: Optional[Dict[str, Any]] = None,
        transformer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Args:
            x: [B, L, input_dim_raw]
            vae_kwargs: extra kwargs for vae.encode(...)
            transformer_kwargs: extra kwargs for transformer(...)

        Returns:
            out or (out, aux)
        """
        vae_kwargs = vae_kwargs or {}
        transformer_kwargs = transformer_kwargs or {}

        # If VAE is frozen, keep it in eval mode to avoid dropout/bn randomness
        if self.freeze_vae:
            self.vae.eval()

        # Encode
        if self.freeze_vae:
            with torch.no_grad():
                mu, logvar = self.vae.encode(x, **vae_kwargs)
        else:
            mu, logvar = self.vae.encode(x, **vae_kwargs)

        # Choose latent representation
        if self.use_sample_z and hasattr(self.vae, "reparameterize"):
            z = self.vae.reparameterize(mu, logvar)
        else:
            z = mu

        if self.freeze_vae and self.detach_z_when_frozen:
            z = z.detach()

        # Forecast in latent space
        out = self.transformer(z, **transformer_kwargs)

        if self.return_aux:
            aux = {"mu": mu, "logvar": logvar}
            return out, aux

        return out
