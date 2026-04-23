from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union, TYPE_CHECKING

import torch
import torch.nn as nn

from src.models.representation.vae import SequenceVAE

if TYPE_CHECKING:
    from src.models.probabilistic.beta.beta_transformer import BetaTimeSeriesTransformer
    from src.models.deterministic.transformer import TimeSeriesTransformer


class VAETransformerBetaForecast(nn.Module):
    """
    Wrapper: VAE encoder -> latent sequence -> Beta Transformer (alpha, beta)

    Supports optional zone ids:
      - If the inner beta_transformer accepts (z), we pass it.
      - Otherwise we fallback to beta_transformer(z_latent) only.

    Input:
      x: [B, L, input_dim_raw]
      z: [B] (optional zone indices)

    Output:
      alpha, beta:
        if num_components == 1: [B, 1]
        else: [B, K]
    """

    def __init__(
        self,
        vae: SequenceVAE,
        beta_transformer: nn.Module,
        freeze_vae: bool = True,
        use_sample_z: bool = False,
        detach_z_when_frozen: bool = True,
        return_aux: bool = False,   # if True, return (alpha, beta, {"mu":..., "logvar":...})
    ):
        super().__init__()
        self.vae = vae
        self.beta_transformer = beta_transformer
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
        z: Optional[torch.Tensor] = None,
        *,
        vae_kwargs: Optional[Dict[str, Any]] = None,
        beta_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]],
    ]:
        vae_kwargs = vae_kwargs or {}
        beta_kwargs = beta_kwargs or {}

        if self.freeze_vae:
            self.vae.eval()

        # Encode
        if self.freeze_vae:
            with torch.no_grad():
                mu, logvar = self.vae.encode(x, **vae_kwargs)
        else:
            mu, logvar = self.vae.encode(x, **vae_kwargs)

        # Latent: sample or deterministic mean
        if self.use_sample_z and hasattr(self.vae, "reparameterize"):
            z_latent = self.vae.reparameterize(mu, logvar)
        else:
            z_latent = mu

        if self.freeze_vae and self.detach_z_when_frozen:
            z_latent = z_latent.detach()

        if z_latent.dim() != 3:
            raise ValueError(f"Expected latent sequence z to be [B, L, dz], got {tuple(z_latent.shape)}")

        # Beta Transformer: try (z_latent, z_zone) if provided
        if z is None:
            alpha, beta = self.beta_transformer(z_latent, **beta_kwargs)
        else:
            z = z.long()
            try:
                alpha, beta = self.beta_transformer(z_latent, z, **beta_kwargs)
            except TypeError:
                alpha, beta = self.beta_transformer(z_latent, **beta_kwargs)

        if self.return_aux:
            aux = {"mu": mu, "logvar": logvar}
            return alpha, beta, aux

        return alpha, beta




class VAETransformerDetForecast(nn.Module):
    def __init__(
        self,
        vae: nn.Module,
        transformer: nn.Module,
        freeze_vae: bool = True,
        use_sample_z: bool = False,
        detach_z_when_frozen: bool = True,
    ):
        super().__init__()
        self.vae = vae
        self.transformer = transformer
        self.freeze_vae = freeze_vae
        self.use_sample_z = use_sample_z
        self.detach_z_when_frozen = detach_z_when_frozen

        if self.freeze_vae:
            for p in self.vae.parameters():
                p.requires_grad = False

    def forward(
        self,
        x: torch.Tensor,
        z: Optional[torch.Tensor] = None,  # zone id (optional)
        *,
        vae_kwargs: Optional[Dict[str, Any]] = None,
        tr_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        vae_kwargs = vae_kwargs or {}
        tr_kwargs = tr_kwargs or {}

        # If VAE is frozen, force eval mode to avoid dropout/BN randomness
        # (outer model.train() can flip child modules back to train mode)
        if self.freeze_vae:
            self.vae.eval()
            with torch.inference_mode():
                mu, logvar = self.vae.encode(x, **vae_kwargs)
        else:
            mu, logvar = self.vae.encode(x, **vae_kwargs)

        # Use either a stochastic latent sample z (if enabled) or the deterministic mean mu
        if self.use_sample_z and hasattr(self.vae, "reparameterize"):
            latent = self.vae.reparameterize(mu, logvar)
        else:
            latent = mu

        # Safety: ensure no gradient flows into VAE when it is frozen
        if self.freeze_vae and self.detach_z_when_frozen:
            latent = latent.detach()

        # Expected latent shape: [B, L, dz]
        if latent.dim() != 3:
            raise ValueError(f"Expected latent to be [B, L, dz], got {tuple(latent.shape)}")

        # Forward through the downstream transformer.
        # Supports both signatures:
        #   - base transformer: forward(x)
        #   - zone wrapper:     forward(x, zone_id)
        if z is None:
            y_hat = self.transformer(latent, **tr_kwargs)
        else:
            try:
                y_hat = self.transformer(latent, z, **tr_kwargs)
            except TypeError:
                # Fallback: transformer does not accept zone_id
                y_hat = self.transformer(latent, **tr_kwargs)

        # Normalize output to [B, 1] for single-step forecasting
        if y_hat.dim() == 1:
            y_hat = y_hat.unsqueeze(-1)
        elif y_hat.dim() == 2 and y_hat.size(-1) != 1:
            y_hat = y_hat[:, :1]

        return y_hat
