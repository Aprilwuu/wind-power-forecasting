from src.models.vae import SequenceVAE
from src.models.transformer import TimeSeriesTransformer

import torch
import torch.nn as nn

class VAETransformerForecast(nn.Module):
    """Wrap a VAE encoder and a TimeSeriesTransformer for latent-space forecasting.
    
    This module assumes that the VAE provides an 'encode(x)' method that returns (mu, logvar),
    and optionally a 'reparameterize(mu, logvar)' method.

    The workflow is:
        1) Take raw input x: [batch, seq_len, input_dim_raw]
        2) Use VAE encoder to obtain latent sequence z: [ batch, seq_len, latent_dim]
        3) Feed z into a TimeSeriesTransformer whose 'input_dim' matches latent_dim
    """

    def __init__(
        self,
        vae: nn.Module,
        transformer: TimeSeriesTransformer,
        freeze_vae: bool = True,
        use_sample_z: bool = False,
    ):
        super().__init__()
        self.vae = vae
        self.transformer = transformer
        self.freeze_vae = freeze_vae
        self.use_sample_z = use_sample_z

        if self.freeze_vae:
            for p in self.vae.parameters():
                p.requires_grad = False

    def foward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using VAE encoder + Transformer.
        
        Args:
            x: [batch, seq_len, input_dim_raw]
            
        Returns:
            Model prediction from the underlying Transformer, e.g. [batch, output_dim].
            """
        
        # Encode to latent sequence with VAE
        if self.freeze_vae:
            with torch.no_grad():
                mu, logvar = self.vae.encode(x)
        else:
            mu, logvar = self.vae.encode(x)

        # Either use mean as deterministic embedding or sample z
        if self.use_sample_z and hasattr(self.vae, "reparameterize"):
            z = self.vae.reparameterize(mu, logvar)
        else:
            z = mu
        
        # z: [batch, seq_len, latent_dim] -> Transformer expects [batch, seq_len, input_dim]
        out = self.transformer(z)
        return out