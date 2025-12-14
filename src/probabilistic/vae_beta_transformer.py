import torch
import torch.nn as nn

class VAEBetaForecast(nn.Module):
    """
    Wrapper model that combines a VAE encoder with a Beta-based transformer head
    to produce prbabilistic forecasts.

    This module takes raw time-series features as input, passes them through a 
    pre-trained VAE encoder to obtain latent representations, and then feeds the
    latent sequence into a BetaTimeSeriesTransformer. The transformer outputs 
    the parameters of one or multiple Beta distributions for each sample.

    Typical workflow:
        X (raw sequence) --> VAE encoder --> latent sequence Z
                          --> BetaTimeSeriesTransformer --> alpha, beta

    Args:
        vae (nn.Module):
            A VAE model that provides an 'encode(x)' method returning (mu, logvar),
            and optionally a 'reparameterize(mu, logvar)' method.
        beta_transformer (nn.Module):
            A time-series model(e.g., BetaTimeSeriesTransformer) that expects latent
            sequence of shape [batch, seq_len, latent_dim] and returns Beta distribution
            parameters (alpha, beta).
        freeze_vae (bool, optional):
            If True, all VAE parameters are frozen and will not be updated during the 
            training of this wrapper. This is the common setup when the VAE is pre-trained 
            separately. Defaults to True.
        use_sample_z (bool, optional):
            If False, the wrapper uses the mean 'mu' from the VAE encoder as a deterministic 
            latent representation. If True and the VAE implements 'reparameterize', it will sample
            'z' from the approximate posterior. Using 'mu' usually leads to more stable training.
    """

    def __init__(
        self,
        vae: nn.Module,
        beta_transformer: nn.Module,
        freeze_vae: bool = True,
        use_sample_z: bool = False,
    ):
        super().__init__()
        self.vae = vae
        self.beta_transformer = beta_transformer
        self.freeze_vae = freeze_vae
        self.use_sample_z = use_sample_z

        if self.freeze_vae:
            # Optionally freeze VAE parameters so only the Beta Transformer is trained.
            for p in self.vae.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the VAE + Beta Transformer pipeline.

        Args:
            x (torch.Tensor):
                Raw input sequence of shape [batch_size, seq_len, input_dim_raw].
                This should match the expected input dimension of the VAE encoder.

        Returns:
            alpha (torch.Tensor):
                Shape [batch_size, num_components] or [batch_size, 1],
                depending on how the BetaTimeSeriesTransformer is defined.
                Represents the alpha parameters of the Beta distribution(s).
            beta (torch.Tensor):
                Shape [batch_size, num_components] or [batch_size, 1],
                representing the beta parameters of the Beta distribution(s).

        Notes:
            - During training, We may compute a Beta negative
              log-likelihood(NLL) loss using the ground-truth target y and
              the predicted (alpha, beta) parameters.
            - If 'use_sample_z' is False, the model behaves like a deterministic
              feature extractor folled by a probablistic head.
        """
        # Encode input sequence into latent space via the VAE encoder
        if self.freeze_vae:
            with torch.no_grad():
                mu, logvar = self.vae.encode(x)
        else:
            mu, logvar = self.vae.encode(x)

        # Choose whether to use sampled z or just the mean mu
        if self.use_sample_z and hasattr(self.vae, "reparameterize"):
            z = self.vae.reparameterize(mu, logvar)   #[B, L, latent_dim]
        else:
            z = mu # deterministic latent representation

        # Pass latent sequence through the Beta Transformer head
        alpha, beta = self.beta_transformer(z)
        return alpha, beta
    

