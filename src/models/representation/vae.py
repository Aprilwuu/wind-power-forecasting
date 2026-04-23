import torch
import torch.nn as nn


class SequenceVAE(nn.Module):
    """
    Time-step-wise (MLP) VAE for sequence features.

    Input:
        x: (B, L, D_in)

    Outputs:
        recon_x: (B, L, D_in)
        mu:      (B, L, D_z)
        logvar:  (B, L, D_z)
        z:       (B, L, D_z)  (sampled via reparameterization)
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, latent_dim: int = 16):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.latent_dim = int(latent_dim)

        # Encoder: apply the same MLP to each time step feature vector
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(self.hidden_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self.hidden_dim, self.latent_dim)

        # Decoder: latent -> reconstruct original feature vector per time step
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.input_dim),
        )

    def encode(self, x: torch.Tensor):
        """
        Encode x into distribution parameters.
        Args:
            x: (B, L, D_in)
        Returns:
            mu, logvar: (B, L, D_z)
        """
        if x.dim() != 3:
            raise ValueError(f"Expected x as [B,L,D], got shape={tuple(x.shape)}")
        B, L, D = x.shape
        h = self.encoder(x.reshape(B * L, D))          # (B*L, hidden_dim)
        mu = self.fc_mu(h).reshape(B, L, self.latent_dim)
        logvar = self.fc_logvar(h).reshape(B, L, self.latent_dim)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = mu + eps * sigma
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent sequence back to reconstructed features.
        Args:
            z: (B, L, D_z)
        Returns:
            recon_x: (B, L, D_in)
        """
        if z.dim() != 3:
            raise ValueError(f"Expected z as [B,L,Dz], got shape={tuple(z.shape)}")
        B, L, Dz = z.shape
        h = self.decoder(z.reshape(B * L, Dz))
        recon_x = h.reshape(B, L, self.input_dim)
        return recon_x

    @torch.no_grad()
    def extract_mu(self, x: torch.Tensor) -> torch.Tensor:
        """
        Deterministic latent extraction (recommended for caching):
            returns mu: (B, L, D_z)
        """
        mu, _ = self.encode(x)
        return mu

    def forward(self, x: torch.Tensor):
        """
        Full VAE forward pass.
        Returns:
            recon_x, mu, logvar, z
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar, z
