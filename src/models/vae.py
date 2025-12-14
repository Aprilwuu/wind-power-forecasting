import torch
import torch.nn as nn
import torch.nn.functional as F

class SequenceVAE(nn.Module):
    """
    RNN based VAE for time-series features.
    Input: x (B, L, D_in)
    Output: recon_x (B, L, D_in), mu (B, L, D_z), logvar (B, L, D_z)
    """
    def __init__(self, input_dim=16, hidden_dim=64, latent_dim=16):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        # Encoder: per time_step MLP
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder: latent -> original feature
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def encode(self, x):
        """
        x: (B, L, D_in)
        return mu, logvar: (B, L, D_z)
        """
        B, L, D = x.shape
        h = self.encoder(x.view(B * L, D))        # (B*L, hidden_dim)
        mu = self.fc_mu(h).view(B, L, -1)
        logvar = self.fc_logvar(h).view(B, L, -1)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: z = mu + eps * sigma
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """
        z: (B, L, D_z)
        return recon_x: (B, L, D_in)
        """
        B, L, D_z = z.shape
        h = self.decoder(z.view(B * L, D_z))
        recon_x = h.view(B, L, self.input_dim)
        return recon_x
    
    def forward(self, x):
        """
        Null VAE forward
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar, z
    
