import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNSequenceVAE(nn.Module):
    """
    RNN-based VAE for time-series features.
    Input: x (B, L, D_in)
    Output: recon_x (B, L, D_in), mu (B, L, D_z), logvar (B, L, D_z), z (B, L, D_z)

    This model keeps the same interface as your original SequenceVAE:
        - encode(x) -> mu, logvar
        - reparameterize(mu, logvar) -> z
        - decode(z) -> recon_x
        - forward(x) -> recon_x, mu, logvar, z
    """

    def __init__(self, input_dim=16, hidden_dim=64, latent_dim=16, num_layers=1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Encoder RNN: explicitly models temporal dependencies
        self.encoder_rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,       # x has shape [B, L, D]
            bidirectional=False
        )

        # Map each time-step hidden state into latent mu and logvar
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder RNN: reconstructs the original sequence from latent sequence
        self.decoder_rnn = nn.GRU(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )

        self.fc_out = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        """
        Encode the input sequence.

        Args:
            x: (B, L, D_in)

        Returns:
            mu, logvar: (B, L, D_Z)
        """
        # h_seq: (B, L, hidden_dim)
        h_seq, _ = self.encoder_rnn(x)
        mu = self.fc_mu(h_seq)         # (B, L, latent_dim)
        logvar = self.fc_logvar(h_seq) # (B, L, latent_dim)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick:
            z = mu + eps * sigma

        Args:
            mu, logvar: (B, L, D_z)

        Returns:
            z: (B, L, D_z)
        """
        std = torch(0.5 * logvar)
        eps = torch.randn_like(std)

        return mu + eps * std
    
    def decode(self, z):
        """
        Decode the latent sequence back to the input space.

        Args:
            z: (B, L, D_z)

        Returns:
            recon_x: (B, L, D_in)
        """
        h_seq, _ = self.decoder_rnn(z)      # (B, L, hidden_dim)
        recon_x = self.fc_out(h_seq)        # (B, L, D_in)
        return recon_x
    
    def forward(self, x):
        """
        Full VAE forward pass.
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar, z
    
def vae_loss_fn(x, recon_x, mu, logvar, beta=1.0):
    """
    Standard VAE loss:
        total_loss = reconstrction_loss + beta * KL

    Args:
        x, recon_x: (B, L, D)
        mu. logvar: (B, L, D_z)

    Returns:
        loss, recon_loss, kl
    """
    # Reconstruction loss
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')

    # KL divergence: D_KL(q(z|x) || N(0, I))
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    loss = recon_loss + beta * kl
    return loss, recon_loss, kl
