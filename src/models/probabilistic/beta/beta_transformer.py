import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.deterministic.transformer import PositionalEncoding

class BetaTimeSeriesTransformer(nn.Module):
    """
    Time series model that outputs parameters of one or multiple Beta distributions.

    - Input:  x[batch, seq_len, input_dim]
    - Output (num_components=1)
        alpha, beta: [batch, 1] each
      Or (num_components=K):
        alpha, beta: [batch, K]
    """

    def __init__(
        self,
        input_dim: int,              # latent dim from VAE
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        num_components: int = 1, # how many Beta distributions
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_components = num_components

        # 1. projection
        self.input_proj = nn.Linear(input_dim, d_model)

        # 2. positional encoding (reuse your PositionalEncoding)
        self.pos_encoder = PositionalEncoding(d_model=d_model, dropout=dropout)

        # 3. Transformer Encoder 
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # 4. Final layer: output 2 * num_components (for alpha and beta)
        self.fc_out = nn.Linear(d_model, 2 * num_components)
    
    def forward(self, x: torch.Tensor):
        """
        x: [batch, seq_len, input_dim]
        return:
            alpha, beta:
              if num_components == 1: shape [batch, 1]
              else: shape [batch, K]
        """
        B, L, D = x.shape
        
        # project
        x_proj = self.input_proj(x)        # [B, L, d_model]
        x_proj = x_proj.transpose(0, 1)    # [L, B, d_model]

        # position + transformer
        x_enc = self.pos_encoder(x_proj)       # [L, B, d_model]
        memory = self.transformer_encoder(x_enc)     # [L, B, d_model]

        # last time step
        last_hidden = memory[-1, :, :]         # [B, d_model]

        raw = self.fc_out(last_hidden)         # [B, 2 * K]
        raw = raw.view(B, self.num_components, 2)  # [B, K, 2]

        alpha_raw = raw[..., 0]    # [B, K]
        beta_raw = raw[..., 1]     # [B, K]

        # make them positive: softplus + small epsilon
        eps = 1e-4
        alpha = F.softplus(alpha_raw) + eps
        beta = F.softplus(beta_raw) + eps

        return alpha, beta





