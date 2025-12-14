# src/models/losses.py
import torch
import torch.nn.functional as F

def vae_loss_fn(x, recon_x, mu, logvar, beta=1.0):
    """
    Standard VAE loss:
        recon_loss + beta * KL
    x, recon_x: (B, L, D)
    """
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    # KL divergence: D_KL( q(z|x) || N(0, I))
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    loss = recon_loss + beta * kl
    return loss, recon_loss, kl


def beta_nll_loss(alpha: torch.Tensor,
                  beta: torch.Tensor,
                  y: torch.Tensor,
                  eps: float = 1e-6) -> torch.Tensor:
    """
    Negative log-likelihood loss for Beta(aopha, beta).

    Args:
        alpha, beta: predicted Beta parameters, shpe [B, K] or [B, 1]
        y:           target in [0, 1], shpe broadcastable to alpha/beta
        eps:         numerical stability

    Returns:
        scalar loss ( mean over all elements)
    """
    y = y.clamp(eps, 1 - eps)
    alpha = alpha.clamp(eps, None)
    beta = beta.clamp(eps, None)

    dist = torch.distributions.Beta(alpha, beta)
    nll = -dist.log_prob(y)  # same shape as y
    return nll.mean()