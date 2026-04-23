"""
Probabilistic metrics for distributional / quantile forecasts.
Includes:
- Pinball Loss
- Multi-quantile Pinball Loss
- CRPS (based on ensemble sample approximation)
- PICP (Prediction Interval Coverage Probability)
- AIW (Average Interval Width)
"""

from typing import Union, Iterable, Sequence
import numpy as np
import torch

ArrayLike = Union[Iterable[float], np.ndarray]

def _to_1d_array(x: ArrayLike) -> np.ndarray:
    return np.asarray(x).reshape(-1)


def pinball_loss(y: torch.Tensor, q: torch.Tensor, quantiles: Sequence[float]) -> torch.Tensor:
    """
    y: (B,) or (B,1)
    q: (B,K)
    """
    if y.ndim == 2 and y.size(1) == 1:
        y = y[:, 0]
    y = y.unsqueeze(-1)  # (B,1)

    taus = torch.tensor(quantiles, device=q.device, dtype=q.dtype).view(1, -1)  # (1,K)
    diff = y - q  # (B,K)
    loss = torch.maximum(taus * diff, (taus - 1.0) * diff)  # (B,K)
    return loss.mean()


def pinball_loss_multi(
    y_true,
    q_preds: np.ndarray,
    quantiles: Sequence[float],
) -> float:
    """
    Multi-quantile pinball loss (numpy, for evaluation).
    y_true: (n,)
    q_preds: (n, K)
    """
    y_true = _to_1d_array(y_true)
    q_preds = np.asarray(q_preds)

    if q_preds.shape[0] != y_true.shape[0]:
        raise ValueError("The number of rows of q_preds should be equal to the length of y_true.")
    if q_preds.shape[1] != len(quantiles):
        raise ValueError("The number of columns of q_preds should be equal to the number of quantiles.")

    losses = []
    for j, q in enumerate(quantiles):
        losses.append(pinball_loss(y_true, q_preds[:, j], q))
    return float(np.mean(losses))


def pinball_loss_multi_np(y_true, q_preds: np.ndarray, quantiles: Sequence[float]) -> float:
    y = np.asarray(y_true).reshape(-1, 1)          # (n,1)
    Q = np.asarray(q_preds)                        # (n,K)
    taus = np.asarray(quantiles).reshape(1, -1)    # (1,K)

    diff = y - Q
    loss = np.maximum(taus * diff, (taus - 1.0) * diff)
    return float(np.mean(loss))



def crps_ensemble(y_true: ArrayLike, ensemble_preds: np.ndarray) -> float:
    """
    CRPS approximation using ensemble forecasts.

    Parameters
    -----------
    y_true: array-like, shape(n_samples,)
    ensemble_preds: np.ndarray, shape (n_samples, n_members)
       every row should be multiple ensemble predictions for the sample.

    Returns
    --------
    float 
         Average CRPS.
    """

    y_true = _to_1d_array(y_true)
    F = np.asanyarray(ensemble_preds)

    if F.shape[0] != y_true.shape[0]:
        raise ValueError("The number of rows of ensemble_preds should be equal to the length of y_true.")
    
    # |y - x_i|
    term1 = np.mean(np.abs(F - y_true[:, None]), axis=1)
    
    # 0.5 * AVG |x_i - x_j|
    # [n, m] -> pairwise difference
    abs_diff = np.abs(F[:, :, None] - F[:, None, : ])
    term2 = 0.5 * np.mean(abs_diff, axis=(1,2))

    crps = term1 - term2
    return float(np.mean(crps))

def picp(y_true: ArrayLike, y_lower: ArrayLike, y_upper: ArrayLike) -> float:
    """
    Prediction Interval Coverage Probability (PICP)
    """
    y_true = _to_1d_array(y_true)
    y_lower = _to_1d_array(y_lower)
    y_upper = _to_1d_array(y_upper)

    if not(len(y_true) == len(y_lower) == len(y_upper)):
        raise ValueError("y_true, y_lower, y_upper must have the same length.")
    
    inside = (y_true >= y_lower) & (y_true <= y_upper)
    return float(inside.mean())

def aiw(y_lower: ArrayLike, y_upper: ArrayLike) -> float:
    """
    Average Interval Width (AIW).
    """
    y_lower = _to_1d_array(y_lower)
    y_upper = _to_1d_array(y_upper)

    if len(y_lower) != len(y_upper):
        raise ValueError("y_lower and y_upper must have the same length.")
    
    width = y_upper - y_lower
    return float(width.mean())