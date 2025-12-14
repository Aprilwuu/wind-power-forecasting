# src/probabilistic/mc_dropout.py
from typing import Tuple, Dict
import numpy as np
import torch
import torch.nn as nn


def enable_mc_dropout(model: nn.Module):
    """
    Generic function: turn on dropout layers in eval mode.
    """
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


def mc_dropout_predict(
    model: nn.Module,
    xb: torch.Tensor,
    device: torch.device,
    mc_runs: int = 50,
    quantiles=(0.05, 0.5, 0.95),
) -> Tuple[np.ndarray, np.ndarray, Dict[float, np.ndarray], np.ndarray]:
    """
    Run MC Dropout inference.

    Args:
        model: trained TCN (must have dropout layers)
        xb: input batch [B, L, D]
        device: torch device
        mc_runs: number of MC samples

    Returns:
        preds_mean: [B]
        preds_std:  [B]
        q_dict:     {q: [B]} for each quantile
        preds_all:  [mc_runs, B]
    """
    model.eval()

    model.enable_mc_dropout()
 

    xb = xb.to(device)
    preds_list = []

    with torch.no_grad():
        for _ in range(mc_runs):
            out = model(xb)                       # [B, 1] or [B, H]
            out_np = out.squeeze(-1).cpu().numpy()  # [B]
            preds_list.append(out_np)

    preds_all = np.stack(preds_list, axis=0)       # [mc_runs, B]
    preds_mean = preds_all.mean(axis=0)
    preds_std = preds_all.std(axis=0)

    q_dict = {q: np.quantile(preds_all, q, axis=0) for q in quantiles}

    return preds_mean, preds_std, q_dict, preds_all