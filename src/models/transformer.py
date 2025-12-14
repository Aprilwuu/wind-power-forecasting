from typing import Optional

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from src.metrics import evaluate_rmse


class PositionalEncoding(nn.Module):
    """
    Standard Transformer sin-cos position encoding.
    Input: [seq_len, batch, d_model] 
    Output: [seq_len, batch, d_model](including position encoding)
    """

    def __init__(self, d_model: int, dropout: float=0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # pe: [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )  # [d_model/2]

        # even postion using sin, odd position using cos
        pe[:, 0::2] = torch.sin(position * div_term) #[max_len, d_model/2]
        pe[:, 1::2] = torch.cos(position * div_term) #[max_len, d_model/2]

        # Changing to [max_len, 1, d_model] to add [seq_len, batch, d_model] more easily
        pe = pe.unsqueeze(1)
        self.register_buffer("pe", pe) # not counting as parameters, but save/load along with the model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [seq_len, batch, d_model]
        """
        seq_len = x.size(0)
        x = x + self.pe[:seq_len]  # only take the first seq_len items
        return self.dropout(x)
    
class TimeSeriesTransformer(nn.Module):
    """
    Use Transformer Encoder for time series regression.

    Input x: [batch, seq_len, input_dim]
    output y_hat: [batch, output_dim] such as output_dim = 1, predict value for next step
    """

    def __init__(
            self, 
            input_dim: int,     # input feature dimension D
            d_model: int = 64,  # Transformer hidden layer dimension
            nhead: int = 4,      # Multi-head attention
            num_layers: int = 2,  # Encoder layers
            dim_feedforward: int = 128,  # FFN hidden layers
            dropout: float = 0.1,
            output_dim: int = 1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model

        #1. project the original dim to d_model d
        self.input_proj = nn.Linear(input_dim, d_model)

        #2. positional encoding
        self.pos_encoder = PositionalEncoding(d_model=d_model, dropout=dropout)

        #3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,      # we use pytorch default form [seq_len, batch, d_model]
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # 4 final : use final step hidden, project to output_dim
        self.fc_out = nn.Linear(d_model, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, seq_len, input_dim]

        dimension changing process:
        1) x -> x_proj: [B, L, D] -> [B, L, d_model]
        2) changed to [L, B, d_model] to match nn.Transformer
        3) adding positional encoding
        4) pass TransformerEncoder
        5) take the last timestamp output: [B, d_model]
        6) Fullly connect regression: [B, output_dim]
        """
        # B: batch size, L: seq_len, D: input_dim
        B, L, D = x.shape

        # 1) linear project to d_model
        x_proj = self.input_proj(x)    # [B, L, d_model]

        # 2) Transformer expected input is [seq_len, batch, d_model]
        x_proj = x_proj.transpose(0, 1)   # [L, B, d_model]

        # 3) adding positional encoder
        x_enc = self.pos_encoder(x_proj)   # [L, B, d_model]

        # 4) passing Transformer Encoder
        memory = self.transformer_encoder(x_enc)   # [L, B, d_model]

        # 5) take the last timestamp hidden state
        last_hidden = memory[-1, :, : ]    # [B, d_model]

        # 6) output regression value  
        out = self.fc_out(last_hidden)     # [B, output_dim]

        return out
    


def train_transformer(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    num_epochs: int = 20,
    lr: float = 1e-3,
    device: Optional[torch.device] = None,
):
    """
    Training loop for the Transformer model.
    Loss function: MSELoss.
    Returns a history dict with train/val loss.
    """
    # 1) Decide device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2) Move model to device and set up optimizer / loss
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    best_state = None

    for epoch in range(1, num_epochs + 1):
        # --------- Train ---------
        model.train()
        running_loss = 0.0
        n_samples = 0

        for xb, yb in train_loader:
            xb = xb.to(device)                         # [B, L, D]
            yb = yb.to(device).view(yb.size(0), -1)    # [B, output_dim]

            optimizer.zero_grad()
            preds = model(xb)                          # [B, output_dim]
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)
            n_samples += xb.size(0)

        train_loss = running_loss / n_samples
        history["train_loss"].append(train_loss)

        # --------- Validate (if val_loader provided) ---------
        if val_loader is not None:
            model.eval()
            val_running = 0.0
            val_n = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device).view(yb.size(0), -1)

                    preds = model(xb)
                    loss = criterion(preds, yb)

                    val_running += loss.item() * xb.size(0)
                    val_n += xb.size(0)

            val_loss = val_running / val_n
            history["val_loss"].append(val_loss)

            # early stopping / record best parameters
            if val_loss < best_val:
                best_val = val_loss
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }

            print(
                f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}"
            )
        else:
            print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f}")

    # 3) Load best validation weights if available
    if val_loader is not None and best_state is not None:
        model.load_state_dict(best_state)

    return history

def evaluate_on_loader(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    y_true_list = []
    y_pred_list = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device).view(yb.size(0), -1)  # [B, 1]

            preds = model(xb)  # [B, 1]

            y_true_list.append(yb.cpu())
            y_pred_list.append(preds.cpu())

    y_true = torch.cat(y_true_list, dim=0).numpy().ravel()
    y_pred = torch.cat(y_pred_list, dim=0).numpy().ravel()

    rmse = evaluate_rmse(y_true, y_pred)
    return rmse