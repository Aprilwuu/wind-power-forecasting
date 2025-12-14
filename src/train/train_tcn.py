# src/training/train_tcn.py
from typing import Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.metrics import evaluate_rmse  


def train_tcn(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    num_epochs: int = 20,
    lr: float = 1e-3,
    device: Optional[torch.device] = None,
):
    """
    Simple supervised training loop for TCN with MSE loss.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    best_state = None

    for epoch in range(1, num_epochs + 1):
        # -------- train --------
        model.train()
        running_loss = 0.0
        n_samples = 0

        for xb, yb in train_loader:
            xb = xb.to(device)             # [B, L, D]
            yb = yb.to(device)             # [B] or [B, H]
            yb = yb.view(yb.size(0), -1)   # [B, output_dim]

            optimizer.zero_grad()
            preds = model(xb)              # [B, output_dim]
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)
            n_samples += xb.size(0)

        train_loss = running_loss / n_samples
        history["train_loss"].append(train_loss)

        # -------- validate --------
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

            if val_loss < best_val:
                best_val = val_loss
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }

            print(
                f"Epoch {epoch:03d} | train_loss={train_loss:.4f} "
                f"| val_loss={val_loss:.4f}"
            )
        else:
            print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


def evaluate_on_loader(model: nn.Module, loader: DataLoader, device: torch.device):
    """
    Evaluate RMSE on a dataloader using your src.metrics.evaluate_rmse.
    """
    model.eval()
    y_true_list = []
    y_pred_list = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)             # [B, L, D]
            yb = yb.to(device).view(yb.size(0), -1)   # [B, 1]

            preds = model(xb)

            y_true_list.append(yb.cpu())
            y_pred_list.append(preds.cpu())

    y_true = torch.cat(y_true_list, dim=0).numpy().ravel()
    y_pred = torch.cat(y_pred_list, dim=0).numpy().ravel()

    rmse = evaluate_rmse(y_true, y_pred)
    return rmse