import sys
from pathlib import Path
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader

# ==========make sure import src.XXX =========
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src.data.load import load_raw_data
from src.data.make_dataset import make_dataset
from src.features.build_features import make_seq_features
from src.models.transformer import(
    TimeSeriesTransformer,
    train_transformer,
    evaluate_on_loader,
)


def save_transformer_outputs(model, history, test_rmse, experiment_name, logger):
    # save model
    model_path = ROOT / "models" / f"transformer_{experiment_name}.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    logger.info(f"Saved model to: {model_path}")

    # save RMSE
    csv_path = ROOT / "reports" / "transformer_results.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    write_header = not csv_path.exists()
    with open(csv_path, "a") as f:
        if write_header:
            f.write("experiment, test_rmse\n")
        f.write(f"{experiment_name}, {test_rmse:.6f}\n")

    logger.info(f"Saved RMSE to: {csv_path}")
    
    # 3) save loss figures (if history is available)
    if history is None:
        logger.warning("History is None; skipping loss figure.")
        return

    # support both dict and (train_loss, val_loss) tuple/list
    train_loss = None
    val_loss = None
    if isinstance(history, dict):
        train_loss = history.get("train_loss")
        val_loss = history.get("val_loss")
    elif isinstance(history, (list, tuple)) and len(history) == 2:
        train_loss, val_loss = history

    if train_loss is None or val_loss is None:
        logger.warning("History format not recognized or missing keys; skipping loss figure.")
        return

    fig_path = ROOT / "reports" / f"transformer_loss_{experiment_name}.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(train_loss, label="Train")
    plt.plot(val_loss, label="Val")
    plt.legend()
    plt.title(experiment_name)
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved loss figure to: {fig_path}")


def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # ======= 1. Loading & Cleaning data ============
    data_path = Path("data/processed/gefcom_wind_all_zones.csv")
    logger.info("Loading raw data")
    df = load_raw_data(data_path)

    logger.info("Cleaning / imputihng...")
    df = make_dataset(df)

    # ======= 2. Construct sequence features =========
    lookback = 24  # using past 24 timestamps
    horizon = 1    # single-step prediction

    seq_out = make_seq_features(
        df,
        zone_col="zone_id",
        time_col="datetime",
        target_col="target",
        lookback=lookback,
        horizon=horizon,
        feature_cols=None,
        include_target_as_input=True,
        add_missing_mask=False,
    )

    X_np = seq_out["X"]  # [N, L, D]
    y_np = seq_out["y"]  # [N] or [N, H]

    logger.info(f"Seq shapes: X {X_np.shape}, y {y_np.shape}")

    # ======= 3. Transfer into torch.Tensor =========
    X = torch.from_numpy(X_np).float()             # [N, L, D]
    y = torch.from_numpy(y_np).float().view(-1, 1)   # [N, 1]

    N = X.size(0)
    train_N = int(N * 0.8)
    val_N = int(N * 0.9)

    X_train = X[:train_N]
    y_train = y[:train_N]

    X_val = X[train_N:val_N]
    y_val = y[train_N:val_N]

    X_test = X[val_N: ]
    y_test = y[val_N: ]

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    test_ds = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    # ======== 4. Define model ================
    _, L, D = X_train.shape      # sequence length L, feature dim D
    
    configs = [
        {
            "name": "tr_small",
            "d_model": 32,
            "nhead": 4,
            "num_layers": 1,
            "dim_feedforward": 64,
            "dropout": 0.1,
            "lr": 5e-4,
        },
        {
            "name": "tr_medium",
            "d_model": 64,
            "nhead": 4,
            "num_layers": 2,
            "dim_feedforward": 128,
            "dropout": 0.1,
            "lr": 5e-4,           
        },
       {
            "name": "tr_deeper",
            "d_model": 64,
            "nhead": 4,
            "num_layers": 3,
            "dim_feedforward": 128,
            "dropout": 0.2,
            "lr": 5e-4,
       },    
    ]

    for cfg in configs:
        experiment_name=cfg["name"]
        logger.info("=" * 80)
        logger.info(f"Running Tranformer experiment: {cfg['name']}")
        logger.info(cfg)


        model = TimeSeriesTransformer(
            input_dim=D, # input feature dim
            d_model=cfg["d_model"],             
            nhead=cfg["nhead"],
            num_layers=cfg["num_layers"],
            dim_feedforward=cfg["dim_feedforward"],
            dropout=cfg["dropout"],
        ).to(device)
        
        logger.info(model)

        history = train_transformer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=50,
            lr=cfg["lr"],
            device=device,
        )

        test_rmse = evaluate_on_loader(
            model=model, 
            loader=test_loader, 
            device=device,
            )
        logger.info(f"[{cfg['name']}] Test RMSE: {test_rmse:.4f}")
        
        # Save model + RMSE + Loss curve
        save_transformer_outputs(
            model=model,
            history=history,
            test_rmse=test_rmse,
            experiment_name=experiment_name,
            logger=logger,
        )
 


if __name__ == "__main__":
    main()