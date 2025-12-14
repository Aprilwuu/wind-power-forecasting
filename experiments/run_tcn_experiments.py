import sys
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

# ==========make sure import src.XXX =========
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src.data.load import load_raw_data
from src.data.make_dataset import make_dataset
from src.features.build_features import make_seq_features
from src.models.tcn import TCN
from src.train.train_tcn import train_tcn, evaluate_on_loader
from src.probabilistic.mc_dropout import mc_dropout_predict

logging.basicConfig(level=logging.INFO)


def main():
    # ===== 1. Loading & Cleaning data =====
    data_path = Path("data/processed/gefcom_wind_all_zones.csv")
    print("Loading raw data...")
    df = load_raw_data(data_path)

    print("Cleaning / imputing...")
    df = make_dataset(df)

    # ===== 2. Build sequential features =====
    lookback = 24   # using past 24h data
    horizon = 1     # predict single step

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

    X_np = seq_out["X"]   # [N, L, D]
    y_np = seq_out["y"]   # [N] or [N, H]

    print("Seq shapes:", X_np.shape, y_np.shape)

    # ===== 3. convert to torch.Tensor =====
    X = torch.from_numpy(X_np).float()               # [N, L, D]
    y = torch.from_numpy(y_np).float().view(-1, 1)   # [N, 1]

    N = X.size(0)
    train_N = int(N * 0.8)
    val_N = int(N * 0.9)

    X_train, y_train = X[:train_N], y[:train_N]
    X_val, y_val = X[train_N:val_N], y[train_N:val_N]
    X_test, y_test = X[val_N:], y[val_N:]

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    test_ds = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    # ===== 4. Experiments configs =====
    _, L, D = X_train.shape  # L=lookback, D=features

    experiments = [
        {
            "name": "tcn_baseline",
            "channels": [32, 64, 64],
            "kernel_size": 3,
            "dropout": 0.2,
        },
        {
            "name": "tcn_wide",
            "channels": [64, 64, 64],
            "kernel_size": 3,
            "dropout": 0.2,
        },
        {
            "name": "tcn_deep",
            "channels": [32, 64, 128],
            "kernel_size": 3,
            "dropout": 0.2,
        },
        {
            "name": "tcn_kernel5",
            "channels": [32, 64, 64],
            "kernel_size": 5,
            "dropout": 0.2,
        },
    ]

    fig_dir = Path("reports/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    results_path = Path("reports/tables/tcn_results.csv")
    results_path.parent.mkdir(parents=True, exist_ok=True)

    prob_dir = Path("reports/tables")
    prob_dir.mkdir(parents=True, exist_ok=True)

    # ===== 5. Loop over TCN configs =====
    for exp in experiments:
        model_name = exp["name"]
        channels = exp["channels"]
        ksize = exp["kernel_size"]
        dropout = exp["dropout"]

        print("=" * 80)
        print(f"Running experiment: {model_name}")
        print(f" channels={channels}, kernel_size={ksize}, dropout={dropout}")
        print("=" * 80)

        model = TCN(
            input_dim=D,
            output_dim=horizon,
            channels=channels,
            kernel_size=ksize,
            dropout=dropout,
        )

        # ======= Train =============
        model, history = train_tcn(
            model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=10,
            lr=1e-3,
        )

        print("Training done.")
        print("Train loss history:", history["train_loss"])
        if history["val_loss"]:
            print("Val loss history:", history["val_loss"])

        # ======= Eval on val / test ===============
        device = next(model.parameters()).device
        val_rmse = evaluate_on_loader(model, val_loader, device)
        test_rmse = evaluate_on_loader(model, test_loader, device)

        print(f"[{model_name}] Validation RMSE: {val_rmse:.4f}")
        print(f"[{model_name}] Test RMSE:       {test_rmse:.4f}")

        # ======= Plot and save loss curves ========
        plt.figure()
        plt.plot(history["train_loss"], label="Train loss")
        if history["val_loss"]:
            plt.plot(history["val_loss"], label="Val loss")
        plt.xlabel("Epoch")
        plt.ylabel("MSE loss")
        plt.title(f"TCN training curve - {model_name}")
        plt.legend()
        plt.grid(True)

        out_fig_path = fig_dir / f"tcn_loss_{model_name}.png"
        plt.savefig(out_fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved loss figure to: {out_fig_path}")

        # ======= Append RMSE results to CSV =========
        results_now = {
            "model_name": model_name,
            "lookback": lookback,
            "horizon": horizon,
            "channels": str(channels),
            "kernel_size": ksize,
            "dropout": dropout,
            "val_rmse": float(val_rmse),
            "test_rmse": float(test_rmse),
        }
        results_df = pd.DataFrame([results_now])

        if results_path.exists():
            results_df.to_csv(results_path, mode="a", header=False, index=False)
        else:
            results_df.to_csv(results_path, index=False)

        print(f"Appended results to: {results_path}")

        # ============ Save model checkpoint =========
        ckpt_dir = Path("models/checkpoints")
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"{model_name}.pt"
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved model checkpoint to: {ckpt_path}")

        # =============== 6. MC Dropout: test uncertainty =============
        print("\nRunning MC Dropout inference (100 samples)...")
        all_means, all_stds = [], []
        all_q05, all_q50, all_q95 = [], [], []
        all_y_true = []

        for xb, yb in test_loader:
            mean_pred, std_pred, q_dict, _ = mc_dropout_predict(
                model,
                xb,
                device=device,
                mc_runs=100,
            )
            all_means.append(mean_pred)
            all_stds.append(std_pred)
            all_q05.append(q_dict[0.05])
            all_q50.append(q_dict[0.5])
            all_q95.append(q_dict[0.95])
            all_y_true.append(yb.numpy().ravel())

        all_means = np.concatenate(all_means)
        all_stds = np.concatenate(all_stds)
        all_q05 = np.concatenate(all_q05)
        all_q50 = np.concatenate(all_q50)
        all_q95 = np.concatenate(all_q95)
        all_y_true = np.concatenate(all_y_true)

        prob_out = pd.DataFrame({
            "mean": all_means,
            "std": all_stds,
            "q05": all_q05,
            "q50": all_q50,
            "q95": all_q95,
            "y_true": all_y_true,
            "model_name": model_name,
        })
        prob_path = prob_dir / f"tcn_prob_{model_name}.csv"
        prob_out.to_csv(prob_path, index=False)
        print(f"Saved probabilistic predictions to: {prob_path}\n")


if __name__ == "__main__":
    main()