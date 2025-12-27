# src/pipelines/tcn_pipeline.py
"""Deterministic TCN pipeline (one single run).

This module is intentionally "config-agnostic":
- It does NOT read YAML.
- It does NOT decide parameter grids / candidates.

A runner (experiment script) should:
- Load YAML configs
- Build the final TCN params for THIS run (base_params + candidate override)
- Loop over seeds / candidates
- Call run_pipeline_any(...) once per run

Pipeline responsibilities:
- Load + clean data
- Split (Track 1 temporal cut) OR (Track 2 LOFO + inner time val)
- Build sequence windows (IMPORTANT: split first, then window)
- Train PyTorch TCN
- Evaluate on val/test
- Save artifacts (model, metrics, config snapshot)
"""

from __future__ import annotations

import json
import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.data.load import load_raw_data
from src.data.make_dataset import make_dataset
from src.data.split import train_valid_split, lofo_time_val_split
from src.features.build_features import make_seq_features
from src.models.tcn import TCN
from src.metrics.deterministic import evaluate_mae, evaluate_r2, evaluate_rmse
from src.models.probabilistic.mc_dropout.mc_dropout import mc_dropout_predict

logger = logging.getLogger(__name__)


# -----------------------------
# Result container
# -----------------------------
@dataclass
class DeterministicTCNResult:
    out_dir: str
    model_path: str
    metrics_path: str
    config_snapshot_path: str
    metrics: Dict[str, Any]


# -----------------------------
# Helpers
# -----------------------------
def _ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Determinism (may slow down)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _device_from_str(device: Optional[str]) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _to_2d_y(y: np.ndarray) -> np.ndarray:
    # make_seq_features returns y as [N] if horizon==1; we want [N, 1]
    if y.ndim == 1:
        return y.reshape(-1, 1)
    return y


def _interval_summary(
    y_true: np.ndarray,
    q_out: Dict[float, np.ndarray],
    lo: float = 0.05,
    hi: float = 0.95,
) -> Dict[str, float]:
    """
    Simple interval summary.
    Returns PICP (coverage) and MPIW (mean interval width).
    """
    y = np.asarray(y_true)
    ql = np.asarray(q_out[float(lo)])
    qu = np.asarray(q_out[float(hi)])

    # ensure [N, H]
    y = _to_2d_y(y)
    ql = _to_2d_y(ql)
    qu = _to_2d_y(qu)

    covered = (y >= ql) & (y <= qu)
    picp = float(np.mean(covered))
    mpiw = float(np.mean(qu - ql))
    return {"picp": picp, "mpiw": mpiw, "lo": float(lo), "hi": float(hi)}


def _save_mc_outputs(
    out_dir: Path,
    split_name: str,
    y_true: np.ndarray,
    p_mean: np.ndarray,
    p_std: np.ndarray,
    q_out: Dict[float, np.ndarray],
) -> Dict[str, str]:
    """
    Save MC Dropout outputs to CSV + NPZ.
    Columns: sample_idx, y_true, mean, std, q05/q50/q95..., y_pred (q50 if exists else mean)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    y = _to_2d_y(np.asarray(y_true))
    mean = _to_2d_y(np.asarray(p_mean))
    std = _to_2d_y(np.asarray(p_std))

    # If horizon > 1, save NPZ only to avoid messy CSV flattening
    if y.shape[1] != 1:
        npz_path = out_dir / f"preds_{split_name}_mc.npz"
        np.savez(
            npz_path,
            y_true=y,
            mean=mean,
            std=std,
            **{f"q{int(float(k)*100):02d}": _to_2d_y(np.asarray(v)) for k, v in q_out.items()},
        )
        return {"npz": str(npz_path)}

    y1 = y[:, 0]
    m1 = mean[:, 0]
    s1 = std[:, 0]

    df = pd.DataFrame(
        {
            "sample_idx": np.arange(len(y1), dtype=int),
            "y_true": y1,
            "mean": m1,
            "std": s1,
        }
    )

    q_keys = sorted([float(k) for k in q_out.keys()])
    for qk in q_keys:
        arr = _to_2d_y(np.asarray(q_out[float(qk)]))[:, 0]
        df[f"q{int(qk * 100):02d}"] = arr

    if 0.5 in q_out:
        df["y_pred"] = df["q50"]
    else:
        df["y_pred"] = df["mean"]

    csv_path = out_dir / f"preds_{split_name}_mc.csv"
    df.to_csv(csv_path, index=False)

    npz_path = out_dir / f"preds_{split_name}_mc.npz"
    np.savez(
        npz_path,
        y_true=y,
        mean=mean,
        std=std,
        **{f"q{int(float(k)*100):02d}": _to_2d_y(np.asarray(v)) for k, v in q_out.items()},
    )

    return {"csv": str(csv_path), "npz": str(npz_path)}


# -----------------------------
# Dataset / Model wrappers
# -----------------------------
class SeqDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,  # [N, L, D]
        y: np.ndarray,  # [N, H]
        zone: Optional[np.ndarray] = None,  # [N]
    ):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
        self.zone = None if zone is None else torch.from_numpy(zone).long()

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int):
        if self.zone is None:
            return self.X[idx], self.y[idx]
        return self.X[idx], self.y[idx], self.zone[idx]


class TCNWithZoneEmbedding(nn.Module):
    """Optional zone embedding for keep_zone=True.

    We concatenate a (learned) zone embedding to every timestep feature vector.
    To support Track2 (unseen held-out site), we reserve index 0 as UNK zone.
    Train zones are mapped to 1..K.
    """

    def __init__(self, base_tcn: TCN, n_zones_with_unk: int, emb_dim: int):
        super().__init__()
        self.base_tcn = base_tcn
        self.emb = nn.Embedding(n_zones_with_unk, emb_dim)

    def forward(self, x: torch.Tensor, zone_idx: torch.Tensor) -> torch.Tensor:
        z = self.emb(zone_idx)  # [B, emb_dim]
        z = z.unsqueeze(1).expand(-1, x.size(1), -1)  # [B, L, emb_dim]
        x_aug = torch.cat([x, z], dim=-1)  # [B, L, D+emb_dim]
        return self.base_tcn(x_aug)


def _build_zone_mapping_from_train(df_train: pd.DataFrame, zone_col: str) -> Dict[Any, int]:
    zones = sorted(df_train[zone_col].unique().tolist())
    return {z: i + 1 for i, z in enumerate(zones)}  # reserve 0 for UNK


def _apply_zone_mapping(df: pd.DataFrame, zone_col: str, mapping: Optional[Dict[Any, int]]) -> pd.DataFrame:
    if mapping is None:
        return df
    df = df.copy()
    df[zone_col] = df[zone_col].map(mapping).fillna(0).astype(int)
    return df


# -----------------------------
# Loss helpers (Huber/SmoothL1/MSE)
# -----------------------------
def _build_loss(name: str = "mse", huber_delta: float = 1.0) -> nn.Module:
    name = (name or "mse").lower().strip()

    if name in {"huber", "huberloss"}:
        # Prefer nn.HuberLoss if available
        try:
            return nn.HuberLoss(delta=float(huber_delta))
        except Exception:
            # fallback to SmoothL1Loss (Huber-like)
            try:
                return nn.SmoothL1Loss(beta=float(huber_delta))
            except Exception:
                return nn.SmoothL1Loss()

    if name in {"smoothl1", "smooth_l1", "smoothl1loss"}:
        try:
            return nn.SmoothL1Loss(beta=float(huber_delta))
        except Exception:
            return nn.SmoothL1Loss()

    # default
    return nn.MSELoss()


# -----------------------------
# Train / Predict
# -----------------------------
def _train_one(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    lr: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
    grad_clip: Optional[float],
    device: torch.device,
    loss_name: str = "mse",
    huber_delta: float = 1.0,
) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = _build_loss(loss_name, huber_delta)

    best_val = float("inf")
    best_state: Dict[str, torch.Tensor] = {}
    bad = 0
    history = {"train_loss": [], "val_rmse": []}

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_losses = []

        for batch in train_loader:
            optim.zero_grad(set_to_none=True)

            if len(batch) == 2:
                x, y = batch
                x, y = x.to(device), y.to(device)
                pred = model(x)
            else:
                x, y, z = batch
                x, y, z = x.to(device), y.to(device), z.to(device)
                pred = model(x, z)

            loss = loss_fn(pred, y)
            loss.backward()

            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optim.step()
            train_losses.append(float(loss.item()))

        train_loss = float(np.mean(train_losses)) if train_losses else math.nan

        # --- val rmse ---
        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 2:
                    x, y = batch
                    x = x.to(device)
                    pred = model(x)
                else:
                    x, y, z = batch
                    x, z = x.to(device), z.to(device)
                    pred = model(x, z)
                ys.append(y.numpy())
                ps.append(pred.detach().cpu().numpy())

        y_val = np.concatenate(ys, axis=0)
        p_val = np.concatenate(ps, axis=0)
        val_rmse = evaluate_rmse(y_val, p_val)

        history["train_loss"].append(train_loss)
        history["val_rmse"].append(val_rmse)

        logger.info(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_rmse={val_rmse:.6f}")

        if val_rmse < best_val - 1e-9:
            best_val = val_rmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                logger.info(f"Early stopping at epoch {epoch} (patience={patience}).")
                break

    summary = {
        "best_val_rmse": float(best_val),
        "epochs_ran": int(len(history["train_loss"])),
        "history": history,
        "loss": {"name": str(loss_name), "huber_delta": float(huber_delta)},
    }
    return summary, best_state


def _predict(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 2:
                x, y = batch
                x = x.to(device)
                pred = model(x)
            else:
                x, y, z = batch
                x, z = x.to(device), z.to(device)
                pred = model(x, z)
            ys.append(y.numpy())
            ps.append(pred.detach().cpu().numpy())
    return np.concatenate(ys, axis=0), np.concatenate(ps, axis=0)


def _predict_mc(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    mc_runs: int,
    quantiles: Tuple[float, ...],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[float, np.ndarray]]:
    """
    MC Dropout probabilistic prediction.

    Returns:
        y_true: [N, H]
        p_mean: [N, H]
        p_std:  [N, H]
        q_dict: {q: [N, H]}
    """
    model.eval()

    ys_all: List[np.ndarray] = []
    mean_all: List[np.ndarray] = []
    std_all: List[np.ndarray] = []
    q_all: Dict[float, List[np.ndarray]] = {float(q): [] for q in quantiles}

    for batch in loader:
        if len(batch) == 2:
            x, y = batch
            z = None
        else:
            x, y, z = batch

        x = x.to(device)
        if z is not None:
            z = z.to(device)

        if z is None:
            p_mean, p_std, q_dict, _ = mc_dropout_predict(
                model=model,
                xb=x,
                device=device,
                mc_runs=int(mc_runs),
                quantiles=quantiles,
                squeeze_single_horizon=False,
                move_to_device=False,
            )
        else:
            # Track1 keep_zone=True path: wrap model(x, z) into a callable model(x)
            class _Wrapper(nn.Module):
                def __init__(self, m: nn.Module, zz: torch.Tensor):
                    super().__init__()
                    self.m = m
                    self.zz = zz

                def forward(self, xb: torch.Tensor) -> torch.Tensor:
                    return self.m(xb, self.zz)

            wrapped = _Wrapper(model, z)
            p_mean, p_std, q_dict, _ = mc_dropout_predict(
                model=wrapped,
                xb=x,
                device=device,
                mc_runs=int(mc_runs),
                quantiles=quantiles,
                squeeze_single_horizon=False,
                move_to_device=False,
            )

        y_np = _to_2d_y(y.numpy())
        p_mean = _to_2d_y(np.asarray(p_mean))
        p_std = _to_2d_y(np.asarray(p_std))

        ys_all.append(y_np)
        mean_all.append(p_mean)
        std_all.append(p_std)
        for q, arr in q_dict.items():
            q_all[float(q)].append(_to_2d_y(np.asarray(arr)))

    y_true = np.concatenate(ys_all, axis=0)
    p_mean = np.concatenate(mean_all, axis=0)
    p_std = np.concatenate(std_all, axis=0)
    q_out: Dict[float, np.ndarray] = {q: np.concatenate(parts, axis=0) for q, parts in q_all.items()}
    return y_true, p_mean, p_std, q_out


# -----------------------------
# Track 1: temporal cuts
# -----------------------------
def run_pipeline(
    *,
    data_path: str | Path,
    out_dir: str | Path,
    train_cut: str,
    val_cut: str,
    lookback: int,
    horizon: int = 1,
    feature_cols: Optional[List[str]] = None,
    include_target_as_input: bool = True,
    add_missing_mask: bool = True,
    target_col: str = "target",
    zone_col: str = "zone_id",
    time_col: str = "datetime",
    keep_zone: bool = False,
    zone_emb_dim: int = 8,
    seed: int | None = None,
    # training
    batch_size: int = 256,
    max_epochs: int = 50,
    patience: int = 8,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    grad_clip: Optional[float] = 1.0,
    device: Optional[str] = None,
    train_loss_name: str = "mse",
    huber_delta: float = 1.0,
    # model
    channels: List[int] = [32, 64, 64],
    kernel_size: int = 3,
    dropout: float = 0.2,
    # probabilistic (MC Dropout)
    use_mc_dropout: bool = False,
    mc_runs: int = 50,
    mc_quantiles: Tuple[float, ...] = (0.05, 0.5, 0.95),
) -> DeterministicTCNResult:
    """Run one deterministic TCN training/evaluation pipeline (Track 1 temporal)."""
    out_dir_p = _ensure_dir(out_dir).resolve()
    _set_seed(seed)
    dev = _device_from_str(device)

    # Snapshot runtime inputs
    config_snapshot_path = out_dir_p / "config_snapshot.json"
    snapshot = {
        "runtime": {
            "data_path": str(Path(data_path).resolve()),
            "out_dir": str(out_dir_p),
            "protocol": "track1_temporal",
            "split": {"train_cut": train_cut, "val_cut": val_cut},
            "seq": {
                "lookback": int(lookback),
                "horizon": int(horizon),
                "feature_cols": feature_cols,
                "include_target_as_input": bool(include_target_as_input),
                "add_missing_mask": bool(add_missing_mask),
            },
            "cols": {"target_col": target_col, "zone_col": zone_col, "time_col": time_col},
            "keep_zone": bool(keep_zone),
            "zone_emb_dim": int(zone_emb_dim),
            "seed": seed,
            "train": {
                "batch_size": int(batch_size),
                "max_epochs": int(max_epochs),
                "patience": int(patience),
                "lr": float(lr),
                "weight_decay": float(weight_decay),
                "grad_clip": grad_clip,
                "device": str(dev),
                "loss": {"name": str(train_loss_name), "huber_delta": float(huber_delta)},
            },
            "model": {
                "channels": list(channels),
                "kernel_size": int(kernel_size),
                "dropout": float(dropout),
            },
            "probabilistic": {
                "use_mc_dropout": bool(use_mc_dropout),
                "mc_runs": int(mc_runs),
                "mc_quantiles": list(mc_quantiles),
            },
        }
    }
    config_snapshot_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")

    logger.info(f"Load: {Path(data_path).resolve()}")
    df = load_raw_data(str(data_path))

    logger.info("Clean + impute")
    df = make_dataset(df, target_col=target_col, zone_col=zone_col, time_col=time_col)

    logger.info(f"Split Track1 by cuts: train_cut={train_cut}, val_cut={val_cut}")
    df_train, df_val, df_test = train_valid_split(
        df,
        time_col=time_col,
        zone_col=zone_col,
        train_cut=train_cut,
        val_cut=val_cut,
    )

    zone_mapping = None
    if keep_zone:
        zone_mapping = _build_zone_mapping_from_train(df_train, zone_col=zone_col)
        (out_dir_p / "zone_mapping.json").write_text(
            json.dumps({"unk": 0, "mapping": zone_mapping}, indent=2),
            encoding="utf-8",
        )
        df_train = _apply_zone_mapping(df_train, zone_col, zone_mapping)
        df_val = _apply_zone_mapping(df_val, zone_col, zone_mapping)
        df_test = _apply_zone_mapping(df_test, zone_col, zone_mapping)

    logger.info("Build sequence windows (split -> window)")
    seq_train = make_seq_features(
        df_train,
        zone_col=zone_col,
        time_col=time_col,
        target_col=target_col,
        lookback=lookback,
        horizon=horizon,
        feature_cols=feature_cols,
        include_target_as_input=include_target_as_input,
        add_missing_mask=add_missing_mask,
    )
    seq_val = make_seq_features(
        df_val,
        zone_col=zone_col,
        time_col=time_col,
        target_col=target_col,
        lookback=lookback,
        horizon=horizon,
        feature_cols=feature_cols,
        include_target_as_input=include_target_as_input,
        add_missing_mask=add_missing_mask,
    )
    seq_test = make_seq_features(
        df_test,
        zone_col=zone_col,
        time_col=time_col,
        target_col=target_col,
        lookback=lookback,
        horizon=horizon,
        feature_cols=feature_cols,
        include_target_as_input=include_target_as_input,
        add_missing_mask=add_missing_mask,
    )

    def _augment_with_mask(seq: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        X = seq["X"]  # [N, L, D]
        y = _to_2d_y(seq["y"])
        z = seq.get("zone", None)
        if add_missing_mask and ("mask_missing" in seq):
            m = seq["mask_missing"]  # [N, L, 1]
            X = np.concatenate([X, m.astype(np.float32)], axis=-1)
        zone = z.astype(np.int64) if z is not None else None
        return X.astype(np.float32), y.astype(np.float32), zone

    Xtr, ytr, ztr = _augment_with_mask(seq_train)
    Xva, yva, zva = _augment_with_mask(seq_val)
    Xte, yte, zte = _augment_with_mask(seq_test)

    train_ds = SeqDataset(Xtr, ytr, ztr if keep_zone else None)
    val_ds = SeqDataset(Xva, yva, zva if keep_zone else None)
    test_ds = SeqDataset(Xte, yte, zte if keep_zone else None)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    input_dim = int(Xtr.shape[-1])
    output_dim = int(ytr.shape[-1])

    base = TCN(
        input_dim=input_dim + (zone_emb_dim if keep_zone else 0),
        output_dim=output_dim,
        channels=channels,
        kernel_size=kernel_size,
        dropout=dropout,
    )

    if keep_zone:
        n_train_zones = len(zone_mapping) if zone_mapping is not None else int(pd.Series(df_train[zone_col]).nunique())
        model: nn.Module = TCNWithZoneEmbedding(
            base_tcn=base,
            n_zones_with_unk=int(n_train_zones + 1),
            emb_dim=int(zone_emb_dim),
        )
    else:
        model = base

    model = model.to(dev)

    logger.info("Train TCN")
    train_summary, best_state = _train_one(
        model,
        train_loader,
        val_loader,
        lr=lr,
        weight_decay=weight_decay,
        max_epochs=max_epochs,
        patience=patience,
        grad_clip=grad_clip,
        device=dev,
        loss_name=train_loss_name,
        huber_delta=huber_delta,
    )
    if best_state:
        model.load_state_dict(best_state)

    logger.info("Predict + evaluate")
    y_val_np, p_val_np = _predict(model, val_loader, dev)
    y_test_np, p_test_np = _predict(model, test_loader, dev)

    prob_metrics: Dict[str, Any] = {}
    if use_mc_dropout:
        yv_mc, p_mean_v, p_std_v, q_v = _predict_mc(
            model, val_loader, dev, mc_runs=int(mc_runs), quantiles=tuple(float(q) for q in mc_quantiles)
        )
        yt_mc, p_mean_t, p_std_t, q_t = _predict_mc(
            model, test_loader, dev, mc_runs=int(mc_runs), quantiles=tuple(float(q) for q in mc_quantiles)
        )

        val_files = _save_mc_outputs(out_dir_p, "val", yv_mc, p_mean_v, p_std_v, q_v)
        test_files = _save_mc_outputs(out_dir_p, "test", yt_mc, p_mean_t, p_std_t, q_t)

        val_sum = _interval_summary(yv_mc, q_v, lo=0.05, hi=0.95)
        test_sum = _interval_summary(yt_mc, q_t, lo=0.05, hi=0.95)

        prob_metrics = {
            "val_prob": {"files": val_files, "summary": val_sum},
            "test_prob": {"files": test_files, "summary": test_sum},
        }

    metrics: Dict[str, Any] = {
        "track": "track1_temporal",
        "split": {"train_cut": train_cut, "val_cut": val_cut},
        "seq": {
            "lookback": int(lookback),
            "horizon": int(horizon),
            "feature_cols": feature_cols,
            "include_target_as_input": bool(include_target_as_input),
            "add_missing_mask": bool(add_missing_mask),
        },
        "keep_zone": bool(keep_zone),
        "loss": {"name": str(train_loss_name), "huber_delta": float(huber_delta)},
        "probabilistic": {"use_mc_dropout": bool(use_mc_dropout), "mc_runs": int(mc_runs), "mc_quantiles": list(mc_quantiles)},
        "model": {
            "tcn": {"channels": list(channels), "kernel_size": int(kernel_size), "dropout": float(dropout)},
            "zone_emb_dim": int(zone_emb_dim) if keep_zone else None,
            "seed": seed,
        },
        "train_summary": train_summary,
        "val": {"rmse": evaluate_rmse(y_val_np, p_val_np), "mae": evaluate_mae(y_val_np, p_val_np), "r2": evaluate_r2(y_val_np, p_val_np), "n_seq": int(len(y_val_np))},
        "test": {"rmse": evaluate_rmse(y_test_np, p_test_np), "mae": evaluate_mae(y_test_np, p_test_np), "r2": evaluate_r2(y_test_np, p_test_np), "n_seq": int(len(y_test_np))},
        "n_rows": {"train": int(len(df_train)), "val": int(len(df_val)), "test": int(len(df_test))},
        "n_seq_samples": {"train": int(len(train_ds)), "val": int(len(val_ds)), "test": int(len(test_ds))},
        "notes": [
            "IMPORTANT: split first, then window (no leakage).",
            "For each split, first 'lookback' points per zone cannot form a full window and are dropped.",
        ],
    }
    if prob_metrics:
        metrics.update(prob_metrics)

    metrics_path = out_dir_p / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    model_path = out_dir_p / "model.pt"
    torch.save({"state_dict": model.state_dict()}, str(model_path))

    logger.info(f"Saved to {out_dir_p}")
    logger.info(f"VAL RMSE={metrics['val']['rmse']:.4f} | TEST RMSE={metrics['test']['rmse']:.4f}")

    return DeterministicTCNResult(
        out_dir=str(out_dir_p),
        model_path=str(model_path),
        metrics_path=str(metrics_path),
        config_snapshot_path=str(config_snapshot_path),
        metrics=metrics,
    )


# -----------------------------
# Track 2: LOFO + inner time val
# -----------------------------
def run_pipeline_lofo(
    *,
    data_path: str | Path,
    out_dir: str | Path,
    held_out_group: int | str,
    val_days: int,
    lookback: int,
    horizon: int = 1,
    feature_cols: Optional[List[str]] = None,
    include_target_as_input: bool = True,
    add_missing_mask: bool = True,
    target_col: str = "target",
    zone_col: str = "zone_id",
    group_col: Optional[str] = None,
    time_col: str = "datetime",
    keep_zone: bool = False,
    zone_emb_dim: int = 8,
    seed: int | None = None,
    min_train: int = 1000,
    # training
    batch_size: int = 256,
    max_epochs: int = 50,
    patience: int = 8,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    grad_clip: Optional[float] = 1.0,
    device: Optional[str] = None,
    train_loss_name: str = "mse",
    huber_delta: float = 1.0,
    # model
    channels: List[int] = [32, 64, 64],
    kernel_size: int = 3,
    dropout: float = 0.2,
    # probabilistic (MC Dropout)
    use_mc_dropout: bool = False,
    mc_runs: int = 50,
    mc_quantiles: Tuple[float, ...] = (0.05, 0.5, 0.95),
) -> DeterministicTCNResult:
    """Run one deterministic TCN pipeline under Track 2 LOFO."""
    out_dir_p = _ensure_dir(out_dir).resolve()
    _set_seed(seed)
    dev = _device_from_str(device)

    config_snapshot_path = out_dir_p / "config_snapshot.json"
    snapshot = {
        "runtime": {
            "data_path": str(Path(data_path).resolve()),
            "out_dir": str(out_dir_p),
            "protocol": "track2_lofo_time_val",
            "split": {
                "held_out_group": held_out_group,
                "val_days": int(val_days),
                "min_train": int(min_train),
                "group_col": (group_col or zone_col),
            },
            "seq": {
                "lookback": int(lookback),
                "horizon": int(horizon),
                "feature_cols": feature_cols,
                "include_target_as_input": bool(include_target_as_input),
                "add_missing_mask": bool(add_missing_mask),
            },
            "cols": {"target_col": target_col, "zone_col": zone_col, "time_col": time_col},
            "keep_zone": bool(keep_zone),
            "zone_emb_dim": int(zone_emb_dim),
            "seed": seed,
            "train": {
                "batch_size": int(batch_size),
                "max_epochs": int(max_epochs),
                "patience": int(patience),
                "lr": float(lr),
                "weight_decay": float(weight_decay),
                "grad_clip": grad_clip,
                "device": str(dev),
                "loss": {"name": str(train_loss_name), "huber_delta": float(huber_delta)},
            },
            "model": {
                "channels": list(channels),
                "kernel_size": int(kernel_size),
                "dropout": float(dropout),
            },
            "probabilistic": {
                "use_mc_dropout": bool(use_mc_dropout),
                "mc_runs": int(mc_runs),
                "mc_quantiles": list(mc_quantiles),
            },
        }
    }
    config_snapshot_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")

    logger.info(f"Load: {Path(data_path).resolve()}")
    df = load_raw_data(str(data_path))

    logger.info("Clean + impute")
    df = make_dataset(df, target_col=target_col, zone_col=zone_col, time_col=time_col)

    logger.info(f"Split Track2 LOFO: held_out_group={held_out_group} | inner_val_days={val_days}")
    df_train_inner, df_val_inner, df_outer_test = lofo_time_val_split(
        df,
        held_out_group=held_out_group,
        group_col=(group_col or zone_col),
        time_col=time_col,
        val_days=val_days,
        min_train=min_train,
    )

    # Simplify Track2: disable zone embedding (held-out site)
    if keep_zone:
        logger.warning("Track2 LOFO: keep_zone=True was requested but is disabled for simplicity.")
    keep_zone = False
    zone_mapping = None

    logger.info("Build sequence windows (split -> window)")
    seq_train = make_seq_features(
        df_train_inner,
        zone_col=zone_col,
        time_col=time_col,
        target_col=target_col,
        lookback=lookback,
        horizon=horizon,
        feature_cols=feature_cols,
        include_target_as_input=include_target_as_input,
        add_missing_mask=add_missing_mask,
    )
    seq_val = make_seq_features(
        df_val_inner,
        zone_col=zone_col,
        time_col=time_col,
        target_col=target_col,
        lookback=lookback,
        horizon=horizon,
        feature_cols=feature_cols,
        include_target_as_input=include_target_as_input,
        add_missing_mask=add_missing_mask,
    )
    seq_test = make_seq_features(
        df_outer_test,
        zone_col=zone_col,
        time_col=time_col,
        target_col=target_col,
        lookback=lookback,
        horizon=horizon,
        feature_cols=feature_cols,
        include_target_as_input=include_target_as_input,
        add_missing_mask=add_missing_mask,
    )

    def _augment_with_mask(seq: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        X = seq["X"]
        y = _to_2d_y(seq["y"])
        z = seq.get("zone", None)
        if add_missing_mask and ("mask_missing" in seq):
            m = seq["mask_missing"]
            X = np.concatenate([X, m.astype(np.float32)], axis=-1)
        zone = z.astype(np.int64) if z is not None else None
        return X.astype(np.float32), y.astype(np.float32), zone

    Xtr, ytr, ztr = _augment_with_mask(seq_train)
    Xva, yva, zva = _augment_with_mask(seq_val)
    Xte, yte, zte = _augment_with_mask(seq_test)

    train_ds = SeqDataset(Xtr, ytr, ztr if keep_zone else None)
    val_ds = SeqDataset(Xva, yva, zva if keep_zone else None)
    test_ds = SeqDataset(Xte, yte, zte if keep_zone else None)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    input_dim = int(Xtr.shape[-1])
    output_dim = int(ytr.shape[-1])

    base = TCN(
        input_dim=input_dim + (zone_emb_dim if keep_zone else 0),
        output_dim=output_dim,
        channels=channels,
        kernel_size=kernel_size,
        dropout=dropout,
    )
    model = base.to(dev)

    logger.info("Train TCN (early stop on inner val)")
    train_summary, best_state = _train_one(
        model,
        train_loader,
        val_loader,
        lr=lr,
        weight_decay=weight_decay,
        max_epochs=max_epochs,
        patience=patience,
        grad_clip=grad_clip,
        device=dev,
        loss_name=train_loss_name,
        huber_delta=huber_delta,
    )
    if best_state:
        model.load_state_dict(best_state)

    logger.info("Predict + evaluate (inner_val + outer_test)")
    y_val_np, p_val_np = _predict(model, val_loader, dev)
    y_test_np, p_test_np = _predict(model, test_loader, dev)

    prob_metrics: Dict[str, Any] = {}
    if use_mc_dropout:
        yv_mc, p_mean_v, p_std_v, q_v = _predict_mc(
            model, val_loader, dev, mc_runs=int(mc_runs), quantiles=tuple(float(q) for q in mc_quantiles)
        )
        yt_mc, p_mean_t, p_std_t, q_t = _predict_mc(
            model, test_loader, dev, mc_runs=int(mc_runs), quantiles=tuple(float(q) for q in mc_quantiles)
        )

        inner_files = _save_mc_outputs(out_dir_p, "inner_val", yv_mc, p_mean_v, p_std_v, q_v)
        outer_files = _save_mc_outputs(out_dir_p, "outer_test", yt_mc, p_mean_t, p_std_t, q_t)

        inner_sum = _interval_summary(yv_mc, q_v, lo=0.05, hi=0.95)
        outer_sum = _interval_summary(yt_mc, q_t, lo=0.05, hi=0.95)

        prob_metrics = {
            "inner_val_prob": {"files": inner_files, "summary": inner_sum},
            "outer_test_prob": {"files": outer_files, "summary": outer_sum},
        }

    metrics: Dict[str, Any] = {
        "track": "track2_lofo",
        "split": {
            "protocol": "lofo_time_val",
            "held_out_group": held_out_group,
            "val_days": int(val_days),
            "min_train": int(min_train),
            "group_col": (group_col or zone_col),
        },
        "seq": {
            "lookback": int(lookback),
            "horizon": int(horizon),
            "feature_cols": feature_cols,
            "include_target_as_input": bool(include_target_as_input),
            "add_missing_mask": bool(add_missing_mask),
        },
        "keep_zone": bool(keep_zone),
        "loss": {"name": str(train_loss_name), "huber_delta": float(huber_delta)},
        "probabilistic": {"use_mc_dropout": bool(use_mc_dropout), "mc_runs": int(mc_runs), "mc_quantiles": list(mc_quantiles)},
        "model": {
            "tcn": {"channels": list(channels), "kernel_size": int(kernel_size), "dropout": float(dropout)},
            "zone_emb_dim": int(zone_emb_dim) if keep_zone else None,
            "seed": seed,
        },
        "train_summary": train_summary,
        "inner_val": {"rmse": evaluate_rmse(y_val_np, p_val_np), "mae": evaluate_mae(y_val_np, p_val_np), "r2": evaluate_r2(y_val_np, p_val_np), "n_seq": int(len(y_val_np))},
        "outer_test": {"rmse": evaluate_rmse(y_test_np, p_test_np), "mae": evaluate_mae(y_test_np, p_test_np), "r2": evaluate_r2(y_test_np, p_test_np), "n_seq": int(len(y_test_np))},
        "n_rows": {"train_inner": int(len(df_train_inner)), "val_inner": int(len(df_val_inner)), "outer_test": int(len(df_outer_test))},
        "n_seq_samples": {"train_inner": int(len(train_ds)), "val_inner": int(len(val_ds)), "outer_test": int(len(test_ds))},
        "notes": [
            "IMPORTANT: split first, then window (no leakage).",
            "Outer_test (held-out site) is cold-start: first `lookback` timestamps cannot be predicted and are dropped.",
        ],
    }
    if prob_metrics:
        metrics.update(prob_metrics)

    metrics_path = out_dir_p / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    model_path = out_dir_p / "model.pt"
    torch.save({"state_dict": model.state_dict()}, str(model_path))

    logger.info(f"Saved to {out_dir_p}")
    logger.info(f"INNER VAL RMSE={metrics['inner_val']['rmse']:.4f} | OUTER TEST RMSE={metrics['outer_test']['rmse']:.4f}")

    return DeterministicTCNResult(
        out_dir=str(out_dir_p),
        model_path=str(model_path),
        metrics_path=str(metrics_path),
        config_snapshot_path=str(config_snapshot_path),
        metrics=metrics,
    )


# -----------------------------
# Dispatcher: Track1 / Track2
# -----------------------------
def run_pipeline_any(
    *,
    data_path: str | Path,
    out_dir: str | Path,
    lookback: int,
    horizon: int = 1,
    feature_cols: Optional[List[str]] = None,
    include_target_as_input: bool = True,
    add_missing_mask: bool = True,
    target_col: str = "target",
    zone_col: str = "zone_id",
    time_col: str = "datetime",
    keep_zone: bool = False,
    zone_emb_dim: int = 8,
    seed: int | None = None,
    # training
    batch_size: int = 256,
    max_epochs: int = 50,
    patience: int = 8,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    grad_clip: Optional[float] = 1.0,
    device: Optional[str] = None,
    # loss (support both names to avoid runner mismatch)
    train_loss_name: str = "mse",
    loss_name: Optional[str] = None,
    huber_delta: float = 1.0,
    # model
    channels: List[int] = [32, 64, 64],
    kernel_size: int = 3,
    dropout: float = 0.2,
    # Track 1 temporal
    train_cut: Optional[str] = None,
    val_cut: Optional[str] = None,
    # Track 2 LOFO
    held_out_group: Optional[Union[int, str]] = None,
    group_col: Optional[str] = None,
    val_days: Optional[int] = None,
    min_train: int = 1000,
    # probabilistic (MC Dropout)
    use_mc_dropout: bool = False,
    mc_runs: int = 50,
    mc_quantiles: Tuple[float, ...] = (0.05, 0.5, 0.95),
) -> DeterministicTCNResult:
    """Convenience dispatcher to run Track1 or Track2 without the runner branching.

    NOTE:
      - `loss_name` is accepted as an alias for compatibility.
      - If both provided, `loss_name` wins.
    """
    effective_loss = (loss_name if loss_name is not None else train_loss_name)

    if train_cut is not None and val_cut is not None:
        return run_pipeline(
            data_path=data_path,
            out_dir=out_dir,
            train_cut=train_cut,
            val_cut=val_cut,
            lookback=lookback,
            horizon=horizon,
            feature_cols=feature_cols,
            include_target_as_input=include_target_as_input,
            add_missing_mask=add_missing_mask,
            target_col=target_col,
            zone_col=zone_col,
            time_col=time_col,
            keep_zone=keep_zone,
            zone_emb_dim=zone_emb_dim,
            seed=seed,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
            lr=lr,
            weight_decay=weight_decay,
            grad_clip=grad_clip,
            device=device,
            train_loss_name=effective_loss,
            huber_delta=huber_delta,
            channels=channels,
            kernel_size=kernel_size,
            dropout=dropout,
            use_mc_dropout=use_mc_dropout,
            mc_runs=int(mc_runs),
            mc_quantiles=mc_quantiles,
        )

    if held_out_group is not None and val_days is not None:
        return run_pipeline_lofo(
            data_path=data_path,
            out_dir=out_dir,
            held_out_group=held_out_group,
            val_days=int(val_days),
            lookback=lookback,
            horizon=horizon,
            feature_cols=feature_cols,
            include_target_as_input=include_target_as_input,
            add_missing_mask=add_missing_mask,
            target_col=target_col,
            zone_col=zone_col,
            group_col=group_col,
            time_col=time_col,
            keep_zone=keep_zone,
            zone_emb_dim=zone_emb_dim,
            seed=seed,
            min_train=min_train,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
            lr=lr,
            weight_decay=weight_decay,
            grad_clip=grad_clip,
            device=device,
            train_loss_name=effective_loss,
            huber_delta=huber_delta,
            channels=channels,
            kernel_size=kernel_size,
            dropout=dropout,
            use_mc_dropout=use_mc_dropout,
            mc_runs=int(mc_runs),
            mc_quantiles=mc_quantiles,
        )

    raise ValueError(
        "run_pipeline_any requires either (train_cut & val_cut) for Track1, "
        "or (held_out_group & val_days) for Track2 LOFO."
    )
