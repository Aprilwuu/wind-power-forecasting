# src/pipelines/utils_data.py
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.load import load_raw_data
from src.data.make_dataset import make_dataset
from src.data.split import train_valid_split, lofo_time_val_split
from src.features.build_features import make_seq_features


# -----------------------------
# Reproducibility (match TCN)
# -----------------------------
def set_seed(seed: int | None, deterministic: bool = False) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    

def device_from_str(device: Optional[str]) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def to_2d_y(y: np.ndarray) -> np.ndarray:
    if y.ndim == 1:
        return y.reshape(-1, 1)
    return y


# -----------------------------
# Dataset aligned with TCN
# -----------------------------
class SeqDataset(Dataset):
    """
    Returns:
      if zone is None: (X, y)
      else: (X, y, zone_idx)
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, zone: Optional[np.ndarray] = None):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
        self.zone = None if zone is None else torch.from_numpy(zone).long()

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int):
        if self.zone is None:
            return self.X[idx], self.y[idx]
        return self.X[idx], self.y[idx], self.zone[idx]


# -----------------------------
# Zone mapping (match TCN)
# -----------------------------
def build_zone_mapping_from_train(df_train, zone_col: str) -> Dict[Any, int]:
    zones = sorted(df_train[zone_col].unique().tolist())
    return {z: i + 1 for i, z in enumerate(zones)}  # reserve 0 for UNK


def apply_zone_mapping(df, zone_col: str, mapping: Optional[Dict[Any, int]]):
    if mapping is None:
        return df
    df = df.copy()
    df[zone_col] = df[zone_col].map(mapping).fillna(0).astype(int)
    return df


# -----------------------------
# Sequence tensor builder (match TCN)
# -----------------------------
def _augment_with_mask(
    seq: Dict[str, np.ndarray],
    *,
    add_missing_mask: bool,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Exactly the same as the helper inside your tcn_pipeline:
      - X = seq["X"]
      - y -> always [N, H] (use to_2d_y)
      - if add_missing_mask and mask_missing exists: concatenate last dim
    """
    X = seq["X"]  # [N, L, D]
    y = to_2d_y(seq["y"])  # [N, H]
    z = seq.get("zone", None)

    if add_missing_mask and ("mask_missing" in seq):
        m = seq["mask_missing"]  # [N, L, 1] in your make_seq_features
        X = np.concatenate([X, m.astype(np.float32)], axis=-1)

    zone = z.astype(np.int64) if z is not None else None
    return X.astype(np.float32), y.astype(np.float32), zone


# -----------------------------
# Public: build loaders aligned with TCN
# -----------------------------
@dataclass
class DataArtifacts:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    input_dim: int
    output_dim: int
    zone_mapping: Optional[Dict[Any, int]]
    meta: Dict[str, Any]


def _load_latents(
    ckg_dir: Union[str, Path],
    *,
    split: str,
) -> np.ndarray:
    """
    Load cached latent sequences for a split.
    Expected filenames:
      - latents_train.npy
      - latents_val.npy
      - latents_test.npy
    """
    ckg_dir = Path(ckg_dir)
    p = ckg_dir / f"latents_{split}.npy"
    if not p.exists():
        raise FileNotFoundError(f"Missing latent cache file: {p}")
    arr = np.load(p)
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"Latents must be a numpy array, got {type(arr)} from {p}")
    return arr


def _ensure_float32(x: np.ndarray) -> np.ndarray:
    if x.dtype != np.float32:
        x = x.astype(np.float32, copy=False)
    return x


def _check_latents_shape(
    lat: np.ndarray,
    *,
    n_expected: int,
    lookback: int,
    name: str,
) -> None:
    if lat.ndim != 3:
        raise ValueError(f"{name} must be rank-3 [N,L,D], got shape={lat.shape}")
    if lat.shape[0] != n_expected:
        raise ValueError(f"{name} N mismatch: expected {n_expected}, got {lat.shape[0]}")
    if lat.shape[1] != lookback:
        raise ValueError(f"{name} lookback mismatch: expected L={lookback}, got {lat.shape[1]}")
    if lat.shape[2] <= 0:
        raise ValueError(f"{name} latent_dim must be > 0, got {lat.shape[2]}")


def build_seq_dataloaders(
    *,
    data_path: Union[str, Path],
    protocol: str,  # "track1_temporal" or "track2_lofo_time_val"

    # split args
    train_cut: Optional[str] = None,
    val_cut: Optional[str] = None,
    held_out_group: Optional[Union[int, str]] = None,
    group_col: Optional[str] = None,
    val_days: Optional[int] = None,
    min_train: int = 1000,

    # seq args
    lookback: int = 168,
    horizon: int = 1,
    feature_cols: Optional[List[str]] = None,
    include_target_as_input: bool = True,
    add_missing_mask: bool = True,

    # columns
    target_col: str = "target",
    zone_col: str = "zone_id",
    time_col: str = "datetime",

    # zone behavior
    keep_zone: bool = False,

    # loader
    batch_size: int = 256,
    num_workers: int = 0,
    pin_memory: bool = True,

    # -------- NEW (optional) cache params --------
    use_vae: bool = False,
    vae_mode: str = "e2e",  # "e2e" or "cache"
    vae_ckg_dir: Optional[Union[str, Path]] = None,
) -> "DataArtifacts":
    """
    Builds loaders using the SAME logic as your original TCN pipeline:
      load_raw_data -> make_dataset -> split -> make_seq_features -> augment mask -> SeqDataset -> DataLoader

    Optional VAE cache mode:
      If use_vae=True and vae_mode="cache":
        - Load latents_{train,val,test}.npy from vae_ckg_dir
        - Replace Xtr/Xva/Xte with these latents (shape [N, lookback, latent_dim])
        - input_dim becomes latent_dim
    """

    # ----------------------------
    # 1) Load + clean
    # ----------------------------
    df = load_raw_data(str(data_path))
    df = make_dataset(df, target_col=target_col, zone_col=zone_col, time_col=time_col)

    zone_mapping = None

    # ----------------------------
    # 2) Split
    # ----------------------------
    if protocol == "track1_temporal":
        if train_cut is None or val_cut is None:
            raise ValueError("track1_temporal requires train_cut and val_cut.")
        df_train, df_val, df_test = train_valid_split(
            df, time_col=time_col, zone_col=zone_col, train_cut=train_cut, val_cut=val_cut
        )

        # keep_zone mapping is allowed in Track1
        if keep_zone:
            zone_mapping = build_zone_mapping_from_train(df_train, zone_col=zone_col)
            df_train = apply_zone_mapping(df_train, zone_col, zone_mapping)
            df_val = apply_zone_mapping(df_val, zone_col, zone_mapping)
            df_test = apply_zone_mapping(df_test, zone_col, zone_mapping)

    elif protocol == "track2_lofo_time_val":
        if held_out_group is None or val_days is None:
            raise ValueError("track2_lofo_time_val requires held_out_group and val_days.")
        df_train, df_val, df_test = lofo_time_val_split(
            df,
            held_out_group=held_out_group,
            group_col=(group_col or zone_col),
            time_col=time_col,
            val_days=int(val_days),
            min_train=int(min_train),
        )

        # IMPORTANT: unknown-zone generalization -> disable keep_zone for LOFO
        keep_zone = False
        zone_mapping = None

    else:
        raise ValueError(f"Unknown protocol: {protocol}")

    # ----------------------------
    # 3) Window features
    # ----------------------------
    def _make(df_part):
        return make_seq_features(
            df_part,
            zone_col=zone_col,
            time_col=time_col,
            target_col=target_col,
            lookback=int(lookback),
            horizon=int(horizon),
            feature_cols=feature_cols,
            include_target_as_input=bool(include_target_as_input),
            add_missing_mask=bool(add_missing_mask),
        )

    seq_train = _make(df_train)
    seq_val = _make(df_val)
    seq_test = _make(df_test)

    Xtr, ytr, ztr = _augment_with_mask(seq_train, add_missing_mask=bool(add_missing_mask))
    Xva, yva, zva = _augment_with_mask(seq_val, add_missing_mask=bool(add_missing_mask))
    Xte, yte, zte = _augment_with_mask(seq_test, add_missing_mask=bool(add_missing_mask))

    # Ensure float32 for torch conversion
    Xtr = _ensure_float32(Xtr)
    Xva = _ensure_float32(Xva)
    Xte = _ensure_float32(Xte)

    # ----------------------------
    # 4) Optional: replace X with cached latents
    # ----------------------------
    if bool(use_vae) and str(vae_mode).lower().strip() == "cache":
        if vae_ckg_dir is None:
            raise ValueError("vae_mode='cache' requires vae_ckg_dir pointing to the cache folder.")

        lat_tr = _load_latents(vae_ckg_dir, split="train")
        lat_va = _load_latents(vae_ckg_dir, split="val")
        lat_te = _load_latents(vae_ckg_dir, split="test")

        # Sanity checks to guarantee alignment
        _check_latents_shape(lat_tr, n_expected=int(ytr.shape[0]), lookback=int(lookback), name="latents_train")
        _check_latents_shape(lat_va, n_expected=int(yva.shape[0]), lookback=int(lookback), name="latents_val")
        _check_latents_shape(lat_te, n_expected=int(yte.shape[0]), lookback=int(lookback), name="latents_test")

        # Force float32 (torch-friendly)
        lat_tr = _ensure_float32(lat_tr)
        lat_va = _ensure_float32(lat_va)
        lat_te = _ensure_float32(lat_te)

        # Replace X*
        Xtr, Xva, Xte = lat_tr, lat_va, lat_te

        # NOTE: z arrays remain unchanged (if Track1 keep_zone=True, model may still use z)
        # For Track2 keep_zone is forced False above.

    # ----------------------------
    # 5) Build datasets + loaders
    # ----------------------------
    train_ds = SeqDataset(Xtr, ytr, ztr if keep_zone else None)
    val_ds = SeqDataset(Xva, yva, zva if keep_zone else None)
    test_ds = SeqDataset(Xte, yte, zte if keep_zone else None)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(batch_size),
        shuffle=True,
        drop_last=False,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(batch_size),
        shuffle=False,
        drop_last=False,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=int(batch_size),
        shuffle=False,
        drop_last=False,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
    )

    input_dim = int(Xtr.shape[-1])
    output_dim = int(ytr.shape[-1])  # y is always [N, H]

    # ----------------------------
    # 6) Meta
    # ----------------------------
    meta = {
        "protocol": protocol,
        "n_rows": {"train": int(len(df_train)), "val": int(len(df_val)), "test": int(len(df_test))},
        "n_seq_samples": {"train": int(len(train_ds)), "val": int(len(val_ds)), "test": int(len(test_ds))},
        "seq": {
            "lookback": int(lookback),
            "horizon": int(horizon),
            "feature_cols": feature_cols,
            "include_target_as_input": bool(include_target_as_input),
            "add_missing_mask": bool(add_missing_mask),
        },
        "cols": {"target_col": target_col, "zone_col": zone_col, "time_col": time_col},
        "keep_zone": bool(keep_zone),
        "zone_mapping": zone_mapping,
        "vae": {
            "use_vae": bool(use_vae),
            "vae_mode": str(vae_mode).lower().strip(),
            "vae_ckg_dir": str(vae_ckg_dir) if vae_ckg_dir is not None else None,
            "input_dim_raw_or_latent": int(input_dim),
        },
    }

    return DataArtifacts(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        input_dim=input_dim,
        output_dim=output_dim,
        zone_mapping=zone_mapping,
        meta=meta,
    )
