# src/experiments/run_vae.py
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
import yaml

from src.pipelines.utils_data import build_seq_dataloaders
from src.models.representation.vae import SequenceVAE

logger = logging.getLogger(__name__)


# -----------------------------
# helpers
# -----------------------------
def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p.resolve()


def load_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("YAML config must parse to a dict.")
    return cfg


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    keys = sorted({k for r in rows for k in r.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})


def summarize(xs: List[float]) -> Dict[str, float]:
    if not xs:
        return {"mean": math.nan, "std": math.nan, "min": math.nan, "max": math.nan}
    arr = np.asarray(xs, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def safe_float(v: Any) -> Optional[float]:
    try:
        if v is None or v == "":
            return None
        return float(v)
    except Exception:
        return None


def unwrap_x(batch) -> torch.Tensor:
    """SeqDataset returns (x,y) or (x,y,z)."""
    return batch[0]


def vae_loss_parts(recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor):
    """Return (reconstruction_loss, kl_loss)."""
    recon = F.mse_loss(recon_x, x, reduction="mean")
    kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
    return recon, kl


def make_export_loader(loader: DataLoader) -> DataLoader:
    """Rebuild a loader with shuffle=False to keep deterministic sample order."""
    return DataLoader(
        loader.dataset,
        batch_size=loader.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=loader.num_workers,
        pin_memory=getattr(loader, "pin_memory", False),
    )


@torch.no_grad()
def export_mu(vae: SequenceVAE, loader: DataLoader, device: torch.device) -> np.ndarray:
    """Export mu (not sampled z) for every sample in the loader."""
    vae.eval()
    mus: List[np.ndarray] = []
    for batch in loader:
        x = unwrap_x(batch).to(device)
        mu, _logvar = vae.encode(x)  # [B, L, Dz]
        mus.append(mu.detach().cpu().numpy())
    return np.concatenate(mus, axis=0)


def _get_train_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Support both 'training' and 'train' keys."""
    return cfg.get("training") or cfg.get("train") or {}


def _pick_device(device_str: str) -> torch.device:
    """Respect user device choice, but fall back safely if CUDA is not available."""
    device_str = (device_str or "cpu").strip().lower()
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_str)


# -----------------------------
# core: one run
# -----------------------------
def train_and_export_one(
    cfg_base: Dict[str, Any],
    *,
    seed: int,
    protocol: str,
    held_out_group: Optional[Union[int, str]],
    out_dir: Path,
) -> Dict[str, Any]:
    # ---- seeding ----
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    out_dir = ensure_dir(out_dir)
    art_dir = ensure_dir(out_dir / "artifacts")

    tr_cfg = _get_train_cfg(cfg_base)

    # Track2: do NOT use zone info (unknown zone forecast).
    keep_zone = False if protocol == "track2_lofo_time_val" else bool(cfg_base.get("keep_zone", False))

    # ---- build loaders (same as other pipelines) ----
    data_art = build_seq_dataloaders(
        data_path=(cfg_base.get("data", {}) or {}).get("path"),
        protocol=protocol,
        # split
        train_cut=((cfg_base.get("split", {}) or {}).get("train_cut")),
        val_cut=((cfg_base.get("split", {}) or {}).get("val_cut")),
        held_out_group=held_out_group,
        group_col=((cfg_base.get("split", {}) or {}).get("group_col")),
        val_days=((cfg_base.get("split", {}) or {}).get("val_days")),
        min_train=int(((cfg_base.get("split", {}) or {}).get("min_train", 1000))),
        # seq
        lookback=int(((cfg_base.get("seq", {}) or {}).get("lookback", 168))),
        horizon=int(((cfg_base.get("seq", {}) or {}).get("horizon", 1))),
        feature_cols=((cfg_base.get("seq", {}) or {}).get("feature_cols")),
        include_target_as_input=bool(((cfg_base.get("seq", {}) or {}).get("include_target_as_input", True))),
        add_missing_mask=bool(((cfg_base.get("seq", {}) or {}).get("add_missing_mask", True))),
        # zone behavior
        keep_zone=keep_zone,
        # loader
        batch_size=int(tr_cfg.get("batch_size", 256)),
        num_workers=int(tr_cfg.get("num_workers", 0)),
        pin_memory=bool(tr_cfg.get("pin_memory", True)),
    )

    train_loader = data_art.train_loader
    val_loader = data_art.val_loader
    test_loader = data_art.test_loader

    input_dim = int(data_art.input_dim)
    lookback = int(((cfg_base.get("seq", {}) or {}).get("lookback", 168)))

    # ---- device ----
    device = _pick_device(str(tr_cfg.get("device", "cpu")))

    # ---- build VAE ----
    vae_cfg = cfg_base.get("vae", {}) or {}
    if not isinstance(vae_cfg, dict):
        raise ValueError("cfg['vae'] must be a dict of SequenceVAE init kwargs.")
    vae_cfg = dict(vae_cfg)
    vae_cfg.setdefault("input_dim", input_dim)

    vae = SequenceVAE(**vae_cfg).to(device)

    # ---- training params ----
    max_epochs = int(tr_cfg.get("max_epochs", 30))
    patience = int(tr_cfg.get("patience", 6))
    lr = float(tr_cfg.get("lr", 1e-3))
    weight_decay = float(tr_cfg.get("weight_decay", 0.0))
    grad_clip = float(tr_cfg.get("grad_clip", 1.0))
    beta_kl = float(cfg_base.get("vae_beta_kl", 1.0))

    opt = torch.optim.Adam(vae.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    bad = 0

    for epoch in range(1, max_epochs + 1):
        # ---- train ----
        vae.train()
        tr_losses: List[float] = []
        for batch in train_loader:
            x = unwrap_x(batch).to(device)

            recon_x, mu, logvar, _z = vae(x)
            recon, kl = vae_loss_parts(recon_x, x, mu, logvar)
            loss = recon + beta_kl * kl

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip and grad_clip > 0:
                nn.utils.clip_grad_norm_(vae.parameters(), grad_clip)
            opt.step()

            tr_losses.append(float(loss.detach().cpu().item()))

        # ---- val ----
        vae.eval()
        va_losses: List[float] = []
        with torch.no_grad():
            for batch in val_loader:
                x = unwrap_x(batch).to(device)
                recon_x, mu, logvar, _z = vae(x)
                recon, kl = vae_loss_parts(recon_x, x, mu, logvar)
                loss = recon + beta_kl * kl
                va_losses.append(float(loss.detach().cpu().item()))

        tr_mean = float(np.mean(tr_losses)) if tr_losses else float("nan")
        va_mean = float(np.mean(va_losses)) if va_losses else float("nan")

        logger.info(
            f"[VAE seed={seed} heldout={held_out_group}] "
            f"Epoch {epoch:03d} | train_loss={tr_mean:.6f} | val_loss={va_mean:.6f}"
        )

        if va_mean < best_val:
            best_val = va_mean
            best_state = {k: v.detach().cpu().clone() for k, v in vae.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    # restore best
    if best_state is not None:
        vae.load_state_dict(best_state, strict=True)

    # ---- save VAE checkpoint ----
    ckpt_path = art_dir / "vae.pt"
    torch.save({"state_dict": vae.state_dict(), "cfg": vae_cfg, "seed": int(seed)}, ckpt_path)

    # ---- export mu caches ----
    mu_train = export_mu(vae, make_export_loader(train_loader), device=device)
    mu_val = export_mu(vae, make_export_loader(val_loader), device=device)
    mu_test = export_mu(vae, make_export_loader(test_loader), device=device)

    # ---- sanity checks ----
    if mu_train.shape[0] != len(train_loader.dataset):
        raise RuntimeError("Exported train mu count mismatch.")
    if mu_val.shape[0] != len(val_loader.dataset):
        raise RuntimeError("Exported val mu count mismatch.")
    if mu_test.shape[0] != len(test_loader.dataset):
        raise RuntimeError("Exported test mu count mismatch.")
    if mu_train.ndim != 3 or mu_train.shape[1] != lookback:
        raise RuntimeError(f"mu_train shape expected [N,{lookback},Dz], got {mu_train.shape}")

    # ---- save latents ----
    lat_tr = art_dir / "latents_train.npy"
    lat_va = art_dir / "latents_val.npy"
    lat_te = art_dir / "latents_test.npy"

    np.save(lat_tr, mu_train.astype(np.float32, copy=False))
    np.save(lat_va, mu_val.astype(np.float32, copy=False))
    np.save(lat_te, mu_test.astype(np.float32, copy=False))

    meta = {
        "protocol": protocol,
        "seed": int(seed),
        "held_out_group": held_out_group,
        "keep_zone": bool(keep_zone),
        "input_dim_raw": int(input_dim),
        "latent_dim": int(mu_train.shape[-1]),
        "lookback": int(lookback),
        "paths": {
            "vae_ckpt": str(ckpt_path),
            "latents_train": str(lat_tr),
            "latents_val": str(lat_va),
            "latents_test": str(lat_te),
        },
        "best_val_loss": float(best_val),
    }
    (art_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "pipeline": "vae",
        "protocol": protocol,
        "seed": int(seed),
        "held_out_group": held_out_group,
        "out_dir": str(out_dir),
        "artifacts_dir": str(art_dir),
        "best_val_loss": float(best_val),
        "vae_ckpt": str(ckpt_path),
        "latents_train": str(lat_tr),
        "latents_val": str(lat_va),
        "latents_test": str(lat_te),
        "latent_dim": int(mu_train.shape[-1]),
    }


# -----------------------------
# main (batch runner)
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str)
    ap.add_argument("--out_root", required=True, type=str, help="e.g. reports/vae")

    ap.add_argument("--seeds", nargs="+", type=int, default=[42])

    # Track2 only
    ap.add_argument("--held_out_groups", nargs="*", default=None)

    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    cfg_base = load_yaml(args.config)
    protocol = cfg_base.get("protocol", None)
    if not protocol:
        raise KeyError("YAML must contain 'protocol'")

    is_track2 = (protocol == "track2_lofo_time_val")

    out_root = ensure_dir(args.out_root)

    heldouts: List[Union[int, str]] = []
    if is_track2:
        if not args.held_out_groups:
            raise ValueError("Track2 requires --held_out_groups (explicit list).")
        for x in args.held_out_groups:
            try:
                heldouts.append(int(x))
            except Exception:
                heldouts.append(x)

    runs: List[Dict[str, Any]] = []

    for seed in args.seeds:
        if not is_track2:
            run_dir = ensure_dir(Path(out_root) / f"seed_{seed}")
            r = train_and_export_one(cfg_base, seed=seed, protocol=protocol, held_out_group=None, out_dir=run_dir)
            runs.append(r)
        else:
            for g in heldouts:
                run_dir = ensure_dir(Path(out_root) / f"heldout_{g}" / f"seed_{seed}")
                r = train_and_export_one(cfg_base, seed=seed, protocol=protocol, held_out_group=g, out_dir=run_dir)
                runs.append(r)

    val_list = [x for x in (safe_float(r.get("best_val_loss")) for r in runs) if x is not None]
    summary = {
        "pipeline": "vae",
        "protocol": protocol,
        "config": str(Path(args.config).resolve()),
        "seeds": list(args.seeds),
        "held_out_groups": heldouts if is_track2 else None,
        "n_runs": len(runs),
        "best_val_loss": summarize(val_list),
        "runs": runs,
        "artifact_root": str(out_root),
    }

    (Path(out_root) / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(Path(out_root) / "runs.csv", runs)

    logger.info(f"Saved summary -> {Path(out_root) / 'summary.json'}")
    logger.info(f"Saved runs.csv  -> {Path(out_root) / 'runs.csv'}")


if __name__ == "__main__":
    main()
