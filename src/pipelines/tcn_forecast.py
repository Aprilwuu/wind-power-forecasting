# src/pipelines/tcn_forecast.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.models.backbones.tcn import TCN
from src.models.probabilistic.mc_dropout.mc_dropout import mc_dropout_predict
from src.pipelines.forecast_base import ForecastBasePipeline
from src.pipelines.utils_data import DataArtifacts


# -----------------------------
# Helpers (same behavior as old tcn_pipeline.py)
# -----------------------------
def _to_2d_y(y: np.ndarray) -> np.ndarray:
    if y.ndim == 1:
        return y.reshape(-1, 1)
    return y


def _interval_summary(
    y_true: np.ndarray,
    q_out: Dict[float, np.ndarray],
    lo: float = 0.05,
    hi: float = 0.95,
) -> Dict[str, float]:
    y = _to_2d_y(np.asarray(y_true))
    ql = _to_2d_y(np.asarray(q_out[float(lo)]))
    qu = _to_2d_y(np.asarray(q_out[float(hi)]))

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
    CSV columns (horizon==1): sample_idx, y_true, mean, std, q05/q50/q95..., y_pred
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    y = _to_2d_y(np.asarray(y_true))
    mean = _to_2d_y(np.asarray(p_mean))
    std = _to_2d_y(np.asarray(p_std))

    # If horizon > 1, save NPZ only
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


class TCNWithZoneEmbedding(nn.Module):
    """
    Reserve 0 as UNK zone, train zones mapped to 1..K.
    Concatenate zone embedding to each timestep feature.
    """

    def __init__(self, base_tcn: TCN, n_zones_with_unk: int, emb_dim: int):
        super().__init__()
        self.base_tcn = base_tcn
        self.emb = nn.Embedding(int(n_zones_with_unk), int(emb_dim))

    def forward(self, x: torch.Tensor, zone_idx: torch.Tensor) -> torch.Tensor:
        z = self.emb(zone_idx)  # [B, emb_dim]
        z = z.unsqueeze(1).expand(-1, x.size(1), -1)  # [B, L, emb_dim]
        x_aug = torch.cat([x, z], dim=-1)  # [B, L, D+emb_dim]
        return self.base_tcn(x_aug)


# -----------------------------
# Pipeline
# -----------------------------
class TCNForecastPipeline(ForecastBasePipeline):
    """
    TCN pipeline implemented as a subclass of ForecastBasePipeline.
    Deterministic part is handled by ForecastBasePipeline.run().
    We override run() to append MC Dropout outputs + PICP/MPIW.
    Naming is unified as val/test for both Track1/Track2.
    """

    def build_model(self, input_dim: int, output_dim: int, data_art: DataArtifacts) -> nn.Module:
        cfg = self.cfg
        keep_zone = bool(data_art.meta.get("keep_zone", False))
        zone_emb_dim = int(cfg.get("zone_emb_dim", 8))

        channels = cfg.get("channels", [32, 64, 64])
        kernel_size = int(cfg.get("kernel_size", 3))
        dropout = float(cfg.get("dropout", 0.2))

        base = TCN(
            input_dim=int(input_dim + (zone_emb_dim if keep_zone else 0)),
            output_dim=int(output_dim),
            channels=list(channels),
            kernel_size=int(kernel_size),
            dropout=float(dropout),
        )

        if not keep_zone:
            return base

        zone_mapping = data_art.zone_mapping
        if zone_mapping is None:
            raise ValueError("keep_zone=True but data_art.zone_mapping is None. Check utils_data/build_seq_dataloaders.")
        n_train_zones = int(len(zone_mapping))
        return TCNWithZoneEmbedding(base_tcn=base, n_zones_with_unk=n_train_zones + 1, emb_dim=zone_emb_dim)

    def run(self) -> Dict[str, Any]:
        # 1) deterministic run
        out = super().run()

        cfg = self.cfg
        if not bool(cfg.get("use_mc_dropout", False)):
            return out

        mc_runs = int(cfg.get("mc_runs", 50))
        quantiles = tuple(float(q) for q in cfg.get("mc_quantiles", (0.05, 0.5, 0.95)))

        # 2) rebuild loaders
        data_art = self._build_data(cfg)
        val_loader, test_loader = data_art.val_loader, data_art.test_loader

        # 3) load trained model
        model = self.build_model(data_art.input_dim, data_art.output_dim, data_art).to(self.device)
        model_path = Path(self.out_dir) / "model.pt"
        ckpt = torch.load(str(model_path), map_location="cpu")
        model.load_state_dict(ckpt["state_dict"])

        # 4) MC predictions
        yv_mc, p_mean_v, p_std_v, q_v = self._predict_mc(model, val_loader, mc_runs=mc_runs, quantiles=quantiles)
        yt_mc, p_mean_t, p_std_t, q_t = self._predict_mc(model, test_loader, mc_runs=mc_runs, quantiles=quantiles)

        val_files = _save_mc_outputs(Path(self.out_dir), "val", yv_mc, p_mean_v, p_std_v, q_v)
        test_files = _save_mc_outputs(Path(self.out_dir), "test", yt_mc, p_mean_t, p_std_t, q_t)

        val_sum = _interval_summary(yv_mc, q_v, lo=0.05, hi=0.95)
        test_sum = _interval_summary(yt_mc, q_t, lo=0.05, hi=0.95)

        # 5) write back metrics.json (unified val/test)
        metrics_path = Path(self.out_dir) / "metrics.json"
        metrics = out["metrics"]

        metrics["probabilistic"] = {
            "use_mc_dropout": True,
            "mc_runs": int(mc_runs),
            "mc_quantiles": list(quantiles),
        }
        metrics["val_prob"] = {"files": val_files, "summary": val_sum}
        metrics["test_prob"] = {"files": test_files, "summary": test_sum}

        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        out["metrics"] = metrics
        # NOTE: these are numpy arrays (not JSON-serializable), but ok for in-process use
        out["predictions"]["val_mc"] = {"y_true": yv_mc, "mean": p_mean_v, "std": p_std_v, "q": q_v}
        out["predictions"]["test_mc"] = {"y_true": yt_mc, "mean": p_mean_t, "std": p_std_t, "q": q_t}
        return out

    @torch.no_grad()
    def _predict_mc(
        self,
        model: nn.Module,
        loader,
        *,
        mc_runs: int,
        quantiles: Tuple[float, ...],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[float, np.ndarray]]:
        """
        Returns:
            y_true: [N,H]
            p_mean: [N,H]
            p_std:  [N,H]
            q_out:  {q: [N,H]}
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

            x = x.to(self.device)
            if z is not None:
                z = z.to(self.device)

            if z is None:
                p_mean, p_std, q_dict, _ = mc_dropout_predict(
                    model=model,
                    xb=x,
                    device=self.device,
                    mc_runs=int(mc_runs),
                    quantiles=quantiles,
                    squeeze_single_horizon=False,
                    move_to_device=False,
                )
            else:
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
                    device=self.device,
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
