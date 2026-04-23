# src/pipelines/forecast_base.py
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.metrics.deterministic import evaluate_rmse, evaluate_mae, evaluate_r2
from .base_pipeline import BasePipeline
from .utils_data import (
    DataArtifacts,
    build_seq_dataloaders,
    device_from_str,
    set_seed,
)

logger = logging.getLogger(__name__)


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _get_nested(d: Dict[str, Any], path: str, default=None):
    cur: Any = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _get_any(*candidates, default=None):
    for v in candidates:
        if v is not None:
            return v
    return default


# -------------------------
# Probabilistic metrics (numpy)
# -------------------------
def pinball_loss_np(y: np.ndarray, q: np.ndarray, taus: List[float]) -> float:
    """
    y: [N,1] or [N]
    q: [N,K] or [N,1,K]
    """
    y = y.reshape(-1, 1) if y.ndim == 1 else y
    if q.ndim == 3:
        q = q[:, 0, :]
    if q.ndim != 2:
        raise ValueError(f"pinball_loss_np expects q as [N,K] or [N,1,K], got {q.shape}")

    taus_arr = np.asarray(taus, dtype=np.float32).reshape(1, -1)  # [1,K]
    diff = y - q  # [N,K]
    loss = np.maximum(taus_arr * diff, (taus_arr - 1.0) * diff)
    return float(np.mean(loss))


def interval_picp_np(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    y = y.reshape(-1, 1) if y.ndim == 1 else y
    lo = lo.reshape(-1, 1) if lo.ndim == 1 else lo
    hi = hi.reshape(-1, 1) if hi.ndim == 1 else hi
    inside = (y >= lo) & (y <= hi)
    return float(np.mean(inside))


def interval_mpiw_np(lo: np.ndarray, hi: np.ndarray) -> float:
    lo = lo.reshape(-1, 1) if lo.ndim == 1 else lo
    hi = hi.reshape(-1, 1) if hi.ndim == 1 else hi
    return float(np.mean(hi - lo))


def save_interval_npz(
    path: Path,
    y_true: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    q50: Optional[np.ndarray] = None,
) -> None:
    """
    Save interval predictions for later conformal/postprocess/plot usage.

    By default saves:
      - y_true
      - q05
      - q95

    Optionally also saves:
      - q50

    This is especially useful for QR models, where the central line should be
    the model-predicted median instead of the midpoint of [q05, q95].
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    y_true = y_true.reshape(-1, 1) if y_true.ndim == 1 else y_true
    lo = lo.reshape(-1, 1) if lo.ndim == 1 else lo
    hi = hi.reshape(-1, 1) if hi.ndim == 1 else hi

    arrays = {
        "y_true": y_true.astype(np.float32),
        "q05": lo.astype(np.float32),
        "q95": hi.astype(np.float32),
    }

    if q50 is not None:
        q50 = q50.reshape(-1, 1) if q50.ndim == 1 else q50
        arrays["q50"] = q50.astype(np.float32)

    np.savez_compressed(path, **arrays)


# -------------------------
# Loss builder (torch)
# -------------------------
def build_loss(name: str = "mse", huber_delta: float = 1.0) -> nn.Module:
    name = (name or "mse").lower().strip()

    if name in {"huber", "huberloss"}:
        try:
            return nn.HuberLoss(delta=float(huber_delta))
        except Exception:
            try:
                return nn.SmoothL1Loss(beta=float(huber_delta))
            except Exception:
                return nn.SmoothL1Loss()

    if name in {"smoothl1", "smooth_l1", "smoothl1loss"}:
        try:
            return nn.SmoothL1Loss(beta=float(huber_delta))
        except Exception:
            return nn.SmoothL1Loss()

    return nn.MSELoss()


@dataclass
class ForecastOutputs:
    y_true: np.ndarray  # [N, H]
    y_pred: np.ndarray  # [N, H]


class ForecastBasePipeline(BasePipeline):
    """
    Generic forecast pipeline:
      - early stop on val RMSE (point prediction)
      - split->window done in build_seq_dataloaders()
      - supports (x,y) or (x,y,zone)

    Subclasses must implement:
      - build_model()

    Subclasses can override:
      - model_forward()
      - compute_loss()
      - postprocess_pred()       -> point pred [B,H]
      - postprocess_interval()   -> (lo,hi) each [B,1]  (optional)
      - postprocess_quantiles()  -> q [B,K] or [B,1,K]  (optional)
    """

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(cfg)

        # out_dir must exist in cfg (runner should inject it).
        self.out_dir = ensure_dir(cfg["out_dir"]).resolve()

        dev = _get_any(cfg.get("device"), _get_nested(cfg, "training.device"), default=None)
        self.device = device_from_str(dev)

    # ---------- hooks ----------
    def build_model(self, input_dim: int, output_dim: int, data_art: DataArtifacts) -> nn.Module:
        raise NotImplementedError

    def model_forward(self, model: nn.Module, batch) -> torch.Tensor:
        if len(batch) == 2:
            x, _y = batch
            x = x.to(self.device)
            return model(x)

        x, _y, z = batch
        x = x.to(self.device)
        z = z.to(self.device).long()

        try:
            return model(x, z)
        except TypeError:
            return model(x)

    def compute_loss(self, loss_fn: nn.Module, pred_raw, y: torch.Tensor) -> torch.Tensor:
        return loss_fn(pred_raw, y)

    def postprocess_pred(self, pred_raw) -> torch.Tensor:
        pred = pred_raw
        if isinstance(pred, (tuple, list)):
            raise TypeError("postprocess_pred got tuple/list pred_raw; override in subclass.")
        if pred.dim() == 1:
            pred = pred.unsqueeze(-1)
        return pred

    def postprocess_interval(
        self, pred_raw, coverage: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def postprocess_quantiles(self, pred_raw) -> torch.Tensor:
        raise NotImplementedError

    # ---------- main ----------
    def run(self) -> Dict[str, Any]:
        cfg = self.cfg
        set_seed(cfg.get("seed", None))

        # snapshot config
        (self.out_dir / "config_snapshot.json").write_text(
            json.dumps({"runtime": cfg}, indent=2), encoding="utf-8"
        )

        # data
        data_art = self._build_data(cfg)
        train_loader, val_loader, test_loader = data_art.train_loader, data_art.val_loader, data_art.test_loader
        input_dim, output_dim = data_art.input_dim, data_art.output_dim

        # model
        model = self.build_model(input_dim, output_dim, data_art).to(self.device)

        # training params (top-level or training.*)
        lr = float(_get_any(cfg.get("lr"), _get_nested(cfg, "training.lr"), default=1e-3))
        weight_decay = float(_get_any(cfg.get("weight_decay"), _get_nested(cfg, "training.weight_decay"), default=0.0))
        max_epochs = int(_get_any(cfg.get("max_epochs"), _get_nested(cfg, "training.max_epochs"), default=50))
        patience = int(_get_any(cfg.get("patience"), _get_nested(cfg, "training.patience"), default=8))
        grad_clip = _get_any(cfg.get("grad_clip"), _get_nested(cfg, "training.grad_clip"), default=1.0)
        grad_clip = float(grad_clip) if grad_clip is not None else None
        val_every = int(_get_any(cfg.get("val_every"), _get_nested(cfg, "training.val_every"), default=1))

        loss_name = str(_get_any(cfg.get("train_loss_name"), cfg.get("loss_name"), default="mse"))
        huber_delta = float(_get_any(cfg.get("huber_delta"), _get_nested(cfg, "training.huber_delta"), default=1.0))
        loss_fn = build_loss(loss_name, huber_delta)

        optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        # train (early stop on val RMSE of point preds)
        train_summary, best_state = self._train_one(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optim=optim,
            loss_fn=loss_fn,
            max_epochs=max_epochs,
            patience=patience,
            grad_clip=grad_clip,
            val_every=val_every,
        )
        if best_state:
            model.load_state_dict(best_state)

        # point prediction eval
        y_val, p_val = self._predict_point(model, val_loader)
        y_test, p_test = self._predict_point(model, test_loader)

        metrics: Dict[str, Any] = {
            "track": cfg.get("protocol", cfg.get("track", None)),
            "split": cfg.get("split", {}),
            "seq": data_art.meta.get("seq", {}),
            "keep_zone": data_art.meta.get("keep_zone", False),
            "loss": {"name": str(loss_name), "huber_delta": float(huber_delta)},
            "train_summary": train_summary,
            "val": {
                "rmse": evaluate_rmse(y_val, p_val),
                "mae": evaluate_mae(y_val, p_val),
                "r2": evaluate_r2(y_val, p_val),
                "n_seq": int(len(y_val)),
            },
            "test": {
                "rmse": evaluate_rmse(y_test, p_test),
                "mae": evaluate_mae(y_test, p_test),
                "r2": evaluate_r2(y_test, p_test),
                "n_seq": int(len(y_test)),
            },
            "n_rows": data_art.meta.get("n_rows", {}),
            "n_seq_samples": data_art.meta.get("n_seq_samples", {}),
            "notes": [
                "IMPORTANT: split first, then window (no leakage).",
                "Sequence tensors are built by make_seq_features().",
            ],
        }

        # ---------- Probabilistic: pinball ----------
        qs = _get_nested(cfg, "probabilistic.quantiles", None)
        if qs is not None:
            try:
                q_val = self._predict_quantiles(model, val_loader)
                q_test = self._predict_quantiles(model, test_loader)

                metrics.setdefault("val_prob", {})
                metrics.setdefault("test_prob", {})

                metrics["val_prob"]["quantiles"] = list(qs)
                metrics["test_prob"]["quantiles"] = list(qs)
                metrics["val_prob"]["pinball"] = pinball_loss_np(y_val, q_val, list(qs))
                metrics["test_prob"]["pinball"] = pinball_loss_np(y_test, q_test, list(qs))
            except NotImplementedError:
                logger.warning("probabilistic.quantiles is set but postprocess_quantiles() is not implemented.")
            except Exception as e:
                logger.exception(f"Pinball evaluation failed: {e}")

        # ---------- Interval metrics: PICP + MPIW ----------
        if bool(cfg.get("compute_interval", False)):
            coverage = float(cfg.get("interval_coverage", 0.9))
            try:
                lo_val, hi_val = self._predict_interval(model, val_loader, coverage=coverage)
                lo_test, hi_test = self._predict_interval(model, test_loader, coverage=coverage)

                # NEW: also save central prediction
                # For QR this is q0.5 if the subclass implements postprocess_pred that way.
                _, q50_val = self._predict_point(model, val_loader)
                _, q50_test = self._predict_point(model, test_loader)

                metrics.setdefault("val_prob", {})
                metrics.setdefault("test_prob", {})

                metrics["val_prob"]["coverage_target"] = coverage
                metrics["val_prob"]["picp"] = interval_picp_np(y_val, lo_val, hi_val)
                metrics["val_prob"]["mpiw"] = interval_mpiw_np(lo_val, hi_val)

                metrics["test_prob"]["coverage_target"] = coverage
                metrics["test_prob"]["picp"] = interval_picp_np(y_test, lo_test, hi_test)
                metrics["test_prob"]["mpiw"] = interval_mpiw_np(lo_test, hi_test)
            except NotImplementedError:
                logger.warning("compute_interval=True but postprocess_interval() not implemented.")
            except Exception as e:
                logger.exception(f"Interval evaluation failed: {e}")
                lo_val, hi_val, lo_test, hi_test = None, None, None, None
                q50_val, q50_test = None, None

            # === save NPZ for conformal calibration ===
            if lo_val is not None and hi_val is not None and lo_test is not None and hi_test is not None:
                protocol = str(cfg.get("protocol", ""))
                if protocol == "track2_lofo_time_val":
                    val_name = "preds_outer_val_qr.npz"
                    test_name = "preds_outer_test_transformer_qr.npz"
                else:
                    val_name = "preds_val_qr.npz"
                    test_name = "preds_test_qr.npz"

                save_interval_npz(self.out_dir / val_name, y_val, lo_val, hi_val, q50=q50_val)
                save_interval_npz(self.out_dir / test_name, y_test, lo_test, hi_test, q50=q50_test)

        # save once
        (self.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        torch.save({"state_dict": model.state_dict()}, str(self.out_dir / "model.pt"))

        return {"metrics": metrics, "config": cfg}

    # ---------- data builder ----------
    def _build_data(self, cfg: Dict[str, Any]) -> DataArtifacts:
        protocol = cfg["protocol"]

        data_path = _get_any(cfg.get("data_path"), _get_nested(cfg, "data.path"))
        if data_path is None:
            raise KeyError("Missing data_path (cfg['data_path'] or data.path).")

        lookback = _get_any(cfg.get("lookback"), _get_nested(cfg, "seq.lookback"))
        if lookback is None:
            raise KeyError("Missing lookback (cfg['lookback'] or seq.lookback).")
        lookback = int(lookback)

        horizon = int(_get_any(cfg.get("horizon"), _get_nested(cfg, "seq.horizon"), default=1))
        feature_cols = _get_any(cfg.get("feature_cols"), _get_nested(cfg, "seq.feature_cols"), default=None)
        include_target_as_input = bool(
            _get_any(cfg.get("include_target_as_input"), _get_nested(cfg, "seq.include_target_as_input"), default=True)
        )
        add_missing_mask = bool(
            _get_any(cfg.get("add_missing_mask"), _get_nested(cfg, "seq.add_missing_mask"), default=True)
        )

        target_col = _get_any(cfg.get("target_col"), _get_nested(cfg, "data.target_col"), default="target")
        zone_col = _get_any(cfg.get("zone_col"), _get_nested(cfg, "data.zone_col"), default="zone_id")
        time_col = _get_any(cfg.get("time_col"), _get_nested(cfg, "data.time_col"), default="datetime")

        keep_zone = bool(_get_any(cfg.get("keep_zone"), _get_nested(cfg, "keep_zone"), default=False))

        batch_size = int(_get_any(cfg.get("batch_size"), _get_nested(cfg, "training.batch_size"), default=256))
        num_workers = int(_get_any(cfg.get("num_workers"), _get_nested(cfg, "training.num_workers"), default=0))
        pin_memory = bool(_get_any(cfg.get("pin_memory"), _get_nested(cfg, "training.pin_memory"), default=True))

        if protocol == "track1_temporal":
            train_cut = _get_any(cfg.get("train_cut"), _get_nested(cfg, "split.train_cut"))
            val_cut = _get_any(cfg.get("val_cut"), _get_nested(cfg, "split.val_cut"))
            if train_cut is None or val_cut is None:
                raise KeyError("Track1 requires split.train_cut and split.val_cut.")

            return build_seq_dataloaders(
                data_path=data_path,
                protocol=protocol,
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
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )

        if protocol == "track2_lofo_time_val":
            held_out_group = _get_any(cfg.get("held_out_group"), _get_nested(cfg, "split.held_out_group"))
            val_days = _get_any(cfg.get("val_days"), _get_nested(cfg, "split.val_days"))
            if held_out_group is None or val_days is None:
                raise KeyError("Track2 requires split.held_out_group and split.val_days.")

            min_train = int(_get_any(cfg.get("min_train"), _get_nested(cfg, "split.min_train"), default=1000))
            group_col = _get_any(cfg.get("group_col"), _get_nested(cfg, "split.group_col"), default=None)

            return build_seq_dataloaders(
                data_path=data_path,
                protocol=protocol,
                held_out_group=held_out_group,
                group_col=group_col,
                val_days=int(val_days),
                min_train=min_train,
                lookback=lookback,
                horizon=horizon,
                feature_cols=feature_cols,
                include_target_as_input=include_target_as_input,
                add_missing_mask=add_missing_mask,
                target_col=target_col,
                zone_col=zone_col,
                time_col=time_col,
                keep_zone=keep_zone,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )

        raise ValueError(f"Unknown protocol: {protocol}")

    # ---------- train ----------
    def _train_one(
        self,
        *,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optim: torch.optim.Optimizer,
        loss_fn: nn.Module,
        max_epochs: int,
        patience: int,
        grad_clip: Optional[float],
        val_every: int = 1,
    ) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
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
                    _x, y = batch
                else:
                    _x, y, _z = batch

                y = y.to(self.device)
                if y.dim() == 1:
                    y = y.unsqueeze(-1)

                pred_raw = self.model_forward(model, batch)
                loss = self.compute_loss(loss_fn, pred_raw, y)
                loss.backward()

                if grad_clip is not None and float(grad_clip) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))

                optim.step()
                train_losses.append(float(loss.item()))

            train_loss = float(np.mean(train_losses)) if train_losses else math.nan
            history["train_loss"].append(train_loss)

            did_val = (epoch % val_every == 0) or (epoch == max_epochs)
            if did_val:
                y_val_np, p_val_np = self._predict_point(model, val_loader)
                val_rmse = evaluate_rmse(y_val_np, p_val_np)

                history["val_rmse"].append(float(val_rmse))
                logger.info(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_rmse={val_rmse:.6f}")

                if val_rmse < best_val - 1e-9:
                    best_val = float(val_rmse)
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    bad = 0
                else:
                    bad += 1
                    if bad >= patience:
                        logger.info(f"Early stopping at epoch {epoch} (patience={patience}).")
                        break
            else:
                logger.info(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_rmse=SKIP (val_every={val_every})")

        summary = {
            "best_val_rmse": float(best_val),
            "epochs_ran": int(len(history["train_loss"])),
            "history": history,
            "val_every": int(val_every),
        }
        return summary, best_state

    # ---------- prediction helpers ----------
    @torch.no_grad()
    def _predict_point(self, model: nn.Module, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        model.eval()
        ys, ps = [], []

        for batch in loader:
            if len(batch) == 2:
                _x, y = batch
            else:
                _x, y, _z = batch

            pred_raw = self.model_forward(model, batch)
            pred_point = self.postprocess_pred(pred_raw).detach().cpu().numpy()

            y_np = y.detach().cpu().numpy()
            y_np = y_np if y_np.ndim == 2 else y_np.reshape(-1, 1)

            ys.append(y_np)
            ps.append(pred_point)

        return np.concatenate(ys, axis=0), np.concatenate(ps, axis=0)

    @torch.no_grad()
    def _predict_interval(
        self, model: nn.Module, loader: DataLoader, *, coverage: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        model.eval()
        los, his = [], []

        for batch in loader:
            pred_raw = self.model_forward(model, batch)
            lo_t, hi_t = self.postprocess_interval(pred_raw, coverage=coverage)

            lo = lo_t.detach().cpu().numpy()
            hi = hi_t.detach().cpu().numpy()
            lo = lo if lo.ndim == 2 else lo.reshape(-1, 1)
            hi = hi if hi.ndim == 2 else hi.reshape(-1, 1)

            los.append(lo)
            his.append(hi)

        return np.concatenate(los, axis=0), np.concatenate(his, axis=0)

    @torch.no_grad()
    def _predict_quantiles(self, model: nn.Module, loader: DataLoader) -> np.ndarray:
        model.eval()
        qs = []
        for batch in loader:
            pred_raw = self.model_forward(model, batch)
            q_t = self.postprocess_quantiles(pred_raw)  # must be implemented by QR pipeline
            qs.append(q_t.detach().cpu().numpy())
        return np.concatenate(qs, axis=0)