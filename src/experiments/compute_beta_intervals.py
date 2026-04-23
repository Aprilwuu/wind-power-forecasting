from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import torch
import yaml

from src.pipelines.beta_transformer_pipeline import TransformerBetaPipeline

logger = logging.getLogger(__name__)


def load_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("YAML config must parse to a dict.")
    return cfg


def load_state_strict(model: torch.nn.Module, ckpt_path: Path) -> None:
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=True)


@torch.no_grad()
def predict_alpha_beta(
    pipe: TransformerBetaPipeline,
    model: torch.nn.Module,
    loader,
) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor]:
    model.eval()
    ys: List[np.ndarray] = []
    alphas: List[torch.Tensor] = []
    betas: List[torch.Tensor] = []

    for batch in loader:
        if len(batch) == 2:
            _x, y = batch
        else:
            _x, y, _z = batch

        alpha, beta = pipe.model_forward(model, batch)
        alphas.append(alpha.detach().cpu())
        betas.append(beta.detach().cpu())

        y_np = y.numpy()
        y_np = y_np if y_np.ndim == 2 else y_np.reshape(-1, 1)
        ys.append(y_np)

    y_true = np.concatenate(ys, axis=0)
    alpha = torch.cat(alphas, dim=0)
    beta = torch.cat(betas, dim=0)
    return y_true, alpha, beta


def picp_mpiw(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> Tuple[float, float]:
    cover = (y_true >= lower) & (y_true <= upper)
    picp = float(cover.mean())
    mpiw = float((upper - lower).mean())
    return picp, mpiw


def patch_metrics_json(
    metrics_path: Path,
    *,
    coverage: float,
    val_picp: Optional[float],
    val_mpiw: Optional[float],
    test_picp: Optional[float],
    test_mpiw: Optional[float],
) -> None:
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    else:
        metrics = {"val": {}, "test": {}}

    metrics.setdefault("val", {})
    metrics.setdefault("test", {})

    metrics["val"]["coverage"] = coverage
    metrics["val"]["picp"] = val_picp
    metrics["val"]["mpiw"] = val_mpiw

    metrics["test"]["coverage"] = coverage
    metrics["test"]["picp"] = test_picp
    metrics["test"]["mpiw"] = test_mpiw

    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str)
    ap.add_argument("--run_dir", required=True, type=str, help="path to seed_xxx dir that contains model.pt")
    ap.add_argument("--coverage", type=float, default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    cfg = load_yaml(args.config)
    run_dir = Path(args.run_dir).resolve()

    cfg = dict(cfg)
    cfg["out_dir"] = str(run_dir)  # required by ForecastBasePipeline
    
    model_path = run_dir / "model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"model.pt not found: {model_path}")

    metrics_path = run_dir / "metrics.json"

    pipe = TransformerBetaPipeline(cfg)

    # build data + model (no training)
    data_art = pipe._build_data(cfg)
    model = pipe.build_model(data_art.input_dim, data_art.output_dim, data_art).to(pipe.device)
    load_state_strict(model, model_path)
    model.to(pipe.device)

    coverage = float(args.coverage if args.coverage is not None else cfg.get("interval_coverage", 0.9))
    logger.info(f"Compute intervals with coverage={coverage}")

    # ---- VAL ----
    yv, av, bv = predict_alpha_beta(pipe, model, data_art.val_loader)
    lv, uv = pipe.postprocess_interval((av.to(pipe.device), bv.to(pipe.device)), coverage=coverage)
    lv_np = lv.detach().cpu().numpy()
    uv_np = uv.detach().cpu().numpy()
    val_picp, val_mpiw = picp_mpiw(yv, lv_np, uv_np)

    # ---- TEST ----
    yt, at, bt = predict_alpha_beta(pipe, model, data_art.test_loader)
    lt, ut = pipe.postprocess_interval((at.to(pipe.device), bt.to(pipe.device)), coverage=coverage)
    lt_np = lt.detach().cpu().numpy()
    ut_np = ut.detach().cpu().numpy()
    test_picp, test_mpiw = picp_mpiw(yt, lt_np, ut_np)

    # save interval artifacts
    val_int_path = run_dir / "intervals_val.npz"
    test_int_path = run_dir / "intervals_test.npz"
    np.savez(val_int_path, y_true=yv, lower=lv_np, upper=uv_np, coverage=coverage)
    np.savez(test_int_path, y_true=yt, lower=lt_np, upper=ut_np, coverage=coverage)

    # patch metrics.json
    patch_metrics_json(
        metrics_path,
        coverage=coverage,
        val_picp=float(val_picp),
        val_mpiw=float(val_mpiw),
        test_picp=float(test_picp),
        test_mpiw=float(test_mpiw),
    )

    logger.info(f"Saved: {val_int_path}")
    logger.info(f"Saved: {test_int_path}")
    logger.info(f"Patched: {metrics_path}")
    logger.info(f"VAL  PICP={val_picp:.4f}  MPIW={val_mpiw:.6f}")
    logger.info(f"TEST PICP={test_picp:.4f}  MPIW={test_mpiw:.6f}")


if __name__ == "__main__":
    main()
