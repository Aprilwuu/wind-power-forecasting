from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import time

import numpy as np
import torch
import yaml

from src.metrics.deterministic import evaluate_rmse, evaluate_mae, evaluate_r2
from src.pipelines.transformer_qr_pipeline import TransformerQRPipeline
from src.pipelines.forecast_base import (
    ensure_dir,
    _get_any,
    _get_nested,
    pinball_loss_np,
    interval_picp_np,
    interval_mpiw_np,
    save_interval_npz,
)

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


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _infer_default_tag(cfg: Dict[str, Any], config_path: Path) -> str:
    tag = cfg.get("tag", None)
    if isinstance(tag, str) and tag.strip():
        return tag.strip()
    return config_path.stem


def _resolve_reports_dir(cfg: Dict[str, Any]) -> Path:
    paths = cfg.get("paths", {}) if isinstance(cfg.get("paths", {}), dict) else {}
    reports_dir = paths.get("reports_dir", None) or "reports/experiments"
    return ensure_dir(reports_dir)


def _is_track2(cfg: Dict[str, Any]) -> bool:
    proto = cfg.get("protocol", None)
    return proto == "track2_lofo_time_val"


def _auto_out_dir(cfg: Dict[str, Any], *, tag: str, seed: int, heldout: Optional[int]) -> Path:
    reports_dir = _resolve_reports_dir(cfg)
    if heldout is None:
        return ensure_dir(reports_dir / tag / f"seed_{seed}")
    return ensure_dir(reports_dir / tag / f"heldout_{heldout}" / f"seed_{seed}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config")
    ap.add_argument("--out_dir", default=None, help="Override base output directory")
    ap.add_argument("--seed", type=int, default=None, help="Single seed (legacy)")
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="Multiple seeds, e.g. 42 43 44")
    ap.add_argument("--heldouts", type=int, nargs="+", default=None, help="Track2 only")
    ap.add_argument("--device", default=None, help="Override device, e.g. cuda/cpu")
    ap.add_argument("--tag", default=None, help="Run tag used in auto-outdir")

    # NEW
    ap.add_argument("--eval_only", action="store_true", help="Skip training and only evaluate from checkpoint")
    ap.add_argument("--ckpt_path", default=None, help="Explicit checkpoint path for eval_only")

    return ap.parse_args()


def _pick_seeds(args: argparse.Namespace) -> List[int]:
    if args.seeds is not None and len(args.seeds) > 0:
        return [int(s) for s in args.seeds]
    if args.seed is not None:
        return [int(args.seed)]
    return [42]


def _pick_heldouts(args: argparse.Namespace, cfg0: Dict[str, Any]) -> List[Optional[int]]:
    if not _is_track2(cfg0):
        return [None]

    if args.heldouts is not None and len(args.heldouts) > 0:
        return [int(h) for h in args.heldouts]

    split = cfg0.get("split", {}) if isinstance(cfg0.get("split", {}), dict) else {}
    if "held_out_group" not in split:
        raise KeyError("Track2 requires split.held_out_group in YAML or pass --heldouts ...")
    return [int(split["held_out_group"])]


def _load_state_strict(model: torch.nn.Module, ckpt_path: Path) -> None:
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=True)


@torch.no_grad()
def _predict_all_qr(
    pipe: TransformerQRPipeline,
    model: torch.nn.Module,
    loader,
    coverage: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      y_true, q50, q05, q95
    """
    model.eval()
    ys, q50s, q05s, q95s = [], [], [], []

    for batch in loader:
        if len(batch) == 2:
            _x, y = batch
        else:
            _x, y, _z = batch

        pred_raw = pipe.model_forward(model, batch)

        q50 = pipe.postprocess_pred(pred_raw).detach().cpu().numpy()
        lo, hi = pipe.postprocess_interval(pred_raw, coverage=coverage)
        lo = lo.detach().cpu().numpy()
        hi = hi.detach().cpu().numpy()

        y_np = y.detach().cpu().numpy()
        y_np = y_np if y_np.ndim == 2 else y_np.reshape(-1, 1)

        q50 = q50 if q50.ndim == 2 else q50.reshape(-1, 1)
        lo = lo if lo.ndim == 2 else lo.reshape(-1, 1)
        hi = hi if hi.ndim == 2 else hi.reshape(-1, 1)

        ys.append(y_np)
        q50s.append(q50)
        q05s.append(lo)
        q95s.append(hi)

    return (
        np.concatenate(ys, axis=0),
        np.concatenate(q50s, axis=0),
        np.concatenate(q05s, axis=0),
        np.concatenate(q95s, axis=0),
    )


@torch.no_grad()
def _predict_quantiles_np(
    pipe: TransformerQRPipeline,
    model: torch.nn.Module,
    loader,
) -> np.ndarray:
    model.eval()
    qs = []
    for batch in loader:
        pred_raw = pipe.model_forward(model, batch)
        q_t = pipe.postprocess_quantiles(pred_raw)
        qs.append(q_t.detach().cpu().numpy())
    return np.concatenate(qs, axis=0)


def _save_qr_npz(path: Path, y_true: np.ndarray, q05: np.ndarray, q50: np.ndarray, q95: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    y_true = y_true.reshape(-1, 1) if y_true.ndim == 1 else y_true
    q05 = q05.reshape(-1, 1) if q05.ndim == 1 else q05
    q50 = q50.reshape(-1, 1) if q50.ndim == 1 else q50
    q95 = q95.reshape(-1, 1) if q95.ndim == 1 else q95

    np.savez_compressed(
        path,
        y_true=y_true.astype(np.float32),
        q05=q05.astype(np.float32),
        q50=q50.astype(np.float32),
        q95=q95.astype(np.float32),
    )


def _eval_only_run(
    cfg: Dict[str, Any],
    out_dir: Path,
    ckpt_path: Path,
) -> Dict[str, Any]:
    pipe = TransformerQRPipeline(cfg)
    data_art = pipe._build_data(cfg)

    model = pipe.build_model(data_art.input_dim, data_art.output_dim, data_art).to(pipe.device)
    _load_state_strict(model, ckpt_path)
    model.to(pipe.device)
    model.eval()

    coverage = float(cfg.get("interval_coverage", 0.9))

    y_val, q50_val, q05_val, q95_val = _predict_all_qr(pipe, model, data_art.val_loader, coverage=coverage)
    y_test, q50_test, q05_test, q95_test = _predict_all_qr(pipe, model, data_art.test_loader, coverage=coverage)

    metrics: Dict[str, Any] = {
        "track": cfg.get("protocol", cfg.get("track", None)),
        "split": cfg.get("split", {}),
        "seq": data_art.meta.get("seq", {}),
        "keep_zone": data_art.meta.get("keep_zone", False),
        "val": {
            "rmse": evaluate_rmse(y_val, q50_val),
            "mae": evaluate_mae(y_val, q50_val),
            "r2": evaluate_r2(y_val, q50_val),
            "n_seq": int(len(y_val)),
        },
        "test": {
            "rmse": evaluate_rmse(y_test, q50_test),
            "mae": evaluate_mae(y_test, q50_test),
            "r2": evaluate_r2(y_test, q50_test),
            "n_seq": int(len(y_test)),
        },
        "n_rows": data_art.meta.get("n_rows", {}),
        "n_seq_samples": data_art.meta.get("n_seq_samples", {}),
        "notes": [
            "EVAL-ONLY mode: no training performed.",
            "IMPORTANT: split first, then window (no leakage).",
        ],
    }

    qs = _get_nested(cfg, "probabilistic.quantiles", None)
    if qs is not None:
        try:
            q_val_all = _predict_quantiles_np(pipe, model, data_art.val_loader)
            q_test_all = _predict_quantiles_np(pipe, model, data_art.test_loader)

            metrics.setdefault("val_prob", {})
            metrics.setdefault("test_prob", {})
            metrics["val_prob"]["quantiles"] = list(qs)
            metrics["test_prob"]["quantiles"] = list(qs)
            metrics["val_prob"]["pinball"] = pinball_loss_np(y_val, q_val_all, list(qs))
            metrics["test_prob"]["pinball"] = pinball_loss_np(y_test, q_test_all, list(qs))
        except Exception as e:
            logger.exception(f"Pinball evaluation failed in eval_only: {e}")

    if bool(cfg.get("compute_interval", False)):
        metrics.setdefault("val_prob", {})
        metrics.setdefault("test_prob", {})
        metrics["val_prob"]["coverage_target"] = coverage
        metrics["val_prob"]["picp"] = interval_picp_np(y_val, q05_val, q95_val)
        metrics["val_prob"]["mpiw"] = interval_mpiw_np(q05_val, q95_val)
        metrics["test_prob"]["coverage_target"] = coverage
        metrics["test_prob"]["picp"] = interval_picp_np(y_test, q05_test, q95_test)
        metrics["test_prob"]["mpiw"] = interval_mpiw_np(q05_test, q95_test)

        protocol = str(cfg.get("protocol", ""))
        if protocol == "track2_lofo_time_val":
            val_name = "preds_outer_val_qr.npz"
            test_name = "preds_outer_test_transformer_qr.npz"
        else:
            val_name = "preds_val_qr.npz"
            test_name = "preds_test_qr.npz"

        _save_qr_npz(out_dir / val_name, y_val, q05_val, q50_val, q95_val)
        _save_qr_npz(out_dir / test_name, y_test, q05_test, q50_test, q95_test)

    write_json(out_dir / "metrics.json", metrics)
    torch.save({"state_dict": model.state_dict()}, str(out_dir / "model.pt"))

    return {"metrics": metrics, "config": cfg}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args()

    config_path = Path(args.config)
    cfg0 = load_yaml(config_path)

    seeds = _pick_seeds(args)
    heldouts = _pick_heldouts(args, cfg0)

    tag = (
        args.tag.strip()
        if isinstance(args.tag, str) and args.tag.strip()
        else _infer_default_tag(cfg0, config_path)
    )

    for heldout in heldouts:
        for seed in seeds:
            cfg: Dict[str, Any] = json.loads(json.dumps(cfg0))

            if args.device is not None:
                cfg["device"] = str(args.device)
                cfg.setdefault("training", {})
                if isinstance(cfg["training"], dict):
                    cfg["training"]["device"] = str(args.device)

            cfg["seed"] = int(seed)

            if heldout is not None:
                cfg.setdefault("split", {})
                if not isinstance(cfg["split"], dict):
                    cfg["split"] = {}
                cfg["split"]["held_out_group"] = int(heldout)

            if args.out_dir:
                base = ensure_dir(args.out_dir)
                if heldout is None:
                    out_dir = ensure_dir(base / tag / f"seed_{seed}")
                else:
                    out_dir = ensure_dir(base / tag / f"heldout_{heldout}" / f"seed_{seed}")
            else:
                out_dir = _auto_out_dir(cfg, tag=tag, seed=seed, heldout=heldout)

            cfg["out_dir"] = str(out_dir)
            cfg["tag"] = tag

            set_seed(seed)

            prefix = f"[heldout={heldout} seed={seed}]" if heldout is not None else f"[seed={seed}]"
            logger.info(f"{prefix} out_dir={out_dir}")

            start_time = time.time()

            if args.eval_only:
                if args.ckpt_path:
                    ckpt_path = Path(args.ckpt_path)
                else:
                    ckpt_path = out_dir / "model.pt"

                if not ckpt_path.exists():
                    raise FileNotFoundError(
                        f"{prefix} eval_only requires checkpoint, but not found: {ckpt_path}"
                    )

                logger.info(f"{prefix} Eval-only from checkpoint: {ckpt_path}")
                out = _eval_only_run(cfg, out_dir, ckpt_path)
            else:
                logger.info(f"{prefix} Loading pipeline...")
                pipe = TransformerQRPipeline(cfg)
                logger.info(f"{prefix} Running training...")
                out = pipe.run()

            end_time = time.time()
            train_minutes = (end_time - start_time) / 60.0
            logger.info(f"{prefix} Total time: {train_minutes:.2f} min")

            runtime = {
                "runtime": {
                    "config": str(config_path.resolve()),
                    "out_dir": str(out_dir),
                    "seed": int(seed),
                    "tag": tag,
                    "device": cfg.get("device", (cfg.get("training", {}) or {}).get("device", None)),
                    "pipeline": cfg.get("pipeline", "transformer_qr"),
                    "protocol": cfg.get("protocol", None),
                    "held_out_group": None if heldout is None else int(heldout),
                    "data_path": (cfg.get("data", {}) or {}).get("path", None),
                    "train_cut": (cfg.get("split", {}) or {}).get("train_cut", None),
                    "val_cut": (cfg.get("split", {}) or {}).get("val_cut", None),
                    "val_days": (cfg.get("split", {}) or {}).get("val_days", None),
                    "lookback": (cfg.get("seq", {}) or {}).get("lookback", None),
                    "quantiles": (cfg.get("probabilistic", {}) or {}).get("quantiles", None),
                    "paths": cfg.get("paths", {}),
                    "train_minutes": train_minutes,
                    "eval_only": bool(args.eval_only),
                    "ckpt_path": str(args.ckpt_path) if args.ckpt_path else None,
                }
            }
            write_json(out_dir / "runtime.json", runtime)

            if isinstance(out, dict) and "metrics" in out and isinstance(out["metrics"], dict):
                write_json(out_dir / "metrics_summary.json", out["metrics"])

            logger.info(f"{prefix} Done.")


if __name__ == "__main__":
    main()