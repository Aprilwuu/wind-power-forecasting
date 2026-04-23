from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple

import numpy as np
import torch
import yaml

from src.pipelines.beta_transformer_pipeline import TransformerBetaPipeline

logger = logging.getLogger(__name__)


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


def is_cache_mode(cfg: Dict[str, Any]) -> bool:
    return bool(cfg.get("use_vae", False)) and str(cfg.get("vae_mode", "")).lower().strip() == "cache"


def build_vae_ckg_dir(
    *,
    vae_root: Path,
    protocol: str,
    seed: int,
    held_out_group: Optional[Union[int, str]],
) -> Path:
    """
    Default convention:
      - Track2: {vae_root}/heldout_{g}/seed_{seed}/artifacts
      - Track1: {vae_root}/seed_{seed}/artifacts
    """
    if protocol == "track2_lofo_time_val":
        if held_out_group is None:
            raise ValueError("Track2 cache mode requires held_out_group to locate VAE artifacts.")
        return vae_root / f"heldout_{held_out_group}" / f"seed_{seed}" / "artifacts"
    return vae_root / f"seed_{seed}" / "artifacts"


def assert_latents_exist(ckg_dir: Path) -> None:
    required = ["latents_train.npy", "latents_val.npy", "latents_test.npy"]
    missing = [name for name in required if not (ckg_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            "VAE cache mode is enabled, but latent files are missing.\n"
            f"  vae_ckg_dir: {ckg_dir}\n"
            f"  missing: {missing}\n"
            "Run your VAE export for this (heldout, seed) first, and ensure it writes into this folder."
        )


def resolve_vae_root_from_cfg(cfg: Dict[str, Any], cli_vae_root: Path) -> Path:
    """
    Resolve VAE root directory for cache mode.
    Priority:
      1) cfg['vae_ckg_dir'] (preferred)
      2) cfg['vae_ckpt_dir'] (legacy name)
      3) CLI --vae_root
    """
    v = cfg.get("vae_ckg_dir", None) or cfg.get("vae_ckpt_dir", None)
    if v:
        return Path(v)
    return cli_vae_root


def _resolve_paths_from_cfg_and_cli(cfg_base: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Path]:
    """
    Decide where to store:
      - artifacts (per-run out_dir): default cfg.paths.featured_dir or data/featured
      - reports (summary.json, runs.csv): default cfg.paths.reports_dir or reports/experiments
    """
    cfg_paths = cfg_base.get("paths", {}) or {}

    if args.out_root:
        artifact_root = Path(args.out_root)
    else:
        artifact_root = Path(cfg_paths.get("featured_dir", "data/featured"))

    if args.reports_root:
        reports_root = Path(args.reports_root)
    else:
        if cfg_paths.get("reports_dir"):
            reports_root = Path(cfg_paths["reports_dir"])
        else:
            reports_root = artifact_root

    return {
        "artifact_root": ensure_dir(artifact_root),
        "reports_root": ensure_dir(reports_root),
    }


@torch.no_grad()
def predict_alpha_beta(
    pipe: TransformerBetaPipeline,
    model: torch.nn.Module,
    loader,
) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor]:
    """
    Predict raw (alpha, beta) for a loader.

    Returns:
      y_true_np: [N,1] numpy
      alpha:     [N,1] torch (cpu)
      beta:      [N,1] torch (cpu)
    """
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
    """
    y_true/lower/upper: [N,1]
    """
    cover = (y_true >= lower) & (y_true <= upper)
    picp = float(cover.mean())
    mpiw = float((upper - lower).mean())
    return picp, mpiw


def load_state_strict(model: torch.nn.Module, ckpt_path: Path) -> None:
    """
    Load a checkpoint strictly.
    """
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=True)


def patch_metrics_json(
    metrics_path: Path,
    *,
    coverage: float,
    val_picp: Optional[float],
    val_mpiw: Optional[float],
    test_picp: Optional[float],
    test_mpiw: Optional[float],
) -> None:
    """
    Add interval metrics into metrics.json without breaking existing keys.
    """
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics.setdefault("val", {})
    metrics.setdefault("test", {})

    metrics["val"]["coverage"] = coverage
    metrics["val"]["picp"] = val_picp
    metrics["val"]["mpiw"] = val_mpiw

    metrics["test"]["coverage"] = coverage
    metrics["test"]["picp"] = test_picp
    metrics["test"]["mpiw"] = test_mpiw

    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")


def compute_intervals_and_save(
    *,
    pipe: TransformerBetaPipeline,
    model: torch.nn.Module,
    data_art,
    cfg: Dict[str, Any],
    run_dir: Path,
) -> Dict[str, Any]:
    """
    Compute prediction intervals + PICP/MPIW for val/test and save npz files.

    Also saves:
      - mu
      - mean (same as mu, for easier downstream compatibility)
      - alpha
      - beta

    Config options:
      - compute_interval: bool
      - interval_coverage: float, e.g. 0.9
      - interval_on_val: bool (default False for speed)
    """
    compute_interval = bool(cfg.get("compute_interval", False))
    coverage = float(cfg.get("interval_coverage", 0.9))
    interval_on_val = bool(cfg.get("interval_on_val", False))

    out = {
        "coverage": coverage,
        "val_picp": None,
        "val_mpiw": None,
        "test_picp": None,
        "test_mpiw": None,
        "intervals_val_path": "",
        "intervals_test_path": "",
    }

    if not compute_interval:
        logger.info("compute_interval=false -> skip interval computation (PICP/MPIW).")
        return out

    # ---- VAL (optional for speed) ----
    if interval_on_val:
        yv, av, bv = predict_alpha_beta(pipe, model, data_art.val_loader)

        av_dev = av.to(pipe.device)
        bv_dev = bv.to(pipe.device)

        mu_v = pipe.postprocess_mu((av_dev, bv_dev))
        lv, uv = pipe.postprocess_interval((av_dev, bv_dev), coverage=coverage)

        mu_v_np = mu_v.detach().cpu().numpy()
        lv_np = lv.detach().cpu().numpy()
        uv_np = uv.detach().cpu().numpy()

        val_picp, val_mpiw = picp_mpiw(yv, lv_np, uv_np)

        val_int_path = run_dir / "intervals_val.npz"
        np.savez(
            val_int_path,
            y_true=yv,
            mu=mu_v_np,
            mean=mu_v_np,
            lower=lv_np,
            upper=uv_np,
            alpha=av.numpy(),
            beta=bv.numpy(),
            coverage=coverage,
        )

        out["val_picp"] = float(val_picp)
        out["val_mpiw"] = float(val_mpiw)
        out["intervals_val_path"] = str(val_int_path)
    else:
        logger.info("interval_on_val=false -> skip val interval computation for speed.")

    # ---- TEST ----
    yt, at, bt = predict_alpha_beta(pipe, model, data_art.test_loader)

    at_dev = at.to(pipe.device)
    bt_dev = bt.to(pipe.device)

    mu_t = pipe.postprocess_mu((at_dev, bt_dev))
    lt, ut = pipe.postprocess_interval((at_dev, bt_dev), coverage=coverage)

    mu_t_np = mu_t.detach().cpu().numpy()
    lt_np = lt.detach().cpu().numpy()
    ut_np = ut.detach().cpu().numpy()

    test_picp, test_mpiw = picp_mpiw(yt, lt_np, ut_np)

    test_int_path = run_dir / "intervals_test.npz"
    np.savez(
        test_int_path,
        y_true=yt,
        mu=mu_t_np,
        mean=mu_t_np,
        lower=lt_np,
        upper=ut_np,
        alpha=at.numpy(),
        beta=bt.numpy(),
        coverage=coverage,
    )

    out.update(
        {
            "test_picp": float(test_picp),
            "test_mpiw": float(test_mpiw),
            "intervals_test_path": str(test_int_path),
        }
    )
    return out


def train_or_load_and_eval(
    *,
    cfg: Dict[str, Any],
    run_dir: Path,
    eval_only: bool,
) -> Tuple[TransformerBetaPipeline, Dict[str, Any], Dict[str, Any]]:
    """
    If eval_only:
      - do NOT train
      - load model.pt and compute metrics + (optional) intervals
    Else:
      - train via pipe.run()
      - then load model.pt and compute intervals (optional)

    Returns:
      pipe, point_metrics (dict), interval_out (dict)
    """
    pipe = TransformerBetaPipeline(cfg)

    data_art = pipe._build_data(cfg)
    model = pipe.build_model(data_art.input_dim, data_art.output_dim, data_art).to(pipe.device)
    model_path = run_dir / "model.pt"
    metrics_path = run_dir / "metrics.json"

    point_metrics: Dict[str, Any] = {}

    if not eval_only:
        out = pipe.run()
        point_metrics = out.get("metrics", {})
    else:
        if not model_path.exists():
            raise FileNotFoundError(f"--eval_only set but checkpoint not found: {model_path}")
        if not metrics_path.exists():
            logger.warning(f"--eval_only: metrics.json not found yet: {metrics_path}. Will create/patch it.")

    load_state_strict(model, model_path)
    model.to(pipe.device)

    interval_out = compute_intervals_and_save(pipe=pipe, model=model, data_art=data_art, cfg=cfg, run_dir=run_dir)

    if not metrics_path.exists():
        metrics_path.write_text(json.dumps({"val": {}, "test": {}}, indent=2), encoding="utf-8")

    patch_metrics_json(
        metrics_path,
        coverage=float(interval_out["coverage"]),
        val_picp=interval_out["val_picp"],
        val_mpiw=interval_out["val_mpiw"],
        test_picp=interval_out["test_picp"],
        test_mpiw=interval_out["test_mpiw"],
    )

    if eval_only and not point_metrics:
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            point_metrics = {"val": metrics.get("val", {}), "test": metrics.get("test", {})}
        except Exception:
            point_metrics = {"val": {}, "test": {}}

    return pipe, point_metrics, interval_out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str)

    ap.add_argument(
        "--out_root",
        default=None,
        type=str,
        help="Override ARTIFACT root (per-run out_dir). Default: cfg.paths.featured_dir or data/featured",
    )
    ap.add_argument(
        "--reports_root",
        default=None,
        type=str,
        help="Override REPORTS root (summary.json, runs.csv). Default: cfg.paths.reports_dir or reports/experiments",
    )

    ap.add_argument("--exp_name", required=True, type=str, help="e.g. beta_transformer_track2_lb168")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])

    # Track2 only
    ap.add_argument("--held_out_groups", nargs="*", default=None)

    # Eval-only
    ap.add_argument(
        "--eval_only",
        action="store_true",
        help="Skip training and only load existing model.pt to compute (optional) intervals and patch metrics.",
    )

    # VAE cache root
    ap.add_argument("--vae_root", type=str, default="reports/vae", help="Root folder for VAE artifacts (cache mode).")

    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    cfg_base = load_yaml(args.config)
    protocol = cfg_base.get("protocol", None)
    if not protocol:
        raise KeyError("YAML must contain 'protocol'")

    is_track2 = (protocol == "track2_lofo_time_val")

    roots = _resolve_paths_from_cfg_and_cli(cfg_base, args)
    artifact_root = roots["artifact_root"]
    reports_root = roots["reports_root"]

    exp_art_dir = ensure_dir(artifact_root / args.exp_name)
    exp_rep_dir = ensure_dir(reports_root / args.exp_name)

    vae_root_cli = Path(args.vae_root)

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
            run_dir = ensure_dir(exp_art_dir / f"seed_{seed}")

            cfg = dict(cfg_base)
            cfg["seed"] = int(seed)
            cfg["out_dir"] = str(run_dir)

            if is_cache_mode(cfg):
                vae_root = resolve_vae_root_from_cfg(cfg, vae_root_cli)

                if (vae_root / "latents_train.npy").exists():
                    ckg_dir = vae_root
                else:
                    ckg_dir = build_vae_ckg_dir(
                        vae_root=vae_root, protocol=protocol, seed=int(seed), held_out_group=None
                    )

                logger.info(f"[cache] resolved VAE artifacts dir: {ckg_dir}")
                assert_latents_exist(ckg_dir)
                cfg["vae_ckg_dir"] = str(ckg_dir)

            start_time = time.time()
            _, point_metrics, interval_out = train_or_load_and_eval(
                cfg=cfg,
                run_dir=run_dir,
                eval_only=args.eval_only,
            )
            end_time = time.time()
            train_minutes = (end_time - start_time) / 60.0

            m_val = point_metrics.get("val", {})
            m_test = point_metrics.get("test", {})

            model_path = run_dir / "model.pt"
            metrics_path = run_dir / "metrics.json"

            runs.append(
                {
                    "pipeline": "transformer_beta",
                    "protocol": protocol,
                    "seed": int(seed),
                    "out_dir": str(run_dir),
                    "vae_ckg_dir": cfg.get("vae_ckg_dir", None),

                    "val_rmse": m_val.get("rmse"),
                    "val_mae": m_val.get("mae"),
                    "val_r2": m_val.get("r2"),
                    "test_rmse": m_test.get("rmse"),
                    "test_mae": m_test.get("mae"),
                    "test_r2": m_test.get("r2"),

                    "coverage": float(interval_out["coverage"]),
                    "val_picp": interval_out["val_picp"],
                    "val_mpiw": interval_out["val_mpiw"],
                    "test_picp": interval_out["test_picp"],
                    "test_mpiw": interval_out["test_mpiw"],

                    "train_minutes": train_minutes,

                    "metrics_path": str(metrics_path),
                    "model_path": str(model_path),
                    "intervals_val_path": interval_out["intervals_val_path"],
                    "intervals_test_path": interval_out["intervals_test_path"],
                }
            )

        else:
            for g in heldouts:
                run_dir = ensure_dir(exp_art_dir / f"heldout_{g}" / f"seed_{seed}")

                cfg = dict(cfg_base)
                cfg["seed"] = int(seed)
                cfg["out_dir"] = str(run_dir)
                cfg["held_out_group"] = g

                if is_cache_mode(cfg):
                    vae_root = resolve_vae_root_from_cfg(cfg, vae_root_cli)

                    if (vae_root / "latents_train.npy").exists():
                        ckg_dir = vae_root
                    else:
                        ckg_dir = build_vae_ckg_dir(
                            vae_root=vae_root, protocol=protocol, seed=int(seed), held_out_group=g
                        )

                    logger.info(f"[cache] resolved VAE artifacts dir: {ckg_dir}")
                    assert_latents_exist(ckg_dir)
                    cfg["vae_ckg_dir"] = str(ckg_dir)

                start_time = time.time()
                _, point_metrics, interval_out = train_or_load_and_eval(
                    cfg=cfg,
                    run_dir=run_dir,
                    eval_only=args.eval_only,
                )
                end_time = time.time()
                train_minutes = (end_time - start_time) / 60.0

                m_val = point_metrics.get("val", {})
                m_test = point_metrics.get("test", {})

                model_path = run_dir / "model.pt"
                metrics_path = run_dir / "metrics.json"

                runs.append(
                    {
                        "pipeline": "transformer_beta",
                        "protocol": protocol,
                        "held_out_group": g,
                        "seed": int(seed),
                        "out_dir": str(run_dir),
                        "vae_ckg_dir": cfg.get("vae_ckg_dir", None),

                        "val_rmse": m_val.get("rmse"),
                        "val_mae": m_val.get("mae"),
                        "val_r2": m_val.get("r2"),
                        "test_rmse": m_test.get("rmse"),
                        "test_mae": m_test.get("mae"),
                        "test_r2": m_test.get("r2"),

                        "coverage": float(interval_out["coverage"]),
                        "val_picp": interval_out["val_picp"],
                        "val_mpiw": interval_out["val_mpiw"],
                        "test_picp": interval_out["test_picp"],
                        "test_mpiw": interval_out["test_mpiw"],

                        "train_minutes": train_minutes,

                        "metrics_path": str(metrics_path),
                        "model_path": str(model_path),
                        "intervals_val_path": interval_out["intervals_val_path"],
                        "intervals_test_path": interval_out["intervals_test_path"],
                    }
                )

    val_rmse_list = [x for x in (safe_float(r.get("val_rmse")) for r in runs) if x is not None]
    test_rmse_list = [x for x in (safe_float(r.get("test_rmse")) for r in runs) if x is not None]

    val_picp_list = [x for x in (safe_float(r.get("val_picp")) for r in runs) if x is not None]
    test_picp_list = [x for x in (safe_float(r.get("test_picp")) for r in runs) if x is not None]
    val_mpiw_list = [x for x in (safe_float(r.get("val_mpiw")) for r in runs) if x is not None]
    test_mpiw_list = [x for x in (safe_float(r.get("test_mpiw")) for r in runs) if x is not None]
    runtime_list = [x for x in (safe_float(r.get("train_minutes")) for r in runs) if x is not None]

    summary = {
        "experiment": args.exp_name,
        "pipeline": "transformer_beta",
        "protocol": protocol,
        "config": str(Path(args.config).resolve()),
        "seeds": list(args.seeds),
        "n_runs": len(runs),
        "held_out_groups": heldouts if is_track2 else None,
        "val": {
            "rmse": summarize(val_rmse_list),
            "picp": summarize(val_picp_list),
            "mpiw": summarize(val_mpiw_list),
        },
        "test": {
            "rmse": summarize(test_rmse_list),
            "picp": summarize(test_picp_list),
            "mpiw": summarize(test_mpiw_list),
        },
        "runtime": summarize(runtime_list),
        "runs": runs,
        "artifact_root": str(artifact_root),
        "artifact_exp_dir": str(exp_art_dir),
        "reports_root": str(reports_root),
        "reports_exp_dir": str(exp_rep_dir),
        "vae_root": str(vae_root_cli.resolve()),
    }

    summary_path = exp_rep_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Saved summary -> {summary_path}")

    runs_csv = exp_rep_dir / "runs.csv"
    write_csv(runs_csv, runs)
    logger.info(f"Saved runs.csv -> {runs_csv}")


if __name__ == "__main__":
    main()