# src/experiments/run_transformer_det.py
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import time

import yaml

from src.pipelines.transformer_det_pipeline import TransformerDetPipeline

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
        return yaml.safe_load(f)


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
    import numpy as np

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

    If your Track1 VAE is stored differently (e.g., vae_root/track1/seed_{seed}/artifacts),
    change the Track1 branch here.
    """
    if protocol == "track2_lofo_time_val":
        if held_out_group is None:
            raise ValueError("Track2 cache mode requires held_out_group to locate VAE artifacts.")
        return vae_root / f"heldout_{held_out_group}" / f"seed_{seed}" / "artifacts"
    # track1_temporal
    return vae_root / f"seed_{seed}" / "artifacts"


def assert_latents_exist(ckg_dir: Path) -> None:
    required = ["latents_train.npy", "latents_val.npy", "latents_test.npy"]
    missing = [name for name in required if not (ckg_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            "VAE cache mode is enabled, but latent files are missing.\n"
            f"  vae_ckg_dir: {ckg_dir}\n"
            f"  missing: {missing}\n"
            "Make sure you have already run VAE export for this (heldout, seed), "
            "and that your VAE runner writes into this folder."
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

    Backward compatibility:
      - If user still passes --out_root, treat it as artifact root override.
      - If cfg lacks reports_dir and user didn't pass --reports_root, fall back to artifact root
        (so old behavior "everything under out_root" is still possible).
    """
    cfg_paths = cfg_base.get("paths", {}) or {}

    # Artifact root (where model/metrics per run go)
    if args.out_root:
        artifact_root = Path(args.out_root)
    else:
        artifact_root = Path(cfg_paths.get("featured_dir", "data/featured"))

    # Reports root (where summary/runs.csv go)
    if args.reports_root:
        reports_root = Path(args.reports_root)
    else:
        if "reports_dir" in cfg_paths and cfg_paths.get("reports_dir"):
            reports_root = Path(cfg_paths["reports_dir"])
        else:
            # fallback to artifact root for backward compatibility
            reports_root = artifact_root

    return {
        "artifact_root": ensure_dir(artifact_root),
        "reports_root": ensure_dir(reports_root),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str)

    # CHANGED:
    # --out_root now overrides artifact root (default from cfg.paths.featured_dir)
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

    ap.add_argument("--exp_name", required=True, type=str, help="e.g. transformer_det_track1_lb168")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])

    # Track2 only
    ap.add_argument("--held_out_groups", nargs="*", default=None)

    # VAE cache root (only used when use_vae=true and vae_mode=cache)
    ap.add_argument("--vae_root", type=str, default="reports/vae", help="Root folder for VAE artifacts.")

    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    cfg_base = load_yaml(args.config)
    protocol = cfg_base.get("protocol", None)
    if not protocol:
        raise KeyError("YAML must contain 'protocol'")

    is_track2 = (protocol == "track2_lofo_time_val")

    # NEW: resolve artifact & reports roots
    roots = _resolve_paths_from_cfg_and_cli(cfg_base, args)
    artifact_root = roots["artifact_root"]
    reports_root = roots["reports_root"]

    # experiment dirs
    exp_art_dir = ensure_dir(artifact_root / args.exp_name)   # per-run artifacts live here
    exp_rep_dir = ensure_dir(reports_root / args.exp_name)    # summaries live here

    # v2: keep CLI root; per-run we may override by cfg['vae_ckg_dir']/cfg['vae_ckpt_dir']
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

            # Auto inject VAE cache dir (Track1) - v2
            if is_cache_mode(cfg):
                vae_root = resolve_vae_root_from_cfg(cfg, vae_root_cli)

                # If user already provided leaf artifacts dir, use it directly.
                if (vae_root / "latents_train.npy").exists():
                    ckg_dir = vae_root
                else:
                    ckg_dir = build_vae_ckg_dir(
                        vae_root=vae_root, protocol=protocol, seed=int(seed), held_out_group=None
                    )

                logger.info(f"[cache] resolved VAE artifacts dir: {ckg_dir}")
                assert_latents_exist(ckg_dir)
                cfg["vae_ckg_dir"] = str(ckg_dir)

            pipe = TransformerDetPipeline(cfg)

            start_time = time.time()

            out = pipe.run()

            end_time = time.time()
            train_minutes = (end_time - start_time) / 60

            logger.info(f"[time] seed={seed} | {train_minutes:.2f} min")

            m = out["metrics"]

            runs.append(
                {
                    "pipeline": "transformer_det",
                    "protocol": protocol,
                    "seed": int(seed),
                    "train_minutes": train_minutes,
                    "out_dir": str(run_dir),
                    "vae_ckg_dir": cfg.get("vae_ckg_dir", None),
                    "val_rmse": m["val"]["rmse"],
                    "val_mae": m["val"]["mae"],
                    "val_r2": m["val"]["r2"],
                    "test_rmse": m["test"]["rmse"],
                    "test_mae": m["test"]["mae"],
                    "test_r2": m["test"]["r2"],
                    "metrics_path": str(Path(run_dir) / "metrics.json"),
                    "model_path": str(Path(run_dir) / "model.pt"),
                }
            )
        else:
            for g in heldouts:
                run_dir = ensure_dir(exp_art_dir / f"heldout_{g}" / f"seed_{seed}")

                cfg = dict(cfg_base)
                cfg["seed"] = int(seed)
                cfg["out_dir"] = str(run_dir)
                cfg["held_out_group"] = g

                # Auto inject VAE cache dir (Track2) - v2
                if is_cache_mode(cfg):
                    vae_root = resolve_vae_root_from_cfg(cfg, vae_root_cli)

                    # If user already provided leaf artifacts dir, use it directly.
                    if (vae_root / "latents_train.npy").exists():
                        ckg_dir = vae_root
                    else:
                        ckg_dir = build_vae_ckg_dir(
                            vae_root=vae_root, protocol=protocol, seed=int(seed), held_out_group=g
                        )

                    logger.info(f"[cache] resolved VAE artifacts dir: {ckg_dir}")
                    assert_latents_exist(ckg_dir)
                    cfg["vae_ckg_dir"] = str(ckg_dir)

                pipe = TransformerDetPipeline(cfg)

                start_time = time.time()

                out = pipe.run()

                end_time = time.time()
                train_minutes = (end_time - start_time) / 60

                logger.info(f"[time] heldout={g} seed={seed} | {train_minutes:.2f} min")

                m = out["metrics"]

                runs.append(
                    {
                        "pipeline": "transformer_det",
                        "protocol": protocol,
                        "held_out_group": g,
                        "seed": int(seed),
                        "train_minutes": train_minutes,
                        "out_dir": str(run_dir),
                        "vae_ckg_dir": cfg.get("vae_ckg_dir", None),
                        "val_rmse": m["val"]["rmse"],
                        "val_mae": m["val"]["mae"],
                        "val_r2": m["val"]["r2"],
                        "test_rmse": m["test"]["rmse"],
                        "test_mae": m["test"]["mae"],
                        "test_r2": m["test"]["r2"],
                        "metrics_path": str(Path(run_dir) / "metrics.json"),
                        "model_path": str(Path(run_dir) / "model.pt"),
                    }
                )

    # ---- summary ----
    val_rmse_list = [x for x in (safe_float(r.get("val_rmse")) for r in runs) if x is not None]
    test_rmse_list = [x for x in (safe_float(r.get("test_rmse")) for r in runs) if x is not None]

    summary = {
        "experiment": args.exp_name,
        "pipeline": "transformer_det",
        "protocol": protocol,
        "config": str(Path(args.config).resolve()),
        "seeds": list(args.seeds),
        "n_runs": len(runs),
        "held_out_groups": heldouts if is_track2 else None,
        "val": {"rmse": summarize(val_rmse_list)},
        "test": {"rmse": summarize(test_rmse_list)},
        "runs": runs,
        # NEW: separate roots for clarity
        "artifact_root": str(artifact_root),
        "artifact_exp_dir": str(exp_art_dir),
        "reports_root": str(reports_root),
        "reports_exp_dir": str(exp_rep_dir),
        "vae_root": str(vae_root_cli.resolve()),
    }

    summary_path = Path(exp_rep_dir) / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Saved summary -> {summary_path}")

    runs_csv = Path(exp_rep_dir) / "runs.csv"
    write_csv(runs_csv, runs)
    logger.info(f"Saved runs.csv -> {runs_csv}")


if __name__ == "__main__":
    main()
