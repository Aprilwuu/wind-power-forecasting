# src/experiments/run_deterministic_tcn.py
from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
import csv
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import yaml

from src.data.load import load_raw_data
from src.pipelines.tcn_pipeline import run_pipeline_any


logger = logging.getLogger(__name__)


# -----------------------------
# utils
# -----------------------------
def _summarize(xs: List[float]) -> Dict[str, float]:
    if len(xs) == 0:
        return {"mean": math.nan, "std": math.nan, "min": math.nan, "max": math.nan}
    arr = np.asarray(xs, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override into base (override wins)."""
    out = dict(base)
    for k, v in override.items():
        if (
            k in out
            and isinstance(out[k], dict)
            and isinstance(v, dict)
        ):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def _load_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_out_root(cfg: Dict[str, Any]) -> Path:
    # Support multiple config layouts
    # Priority: cfg["paths"]["featured_dir"], then cfg["out_root"], else default to data/featured
    paths = cfg.get("paths", {}) if isinstance(cfg.get("paths", {}), dict) else {}
    out_root = paths.get("featured_dir") or cfg.get("out_root") or "data/featured"
    return Path(out_root).expanduser().resolve()


# NEW: resolve reports root for experiment summaries
def _resolve_reports_root(cfg: Dict[str, Any]) -> Path:
    # Priority: cfg["paths"]["reports_dir"], then cfg["reports_root"], else default to reports/experiments
    paths = cfg.get("paths", {}) if isinstance(cfg.get("paths", {}), dict) else {}
    reports_root = paths.get("reports_dir") or cfg.get("reports_root") or cfg.get("reports_root") or "reports/experiments"
    return Path(reports_root).expanduser().resolve()


# Helper: write runs CSV
def _write_runs_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    # Collect all keys
    all_keys = set()
    for row in rows:
        all_keys.update(row.keys())
    columns = sorted(all_keys)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            # Fill missing with ""
            row_filled = {k: row.get(k, "") for k in columns}
            writer.writerow(row_filled)


def _resolve_data_path(cfg: Dict[str, Any]) -> Path:
    # Support: cfg["data"]["path"] or cfg["data_path"]
    if isinstance(cfg.get("data"), dict) and cfg["data"].get("path"):
        return Path(cfg["data"]["path"]).expanduser().resolve()
    if cfg.get("data_path"):
        return Path(cfg["data_path"]).expanduser().resolve()
    raise KeyError("Config must provide data.path or data_path")


def _resolve_split(cfg: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    Returns (split_type, split_cfg)
      split_type in {"temporal", "lofo"}
    """
    split_cfg = cfg.get("split", {}) if isinstance(cfg.get("split", {}), dict) else {}
    raw = (split_cfg.get("type") or split_cfg.get("split_type") or "temporal").lower()

    if raw in ("temporal", "track1", "track1_temporal"):
        return "temporal", split_cfg
    if raw in ("lofo", "track2", "track2_lofo"):
        return "lofo", split_cfg

    # Fallback: if both train_cut and val_cut exist -> temporal; otherwise -> lofo
    if split_cfg.get("train_cut") and split_cfg.get("val_cut"):
        return "temporal", split_cfg
    if split_cfg.get("val_days") or split_cfg.get("held_out_group"):
        return "lofo", split_cfg

    raise ValueError(f"Unrecognized split.type: {raw}")


def _resolve_candidates(cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Supported formats:
      1) Mapping:
           candidates:
             baseline: {}
             big: { model: {channels: [64, 64, 64]} }
      2) List:
           candidates:
             - name: baseline
               override: {...}
    """
    cand = cfg.get("candidates", None)
    if cand is None:
        return {"baseline": {}}

    if isinstance(cand, dict):
        out: Dict[str, Dict[str, Any]] = {}
        for name, ov in cand.items():
            out[str(name)] = ov or {}
        return out

    if isinstance(cand, list):
        out = {}
        for item in cand:
            if not isinstance(item, dict) or "name" not in item:
                continue
            name = str(item["name"])
            ov = item.get("override", {}) or {}
            out[name] = ov
        return out or {"baseline": {}}

    return {"baseline": {}}


def _choose_heldout_groups(
    *,
    data_path: Path,
    zone_col: str,
    held_out_groups: Optional[List[Union[int, str]]],
    n_groups: Optional[int],
) -> List[Union[int, str]]:
    if held_out_groups:
        return held_out_groups

    if n_groups is None:
        # Default: evaluate the first 10 sites/zones
        n_groups = 10

    df = load_raw_data(str(data_path))
    if zone_col not in df.columns:
        raise KeyError(f"zone_col '{zone_col}' not found in data columns")
    zones = sorted(df[zone_col].dropna().unique().tolist())
    if len(zones) == 0:
        raise ValueError("No zones found in data.")
    return zones[: int(n_groups)]


# -----------------------------
# runner main
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--exp_name", required=True, type=str)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)

    # Optional: run a subset of candidates
    parser.add_argument("--candidates", nargs="*", default=None)

    # Track2 helpers (LOFO)
    parser.add_argument("--held_out_groups", nargs="*", default=None)
    parser.add_argument("--n_groups", type=int, default=None)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )

    cfg = _load_yaml(args.config)
    out_root = _resolve_out_root(cfg)
    reports_root = _resolve_reports_root(cfg)
    data_path = _resolve_data_path(cfg)

    split_type, split_cfg = _resolve_split(cfg)
    logger.info(f"Loaded split config: {split_cfg}")
    logger.info(f"Split type resolved: '{split_type}'")

    # Column names
    cols_cfg = cfg.get("cols", {}) if isinstance(cfg.get("cols", {}), dict) else {}
    target_col = cols_cfg.get("target_col", "target")
    zone_col = cols_cfg.get("zone_col", "zone_id")
    time_col = cols_cfg.get("time_col", "datetime")
    group_col = cols_cfg.get("group_col", None)

    # Sequence (windowing) config
    seq_cfg = cfg.get("seq", {}) if isinstance(cfg.get("seq", {}), dict) else {}
    lookback = int(seq_cfg.get("lookback", 168))
    horizon = int(seq_cfg.get("horizon", 1))
    feature_cols = seq_cfg.get("feature_cols", None)
    include_target_as_input = bool(seq_cfg.get("include_target_as_input", True))
    add_missing_mask = bool(seq_cfg.get("add_missing_mask", True))

    # Training config
    train_cfg = cfg.get("train", {}) if isinstance(cfg.get("train", {}), dict) else {}
    batch_size = int(train_cfg.get("batch_size", 256))
    max_epochs = int(train_cfg.get("max_epochs", 50))
    patience = int(train_cfg.get("patience", 8))
    lr = float(train_cfg.get("lr", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    grad_clip = train_cfg.get("grad_clip", 1.0)
    device = train_cfg.get("device", None)

    # Model config
    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model", {}), dict) else {}
    channels = model_cfg.get("channels", [32, 64, 64])
    kernel_size = int(model_cfg.get("kernel_size", 3))
    dropout = float(model_cfg.get("dropout", 0.2))

    # Probabilistic config (MC Dropout)
    prob_cfg = cfg.get("prob", {}) if isinstance(cfg.get("prob", {}), dict) else {}
    use_mc_dropout = bool(prob_cfg.get("use_mc_dropout", False))
    mc_runs = int(prob_cfg.get("mc_runs", 50))
    mc_quantiles = tuple(prob_cfg.get("mc_quantiles", (0.05, 0.5, 0.95)))

    # Optional zone embedding
    keep_zone = bool(cfg.get("keep_zone", False))

    # YAML may set zone_emb_dim: null when keep_zone=false. Avoid int(None) crash.
    _raw_zone_emb_dim = cfg.get("zone_emb_dim", 8)
    if _raw_zone_emb_dim is None:
        _raw_zone_emb_dim = 8
    zone_emb_dim = int(_raw_zone_emb_dim)

    # Track1 (temporal) split params
    train_cut = split_cfg.get("train_cut", None)
    val_cut = split_cfg.get("val_cut", None)

    # Track2 (LOFO) split params
    val_days = split_cfg.get("val_days", None)
    min_train = int(split_cfg.get("min_train", 1000))
    if group_col is None:
        group_col = split_cfg.get("group_col", None)

    # Candidates
    candidates = _resolve_candidates(cfg)
    if args.candidates:
        allow = set(args.candidates)
        candidates = {k: v for k, v in candidates.items() if k in allow}
        if not candidates:
            raise ValueError(f"No candidates matched {args.candidates}")

    artifact_exp_dir = (out_root / args.exp_name).resolve()
    artifact_exp_dir.mkdir(parents=True, exist_ok=True)
    reports_exp_dir = (reports_root / args.exp_name).resolve()
    reports_exp_dir.mkdir(parents=True, exist_ok=True)

    # Track2 held-out groups (if needed)
    heldout_list: List[Union[int, str]] = []
    if split_type == "lofo":
        parsed = None
        if args.held_out_groups:
            # Try to parse numeric zone ids as int; otherwise keep as str
            tmp = []
            for x in args.held_out_groups:
                try:
                    tmp.append(int(x))
                except Exception:
                    tmp.append(x)
            parsed = tmp
        heldout_list = _choose_heldout_groups(
            data_path=data_path,
            zone_col=zone_col,
            held_out_groups=parsed,
            n_groups=args.n_groups,
        )
        logger.info(f"Track2 held_out_groups resolved: {heldout_list}")

    # Collect per-run records
    runs: List[Dict[str, Any]] = []

    # Run loop
    for cand_name, cand_override in candidates.items():
        # Candidate overrides can modify seq/train/model/keep_zone, etc.
        merged_cfg = _deep_update(
            {
                "seq": seq_cfg,
                "train": train_cfg,
                "model": model_cfg,
                "keep_zone": keep_zone,
                "zone_emb_dim": zone_emb_dim,
                "cols": {"target_col": target_col, "zone_col": zone_col, "time_col": time_col, "group_col": group_col},
                "split": split_cfg,
            },
            cand_override or {},
        )

        # Re-resolve merged values (candidate-level)
        _keep_zone = bool(merged_cfg.get("keep_zone", keep_zone))
        _zone_emb_dim = int(merged_cfg.get("zone_emb_dim", zone_emb_dim))

        _seq = merged_cfg.get("seq", {}) or {}
        _lookback = int(_seq.get("lookback", lookback))
        _horizon = int(_seq.get("horizon", horizon))
        _feature_cols = _seq.get("feature_cols", feature_cols)
        _include_target = bool(_seq.get("include_target_as_input", include_target_as_input))
        _add_mask = bool(_seq.get("add_missing_mask", add_missing_mask))

        _train = merged_cfg.get("train", {}) or {}
        _batch_size = int(_train.get("batch_size", batch_size))
        _max_epochs = int(_train.get("max_epochs", max_epochs))
        _patience = int(_train.get("patience", patience))
        _lr = float(_train.get("lr", lr))
        _weight_decay = float(_train.get("weight_decay", weight_decay))
        _grad_clip = _train.get("grad_clip", grad_clip)
        _device = _train.get("device", device)

        _model = merged_cfg.get("model", {}) or {}
        _channels = _model.get("channels", channels)
        _kernel_size = int(_model.get("kernel_size", kernel_size))
        _dropout = float(_model.get("dropout", dropout))

        # Probabilistic config (candidate override)
        _prob = merged_cfg.get("prob", {}) or {}
        _use_mc_dropout = bool(_prob.get("use_mc_dropout", use_mc_dropout))
        _mc_runs = int(_prob.get("mc_runs", mc_runs))
        _mc_quantiles = tuple(_prob.get("mc_quantiles", mc_quantiles))

        for seed in args.seeds:
            if split_type == "temporal":
                if not train_cut or not val_cut:
                    raise ValueError("Track1 temporal requires split.train_cut and split.val_cut in YAML.")

                out_dir = artifact_exp_dir / cand_name / f"seed_{seed}"
                logger.info(
                    f"Running Track1 temporal (TCN) | candidate={cand_name} | seed={seed} | out_dir={out_dir}"
                )

                result = run_pipeline_any(
                    data_path=str(data_path),
                    out_dir=str(out_dir),
                    train_cut=str(train_cut),
                    val_cut=str(val_cut),
                    lookback=_lookback,
                    horizon=_horizon,
                    feature_cols=_feature_cols,
                    include_target_as_input=_include_target,
                    add_missing_mask=_add_mask,
                    target_col=target_col,
                    zone_col=zone_col,
                    time_col=time_col,
                    keep_zone=_keep_zone,
                    zone_emb_dim=_zone_emb_dim,
                    seed=int(seed),
                    batch_size=_batch_size,
                    max_epochs=_max_epochs,
                    patience=_patience,
                    lr=_lr,
                    weight_decay=_weight_decay,
                    grad_clip=_grad_clip,
                    device=_device,
                    channels=_channels,
                    kernel_size=_kernel_size,
                    dropout=_dropout,
                    use_mc_dropout=_use_mc_dropout,
                    mc_runs=_mc_runs,
                    mc_quantiles=_mc_quantiles,
                )

                runs.append(
                    {
                        "candidate": cand_name,
                        "seed": seed,
                        "val_rmse": result.metrics["val"]["rmse"],
                        "val_mae": result.metrics["val"]["mae"],
                        "val_r2": result.metrics["val"]["r2"],
                        "test_rmse": result.metrics["test"]["rmse"],
                        "test_mae": result.metrics["test"]["mae"],
                        "test_r2": result.metrics["test"]["r2"],
                        "pipeline_out_dir": result.out_dir,
                        "pipeline_model_path": result.model_path,
                        "pipeline_metrics_path": result.metrics_path,
                        "param_overrides": json.dumps(cand_override or {}, ensure_ascii=False),
                        "use_mc_dropout": bool(_use_mc_dropout),
                        "mc_runs": int(_mc_runs),
                        "mc_quantiles": list(_mc_quantiles),
                    }
                )

            else:
                # Track2 LOFO
                if val_days is None:
                    raise ValueError("Track2 lofo requires split.val_days in YAML.")

                for g in heldout_list:
                    out_dir = artifact_exp_dir / cand_name / f"heldout_{g}" / f"seed_{seed}"
                    logger.info(
                        f"Running Track2 LOFO (TCN) | candidate={cand_name} | held_out_group={g} | seed={seed} | out_dir={out_dir}"
                    )

                    result = run_pipeline_any(
                        data_path=str(data_path),
                        out_dir=str(out_dir),
                        held_out_group=g,
                        group_col=group_col,
                        val_days=int(val_days),
                        min_train=int(min_train),
                        lookback=_lookback,
                        horizon=_horizon,
                        feature_cols=_feature_cols,
                        include_target_as_input=_include_target,
                        add_missing_mask=_add_mask,
                        target_col=target_col,
                        zone_col=zone_col,
                        time_col=time_col,
                        keep_zone=_keep_zone,
                        zone_emb_dim=_zone_emb_dim,
                        seed=int(seed),
                        batch_size=_batch_size,
                        max_epochs=_max_epochs,
                        patience=_patience,
                        lr=_lr,
                        weight_decay=_weight_decay,
                        grad_clip=_grad_clip,
                        device=_device,
                        channels=_channels,
                        kernel_size=_kernel_size,
                        dropout=_dropout,
                        use_mc_dropout=_use_mc_dropout,
                        mc_runs=_mc_runs,
                        mc_quantiles=_mc_quantiles,
                    )

                    runs.append(
                        {
                            "candidate": cand_name,
                            "held_out_group": g,
                            "seed": seed,
                            "inner_val_rmse": result.metrics["inner_val"]["rmse"],
                            "inner_val_mae": result.metrics["inner_val"]["mae"],
                            "inner_val_r2": result.metrics["inner_val"]["r2"],
                            "outer_test_rmse": result.metrics["outer_test"]["rmse"],
                            "outer_test_mae": result.metrics["outer_test"]["mae"] if "outer_test_mae" in result.metrics else result.metrics["outer_test"]["mae"],
                            "outer_test_r2": result.metrics["outer_test"]["r2"],
                            "pipeline_out_dir": result.out_dir,
                            "pipeline_model_path": result.model_path,
                            "pipeline_metrics_path": result.metrics_path,
                            "param_overrides": json.dumps(cand_override or {}, ensure_ascii=False),
                            "use_mc_dropout": bool(_use_mc_dropout),
                            "mc_runs": int(_mc_runs),
                            "mc_quantiles": list(_mc_quantiles),
                        }
                    )

    # -----------------------------
    # summarize
    # -----------------------------
    by_candidate: Dict[str, Any] = {}
    for cand_name in candidates.keys():
        cand_runs = [r for r in runs if r.get("candidate") == cand_name]
        if not cand_runs:
            continue

        if split_type == "temporal":
            val_rmse = [float(r["val_rmse"]) for r in cand_runs]
            val_mae = [float(r["val_mae"]) for r in cand_runs]
            val_r2 = [float(r["val_r2"]) for r in cand_runs]
            test_rmse = [float(r["test_rmse"]) for r in cand_runs]
            test_mae = [float(r["test_mae"]) for r in cand_runs]
            test_r2 = [float(r["test_r2"]) for r in cand_runs]

            by_candidate[cand_name] = {
                "n_runs": len(cand_runs),
                "val": {
                    "rmse": _summarize(val_rmse),
                    "mae": _summarize(val_mae),
                    "r2": _summarize(val_r2),
                },
                "test": {
                    "rmse": _summarize(test_rmse),
                    "mae": _summarize(test_mae),
                    "r2": _summarize(test_r2),
                },
            }

        else:
            iv_rmse = [float(r["inner_val_rmse"]) for r in cand_runs]
            iv_mae = [float(r["inner_val_mae"]) for r in cand_runs]
            iv_r2 = [float(r["inner_val_r2"]) for r in cand_runs]
            ot_rmse = [float(r["outer_test_rmse"]) for r in cand_runs]
            ot_mae = [float(r["outer_test_mae"]) for r in cand_runs]
            ot_r2 = [float(r["outer_test_r2"]) for r in cand_runs]

            by_candidate[cand_name] = {
                "n_runs": len(cand_runs),
                "inner_val": {
                    "rmse": _summarize(iv_rmse),
                    "mae": _summarize(iv_mae),
                    "r2": _summarize(iv_r2),
                },
                "outer_test": {
                    "rmse": _summarize(ot_rmse),
                    "mae": _summarize(ot_mae),
                    "r2": _summarize(ot_r2),
                },
            }

    # top-level summary
    summary: Dict[str, Any] = {
        "experiment": args.exp_name,
        "protocol": "track1_temporal" if split_type == "temporal" else "track2_lofo",
        "config": str(Path(args.config).resolve()),
        "keep_zone": bool(keep_zone),
        "seeds": list(args.seeds),
        "n_runs": int(len(runs)),
        "n_groups": (len(heldout_list) if split_type == "lofo" else None),
        "use_mc_dropout": bool(use_mc_dropout),
        "mc_runs": int(mc_runs),
        "mc_quantiles": list(mc_quantiles),
        "runs": runs,
        "by_candidate": by_candidate,
        # Add artifact and reports roots/dirs for traceability
        "artifact_root": str(out_root),
        "reports_root": str(reports_root),
        "artifact_exp_dir": str(artifact_exp_dir),
        "reports_exp_dir": str(reports_exp_dir),
    }

    # also add aggregate at root 
    if split_type == "temporal":
        summary["val"] = {
            "rmse": _summarize([float(r["val_rmse"]) for r in runs]),
            "mae": _summarize([float(r["val_mae"]) for r in runs]),
            "r2": _summarize([float(r["val_r2"]) for r in runs]),
        }
        summary["test"] = {
            "rmse": _summarize([float(r["test_rmse"]) for r in runs]),
            "mae": _summarize([float(r["test_mae"]) for r in runs]),
            "r2": _summarize([float(r["test_r2"]) for r in runs]),
        }
    else:
        summary["val"] = {  # Use inner_val to align with the previous quantile runner's "val"
            "rmse": _summarize([float(r["inner_val_rmse"]) for r in runs]),
            "mae": _summarize([float(r["inner_val_mae"]) for r in runs]),
            "r2": _summarize([float(r["inner_val_r2"]) for r in runs]),
        }
        summary["test"] = {  # Use outer_test to align with the previous quantile runner's "test"
            "rmse": _summarize([float(r["outer_test_rmse"]) for r in runs]),
            "mae": _summarize([float(r["outer_test_mae"]) for r in runs]),
            "r2": _summarize([float(r["outer_test_r2"]) for r in runs]),
        }

    summary_path = reports_exp_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Saved summary -> {summary_path}")

    # Optionally write runs.csv if runs exist
    if runs:
        runs_csv_path = reports_exp_dir / "runs.csv"
        _write_runs_csv(runs_csv_path, runs)
        logger.info(f"Saved runs CSV -> {runs_csv_path}")


if __name__ == "__main__":
    main()