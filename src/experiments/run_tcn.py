# src/experiments/run_tcn.py
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import yaml

from src.data.load import load_raw_data
from src.pipelines.tcn_pipeline import run_pipeline_any
from src.utils.conformal_widening import conformal_widen_exp_dir

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
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
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
    paths = cfg.get("paths", {}) if isinstance(cfg.get("paths", {}), dict) else {}
    out_root = paths.get("featured_dir") or cfg.get("out_root") or "data/featured"
    return Path(out_root).expanduser().resolve()


def _resolve_reports_root(cfg: Dict[str, Any]) -> Path:
    paths = cfg.get("paths", {}) if isinstance(cfg.get("paths", {}), dict) else {}
    reports_root = paths.get("reports_dir") or cfg.get("reports_root") or "reports/experiments"
    return Path(reports_root).expanduser().resolve()


def _write_runs_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    all_keys = set()
    for row in rows:
        all_keys.update(row.keys())
    columns = sorted(all_keys)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            row_filled = {k: row.get(k, "") for k in columns}
            writer.writerow(row_filled)


def _resolve_data_path(cfg: Dict[str, Any]) -> Path:
    if isinstance(cfg.get("data"), dict) and cfg["data"].get("path"):
        return Path(cfg["data"]["path"]).expanduser().resolve()
    if cfg.get("data_path"):
        return Path(cfg["data_path"]).expanduser().resolve()
    raise KeyError("Config must provide data.path or data_path")


def _resolve_split(cfg: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    split_cfg = cfg.get("split", {}) if isinstance(cfg.get("split", {}), dict) else {}
    raw = (split_cfg.get("type") or split_cfg.get("split_type") or "temporal").lower()

    if raw in ("temporal", "track1", "track1_temporal"):
        return "temporal", split_cfg
    if raw in ("lofo", "track2", "track2_lofo"):
        return "lofo", split_cfg

    # Fallback
    if split_cfg.get("train_cut") and split_cfg.get("val_cut"):
        return "temporal", split_cfg
    if split_cfg.get("val_days") or split_cfg.get("held_out_group"):
        return "lofo", split_cfg

    raise ValueError(f"Unrecognized split.type: {raw}")


def _resolve_candidates(cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
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
        n_groups = 10

    df = load_raw_data(str(data_path))
    if zone_col not in df.columns:
        raise KeyError(f"zone_col '{zone_col}' not found in data columns")
    zones = sorted(df[zone_col].dropna().unique().tolist())
    if len(zones) == 0:
        raise ValueError("No zones found in data.")
    return zones[: int(n_groups)]


def _get_dict(cfg: Dict[str, Any], *keys: str) -> Dict[str, Any]:
    """Return the first existing dict among keys."""
    for k in keys:
        v = cfg.get(k, None)
        if isinstance(v, dict):
            return v
    return {}


def _get_model_tcn_block(model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Support:
      model:
        tcn: {...}
    OR older:
      model: {channels, kernel_size, dropout}
    """
    tcn = model_cfg.get("tcn", None)
    if isinstance(tcn, dict):
        return tcn
    return model_cfg


def _resolve_loss(model_cfg: Dict[str, Any]) -> Tuple[str, float]:
    """Support:
      model:
        train_loss: {name: huber, huber_delta: 1.0}
    Returns (loss_name, huber_delta)
    """
    loss_cfg = model_cfg.get("train_loss", None)
    if not isinstance(loss_cfg, dict):
        return "mse", 1.0
    name = str(loss_cfg.get("name", "mse")).lower().strip()
    delta = float(loss_cfg.get("huber_delta", 1.0))
    return name, delta


def _qcol(q: float) -> str:
    # 0.05 -> q05, 0.95 -> q95
    return f"q{int(round(float(q) * 100)):02d}"


def _run_conformal_if_needed(
    *,
    out_dir: str,
    target_picp: float,
    use_mc_dropout: bool,
    mc_quantiles: Tuple[float, ...],
) -> Dict[str, Any]:
    """
    Returns a dict with keys:
      conformal_used, conformal_t, conformal_target_picp,
      raw_val_picp, raw_test_picp, cal_val_picp, cal_test_picp,
      cal_val_out, cal_test_out, lo_col, hi_col, conformal_error
    """
    base: Dict[str, Any] = {
        "conformal_used": False,
        "conformal_t": "",
        "conformal_target_picp": "",
        "raw_val_picp": "",
        "raw_test_picp": "",
        "cal_val_picp": "",
        "cal_test_picp": "",
        "cal_val_out": "",
        "cal_test_out": "",
        "lo_col": "",
        "hi_col": "",
        "conformal_error": "",
    }
    if not use_mc_dropout:
        return base

    lo_q = float(min(mc_quantiles))
    hi_q = float(max(mc_quantiles))
    lo_col = _qcol(lo_q)
    hi_col = _qcol(hi_q)

    base["lo_col"] = lo_col
    base["hi_col"] = hi_col
    base["conformal_used"] = True
    base["conformal_target_picp"] = float(target_picp)

    try:
        cal = conformal_widen_exp_dir(
            out_dir,
            target_picp=float(target_picp),
            y_col="y_true",
            lo_col=lo_col,
            hi_col=hi_col,
        )
        base["conformal_t"] = float(cal.get("t", math.nan))
        base["raw_val_picp"] = float(cal["raw"]["val_picp"])
        base["raw_test_picp"] = float(cal["raw"]["test_picp"])
        base["cal_val_picp"] = float(cal["cal"]["val_picp"])
        base["cal_test_picp"] = float(cal["cal"]["test_picp"])
        base["cal_val_out"] = str(cal.get("val_out", ""))
        base["cal_test_out"] = str(cal.get("test_out", ""))
        return base
    except Exception as e:
        base["conformal_error"] = repr(e)
        return base


# -----------------------------
# runner main
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--exp_name", required=True, type=str)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)

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
    cols_cfg = _get_dict(cfg, "cols")
    target_col = cols_cfg.get("target_col", "target")
    zone_col = cols_cfg.get("zone_col", "zone_id")
    time_col = cols_cfg.get("time_col", "datetime")
    group_col = cols_cfg.get("group_col", None)

    # Sequence config
    seq_cfg = _get_dict(cfg, "seq")
    lookback = int(seq_cfg.get("lookback", 168))
    horizon = int(seq_cfg.get("horizon", 1))
    feature_cols = seq_cfg.get("feature_cols", None)
    include_target_as_input = bool(seq_cfg.get("include_target_as_input", True))
    add_missing_mask = bool(seq_cfg.get("add_missing_mask", True))

    # Training config (support training: or train:)
    train_cfg = _get_dict(cfg, "training", "train")
    batch_size = int(train_cfg.get("batch_size", 256))
    max_epochs = int(train_cfg.get("max_epochs", 50))
    patience = int(train_cfg.get("patience", 8))
    lr = float(train_cfg.get("lr", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    grad_clip = train_cfg.get("grad_clip", 1.0)
    device = train_cfg.get("device", None)

    # Model config
    model_cfg = _get_dict(cfg, "model")
    tcn_cfg = _get_model_tcn_block(model_cfg)
    channels = tcn_cfg.get("channels", [32, 64, 64])
    kernel_size = int(tcn_cfg.get("kernel_size", 3))
    dropout = float(tcn_cfg.get("dropout", 0.2))

    # Loss config (from model.train_loss)
    loss_name, huber_delta = _resolve_loss(model_cfg)

    # Probabilistic config
    prob_cfg = _get_dict(cfg, "prob")
    use_mc_dropout = bool(prob_cfg.get("use_mc_dropout", False))
    mc_runs = int(prob_cfg.get("mc_runs", 50))
    mc_quantiles = tuple(prob_cfg.get("mc_quantiles", (0.05, 0.5, 0.95)))
    target_picp = float(prob_cfg.get("target_picp", 0.9))  # <= NEW

    # Zone embedding (support both top-level keep_zone/zone_emb_dim and nested zone:)
    keep_zone = bool(cfg.get("keep_zone", cfg.get("zone", {}).get("keep_zone", False)))
    _raw_zone_emb_dim = cfg.get("zone_emb_dim", cfg.get("zone", {}).get("zone_emb_dim", 8))
    if _raw_zone_emb_dim is None:
        _raw_zone_emb_dim = 8
    zone_emb_dim = int(_raw_zone_emb_dim)

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

    # Track2 held-out groups (based on base cfg; candidate override can still filter via args.held_out_groups)
    heldout_list: List[Union[int, str]] = []
    if split_type == "lofo":
        parsed = None
        if args.held_out_groups:
            tmp: List[Union[int, str]] = []
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
        # Normalize for override (candidate can override split/seq/train/model/prob/etc.)
        merged_cfg = _deep_update(
            {
                "seq": seq_cfg,
                "train": train_cfg,  # canonical key
                "model": model_cfg,
                "prob": prob_cfg,
                "keep_zone": keep_zone,
                "zone_emb_dim": zone_emb_dim,
                "cols": {
                    "target_col": target_col,
                    "zone_col": zone_col,
                    "time_col": time_col,
                    "group_col": group_col,
                },
                "split": split_cfg,
            },
            cand_override or {},
        )

        # Candidate-level resolve
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
        _tcn = _get_model_tcn_block(_model)
        _channels = _tcn.get("channels", channels)
        _kernel_size = int(_tcn.get("kernel_size", kernel_size))
        _dropout = float(_tcn.get("dropout", dropout))
        _loss_name, _huber_delta = _resolve_loss(_model)

        _prob = merged_cfg.get("prob", {}) or {}
        _use_mc_dropout = bool(_prob.get("use_mc_dropout", use_mc_dropout))
        _mc_runs = int(_prob.get("mc_runs", mc_runs))
        _mc_quantiles = tuple(_prob.get("mc_quantiles", mc_quantiles))
        _target_picp = float(_prob.get("target_picp", target_picp))

        _split = merged_cfg.get("split", {}) or {}

        # IMPORTANT FIX: split values should also be override-able per candidate
        _train_cut = _split.get("train_cut", None)
        _val_cut = _split.get("val_cut", None)
        _val_days = _split.get("val_days", None)
        _min_train = int(_split.get("min_train", 1000))
        _group_col = _split.get("group_col", None) or group_col

        for seed in args.seeds:
            if split_type == "temporal":
                if not _train_cut or not _val_cut:
                    raise ValueError("Track1 temporal requires split.train_cut and split.val_cut in YAML (or candidate override).")

                out_dir = artifact_exp_dir / cand_name / f"seed_{seed}"
                logger.info(
                    f"Running Track1 temporal (TCN) | candidate={cand_name} | seed={seed} | out_dir={out_dir}"
                )

                result = run_pipeline_any(
                    data_path=str(data_path),
                    out_dir=str(out_dir),
                    train_cut=str(_train_cut),
                    val_cut=str(_val_cut),
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
                    loss_name=_loss_name,
                    huber_delta=_huber_delta,
                    channels=_channels,
                    kernel_size=_kernel_size,
                    dropout=_dropout,
                    use_mc_dropout=_use_mc_dropout,
                    mc_runs=_mc_runs,
                    mc_quantiles=_mc_quantiles,
                )

                conf = _run_conformal_if_needed(
                    out_dir=result.out_dir,
                    target_picp=_target_picp,
                    use_mc_dropout=_use_mc_dropout,
                    mc_quantiles=_mc_quantiles,
                )
                if conf["conformal_used"] and not conf["conformal_error"]:
                    logger.info(
                        f"[conformal][track1] lo={conf['lo_col']} hi={conf['hi_col']} | t={conf['conformal_t']:.6f} "
                        f"| raw_val={conf['raw_val_picp']:.3f} raw_test={conf['raw_test_picp']:.3f} "
                        f"| cal_val={conf['cal_val_picp']:.3f} cal_test={conf['cal_test_picp']:.3f}"
                    )
                elif conf["conformal_used"] and conf["conformal_error"]:
                    logger.warning(f"[conformal][track1] failed: {conf['conformal_error']}")

                row = {
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
                    "target_picp": float(_target_picp),
                    "loss_name": _loss_name,
                    "huber_delta": float(_huber_delta),
                    # conformal fields
                    **conf,
                }
                runs.append(row)

            else:
                # Track2 LOFO
                if _val_days is None:
                    raise ValueError("Track2 lofo requires split.val_days in YAML (or candidate override).")
                if not _group_col:
                    raise ValueError("Track2 lofo requires split.group_col (or cols.group_col).")

                for g in heldout_list:
                    out_dir = artifact_exp_dir / cand_name / f"heldout_{g}" / f"seed_{seed}"
                    logger.info(
                        f"Running Track2 LOFO (TCN) | candidate={cand_name} | held_out_group={g} | seed={seed} | out_dir={out_dir}"
                    )

                    result = run_pipeline_any(
                        data_path=str(data_path),
                        out_dir=str(out_dir),
                        held_out_group=g,
                        group_col=_group_col,
                        val_days=int(_val_days),
                        min_train=int(_min_train),
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
                        loss_name=_loss_name,
                        huber_delta=_huber_delta,
                        channels=_channels,
                        kernel_size=_kernel_size,
                        dropout=_dropout,
                        use_mc_dropout=_use_mc_dropout,
                        mc_runs=_mc_runs,
                        mc_quantiles=_mc_quantiles,
                    )

                    conf = _run_conformal_if_needed(
                        out_dir=result.out_dir,
                        target_picp=_target_picp,
                        use_mc_dropout=_use_mc_dropout,
                        mc_quantiles=_mc_quantiles,
                    )
                    if conf["conformal_used"] and not conf["conformal_error"]:
                        logger.info(
                            f"[conformal][track2] heldout={g} | t={conf['conformal_t']:.6f} "
                            f"| raw_val={conf['raw_val_picp']:.3f} raw_test={conf['raw_test_picp']:.3f} "
                            f"| cal_val={conf['cal_val_picp']:.3f} cal_test={conf['cal_test_picp']:.3f}"
                        )
                    elif conf["conformal_used"] and conf["conformal_error"]:
                        logger.warning(f"[conformal][track2] heldout={g} failed: {conf['conformal_error']}")

                    row = {
                        "candidate": cand_name,
                        "held_out_group": g,
                        "seed": seed,
                        "inner_val_rmse": result.metrics["inner_val"]["rmse"],
                        "inner_val_mae": result.metrics["inner_val"]["mae"],
                        "inner_val_r2": result.metrics["inner_val"]["r2"],
                        "outer_test_rmse": result.metrics["outer_test"]["rmse"],
                        "outer_test_mae": result.metrics["outer_test"]["mae"],
                        "outer_test_r2": result.metrics["outer_test"]["r2"],
                        "pipeline_out_dir": result.out_dir,
                        "pipeline_model_path": result.model_path,
                        "pipeline_metrics_path": result.metrics_path,
                        "param_overrides": json.dumps(cand_override or {}, ensure_ascii=False),
                        "use_mc_dropout": bool(_use_mc_dropout),
                        "mc_runs": int(_mc_runs),
                        "mc_quantiles": list(_mc_quantiles),
                        "target_picp": float(_target_picp),
                        "loss_name": _loss_name,
                        "huber_delta": float(_huber_delta),
                        # conformal fields
                        **conf,
                    }
                    runs.append(row)

    # -----------------------------
    # summarize
    # -----------------------------
    by_candidate: Dict[str, Any] = {}
    for cand_name in candidates.keys():
        cand_runs = [r for r in runs if r.get("candidate") == cand_name]
        if not cand_runs:
            continue

        def _safe_float_list(key: str) -> List[float]:
            xs = []
            for r in cand_runs:
                v = r.get(key, "")
                if v == "" or v is None:
                    continue
                try:
                    xs.append(float(v))
                except Exception:
                    continue
            return xs

        if split_type == "temporal":
            by_candidate[cand_name] = {
                "n_runs": len(cand_runs),
                "val": {
                    "rmse": _summarize(_safe_float_list("val_rmse")),
                    "mae": _summarize(_safe_float_list("val_mae")),
                    "r2": _summarize(_safe_float_list("val_r2")),
                },
                "test": {
                    "rmse": _summarize(_safe_float_list("test_rmse")),
                    "mae": _summarize(_safe_float_list("test_mae")),
                    "r2": _summarize(_safe_float_list("test_r2")),
                },
                "conformal": {
                    "t": _summarize(_safe_float_list("conformal_t")),
                    "raw_val_picp": _summarize(_safe_float_list("raw_val_picp")),
                    "raw_test_picp": _summarize(_safe_float_list("raw_test_picp")),
                    "cal_val_picp": _summarize(_safe_float_list("cal_val_picp")),
                    "cal_test_picp": _summarize(_safe_float_list("cal_test_picp")),
                },
            }
        else:
            by_candidate[cand_name] = {
                "n_runs": len(cand_runs),
                "inner_val": {
                    "rmse": _summarize(_safe_float_list("inner_val_rmse")),
                    "mae": _summarize(_safe_float_list("inner_val_mae")),
                    "r2": _summarize(_safe_float_list("inner_val_r2")),
                },
                "outer_test": {
                    "rmse": _summarize(_safe_float_list("outer_test_rmse")),
                    "mae": _summarize(_safe_float_list("outer_test_mae")),
                    "r2": _summarize(_safe_float_list("outer_test_r2")),
                },
                "conformal": {
                    "t": _summarize(_safe_float_list("conformal_t")),
                    "raw_val_picp": _summarize(_safe_float_list("raw_val_picp")),
                    "raw_test_picp": _summarize(_safe_float_list("raw_test_picp")),
                    "cal_val_picp": _summarize(_safe_float_list("cal_val_picp")),
                    "cal_test_picp": _summarize(_safe_float_list("cal_test_picp")),
                },
            }

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
        "target_picp": float(target_picp),
        "runs": runs,
        "by_candidate": by_candidate,
        "artifact_root": str(out_root),
        "reports_root": str(reports_root),
        "artifact_exp_dir": str(artifact_exp_dir),
        "reports_exp_dir": str(reports_exp_dir),
    }

    summary_path = reports_exp_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Saved summary -> {summary_path}")

    if runs:
        runs_csv_path = reports_exp_dir / "runs.csv"
        _write_runs_csv(runs_csv_path, runs)
        logger.info(f"Saved runs CSV -> {runs_csv_path}")


if __name__ == "__main__":
    main()
