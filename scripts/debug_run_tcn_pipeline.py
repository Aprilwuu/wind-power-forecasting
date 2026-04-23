# scripts/debug_run_tcn_pipeline.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import yaml

from src.pipelines.tcn_forecast import TCNForecastPipeline


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override into base (override wins)."""
    out = dict(base)
    for k, v in (override or {}).items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def get_dict(cfg: Dict[str, Any], *keys: str) -> Dict[str, Any]:
    for k in keys:
        v = cfg.get(k)
        if isinstance(v, dict):
            return v
    return {}


def resolve_protocol_and_split(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert your YAML into the minimal flat cfg that ForecastBasePipeline expects:
      - protocol: "track1_temporal" or "track2_lofo_time_val"
      - split fields:
          track1: train_cut, val_cut
          track2: held_out_group, val_days, group_col, min_train
    """
    split_cfg = get_dict(cfg, "split")
    raw = (split_cfg.get("type") or split_cfg.get("split_type") or "temporal").lower()

    # columns
    cols = get_dict(cfg, "cols")
    target_col = cols.get("target_col", "target")
    zone_col = cols.get("zone_col", "zone_id")
    time_col = cols.get("time_col", "datetime")
    group_col_default = cols.get("group_col", None)

    if raw in ("temporal", "track1", "track1_temporal"):
        return {
            "protocol": "track1_temporal",
            "split": {"train_cut": split_cfg.get("train_cut"), "val_cut": split_cfg.get("val_cut")},
            "train_cut": split_cfg.get("train_cut"),
            "val_cut": split_cfg.get("val_cut"),
            "target_col": target_col,
            "zone_col": zone_col,
            "time_col": time_col,
        }

    if raw in ("lofo", "track2", "track2_lofo"):
      
        return {
            "protocol": "track2_lofo_time_val",
            "split": {
                "held_out_group": split_cfg.get("held_out_group"),
                "val_days": split_cfg.get("val_days"),
                "min_train": split_cfg.get("min_train", 1000),
                "group_col": split_cfg.get("group_col") or group_col_default or zone_col,
            },
            "held_out_group": split_cfg.get("held_out_group"),
            "val_days": split_cfg.get("val_days"),
            "min_train": int(split_cfg.get("min_train", 1000)),
            "group_col": split_cfg.get("group_col") or group_col_default or zone_col,
            "target_col": target_col,
            "zone_col": zone_col,
            "time_col": time_col,
        }

    raise ValueError(f"Unknown split.type: {raw}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str, help="Path to YAML config")
    ap.add_argument("--out_dir", default="reports/debug_tcn", type=str, help="Output directory for this debug run")
    ap.add_argument("--seed", default=42, type=int)
    ap.add_argument("--held_out_group", default=None, help="Override Track2 held_out_group (optional)")
    args = ap.parse_args()

    cfg_yaml = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    # ---------- resolve blocks ----------
    data_block = get_dict(cfg_yaml, "data")
    seq_block = get_dict(cfg_yaml, "seq")
    train_block = get_dict(cfg_yaml, "training", "train")
    model_block = get_dict(cfg_yaml, "model")
    prob_block = get_dict(cfg_yaml, "prob")
    zone_block = get_dict(cfg_yaml, "zone")

    # required
    data_path = data_block.get("path") or cfg_yaml.get("data_path")
    if not data_path:
        raise KeyError("YAML must provide data.path or data_path")

    # protocol + split
    base = resolve_protocol_and_split(cfg_yaml)

    # override held_out_group for Track2 if provided
    if base["protocol"] == "track2_lofo_time_val" and args.held_out_group is not None:
        base["held_out_group"] = args.held_out_group
        base["split"]["held_out_group"] = args.held_out_group

    # ---------- flatten to ForecastBasePipeline cfg ----------
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    keep_zone = bool(cfg_yaml.get("keep_zone", zone_block.get("keep_zone", False)))
    zone_emb_dim = int(cfg_yaml.get("zone_emb_dim", zone_block.get("zone_emb_dim", 8)) or 8)

    tcn_block = model_block.get("tcn") if isinstance(model_block.get("tcn"), dict) else model_block
    channels = tcn_block.get("channels", [32, 64, 64])
    kernel_size = int(tcn_block.get("kernel_size", 3))
    dropout = float(tcn_block.get("dropout", 0.2))

    # loss
    loss_cfg = model_block.get("train_loss", {}) if isinstance(model_block.get("train_loss"), dict) else {}
    loss_name = str(loss_cfg.get("name", "mse"))
    huber_delta = float(loss_cfg.get("huber_delta", 1.0))

    # training
    flat_cfg: Dict[str, Any] = {
        "out_dir": str(out_dir),
        "data_path": str(Path(data_path).expanduser().resolve()),
        "protocol": base["protocol"],
        "split": base.get("split", {}),
        "seed": int(args.seed),
        # cols
        "target_col": base["target_col"],
        "zone_col": base["zone_col"],
        "time_col": base["time_col"],
        "group_col": base.get("group_col", None),
        # seq
        "lookback": int(seq_block.get("lookback", 168)),
        "horizon": int(seq_block.get("horizon", 1)),
        "feature_cols": seq_block.get("feature_cols", None),
        "include_target_as_input": bool(seq_block.get("include_target_as_input", True)),
        "add_missing_mask": bool(seq_block.get("add_missing_mask", True)),
        # split args
        "train_cut": base.get("train_cut", None),
        "val_cut": base.get("val_cut", None),
        "held_out_group": base.get("held_out_group", None),
        "val_days": base.get("val_days", None),
        "min_train": base.get("min_train", 1000),
        # loader
        "batch_size": int(train_block.get("batch_size", 256)),
        "num_workers": int(train_block.get("num_workers", 0)),
        "pin_memory": bool(train_block.get("pin_memory", True)),
        # train
        "max_epochs": int(train_block.get("max_epochs", 50)),
        "patience": int(train_block.get("patience", 8)),
        "lr": float(train_block.get("lr", 1e-3)),
        "weight_decay": float(train_block.get("weight_decay", 0.0)),
        "grad_clip": train_block.get("grad_clip", 1.0),
        "device": train_block.get("device", None),
        "train_loss_name": loss_name,
        "huber_delta": huber_delta,
        # model
        "channels": list(channels),
        "kernel_size": int(kernel_size),
        "dropout": float(dropout),
        # zone
        "keep_zone": bool(keep_zone),
        "zone_emb_dim": int(zone_emb_dim),
        # prob
        "use_mc_dropout": bool(prob_block.get("use_mc_dropout", False)),
        "mc_runs": int(prob_block.get("mc_runs", 50)),
        "mc_quantiles": list(prob_block.get("mc_quantiles", (0.05, 0.5, 0.95))),
    }

    # sanity checks
    if flat_cfg["protocol"] == "track1_temporal":
        if not flat_cfg["train_cut"] or not flat_cfg["val_cut"]:
            raise ValueError("Track1 requires split.train_cut and split.val_cut in YAML.")
    if flat_cfg["protocol"] == "track2_lofo_time_val":
        if flat_cfg["held_out_group"] is None:
            raise ValueError("Track2 requires split.held_out_group (or pass --held_out_group).")
        if flat_cfg["val_days"] is None:
            raise ValueError("Track2 requires split.val_days in YAML.")
        if not flat_cfg["group_col"]:
            flat_cfg["group_col"] = flat_cfg["zone_col"]

    # dump final cfg for debugging
    (out_dir / "debug_flat_cfg.json").write_text(json.dumps(flat_cfg, indent=2), encoding="utf-8")

    pipe = TCNForecastPipeline(flat_cfg)
    out = pipe.run()

    # write a small run summary
    summary = {
        "out_dir": str(out_dir),
        "protocol": flat_cfg["protocol"],
        "seed": flat_cfg["seed"],
        "val_rmse": out["metrics"]["val"]["rmse"],
        "test_rmse": out["metrics"]["test"]["rmse"],
        "use_mc_dropout": bool(flat_cfg["use_mc_dropout"]),
    }
    (out_dir / "debug_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
