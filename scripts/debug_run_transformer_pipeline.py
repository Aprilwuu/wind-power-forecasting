# scripts/debug_run_transformer_pipeline.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from src.pipelines.transformer_pipeline import TransformerDetPipeline


def load_yaml(p: str | Path) -> dict:
    p = Path(p)
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def flatten_cfg(y: dict, out_dir: str, seed: int) -> dict:
    # ---- required blocks ----
    protocol = y["protocol"]
    data_path = y["data"]["path"]

    cols = y.get("cols", {})
    target_col = cols.get("target_col", "target")
    zone_col = cols.get("zone_col", "zone_id")
    time_col = cols.get("time_col", "datetime")
    group_col = cols.get("group_col", None)

    split = y.get("split", {})
    seq = y.get("seq", {})
    train = y.get("train", {})
    model = y.get("model", {})
    model_block = model.get("transformer", model)  # 支持 model.transformer 或直接 model.xxx

    cfg = {
        # base
        "out_dir": str(Path(out_dir).resolve()),
        "protocol": protocol,
        "seed": int(seed),

        # data
        "data_path": str(Path(data_path).as_posix()),
        "target_col": target_col,
        "zone_col": zone_col,
        "time_col": time_col,

        # seq (扁平)
        "lookback": int(seq.get("lookback", 168)),
        "horizon": int(seq.get("horizon", 1)),
        "feature_cols": seq.get("feature_cols", None),
        "include_target_as_input": bool(seq.get("include_target_as_input", True)),
        "add_missing_mask": bool(seq.get("add_missing_mask", True)),

        # train
        "device": train.get("device", None),
        "batch_size": int(train.get("batch_size", 256)),
        "max_epochs": int(train.get("max_epochs", 20)),
        "patience": int(train.get("patience", 5)),
        "lr": float(train.get("lr", 1e-3)),
        "weight_decay": float(train.get("weight_decay", 0.0)),
        "grad_clip": train.get("grad_clip", 1.0),
        "val_every": int(train.get("val_every", 1)),

        # loss
        "train_loss_name": model.get("train_loss", {}).get("name", "mse"),
        "huber_delta": float(model.get("train_loss", {}).get("huber_delta", 1.0)),

        # keep_zone (你说 track2 肯定要加就设 true/false 都行；这里用 yaml 的值)
        "keep_zone": bool(y.get("keep_zone", False)),
        "zone_emb_dim": int(y.get("zone_emb_dim", 8)),

        # transformer params
        "d_model": int(model_block.get("d_model", 64)),
        "nhead": int(model_block.get("nhead", 4)),
        "num_layers": int(model_block.get("num_layers", 2)),
        "dim_feedforward": int(model_block.get("dim_feedforward", 128)),
        "dropout": float(model_block.get("dropout", 0.1)),
    }

    # ---- split: track1 / track2 ----
    if protocol == "track1_temporal":
        cfg["train_cut"] = split["train_cut"]
        cfg["val_cut"] = split["val_cut"]
        cfg["split"] = {"train_cut": cfg["train_cut"], "val_cut": cfg["val_cut"]}
    elif protocol == "track2_lofo_time_val":
        cfg["held_out_group"] = split["held_out_group"]
        cfg["val_days"] = int(split["val_days"])
        cfg["min_train"] = int(split.get("min_train", 1000))
        cfg["group_col"] = split.get("group_col", group_col) or zone_col
        cfg["split"] = {
            "held_out_group": cfg["held_out_group"],
            "val_days": cfg["val_days"],
            "min_train": cfg["min_train"],
            "group_col": cfg["group_col"],
        }
    else:
        raise ValueError(f"Unknown protocol: {protocol}")

    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    y = load_yaml(args.config)
    cfg = flatten_cfg(y, out_dir=args.out_dir, seed=args.seed)

    pipe = TransformerDetPipeline(cfg)
    out = pipe.run()

    print(json.dumps({
        "out_dir": cfg["out_dir"],
        "protocol": cfg["protocol"],
        "seed": cfg["seed"],
        "val_rmse": out["metrics"]["val"]["rmse"],
        "test_rmse": out["metrics"]["test"]["rmse"],
        "keep_zone": out["metrics"]["keep_zone"],
    }, indent=2))


if __name__ == "__main__":
    main()
