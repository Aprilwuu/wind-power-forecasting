from __future__ import annotations
import argparse
import yaml
import torch
import numpy as np

from src.pipelines.tcn_forecast import TCNForecastPipeline

def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def flatten_cfg(y: dict, out_dir: str, seed: int):
    # mimic your debug runner's mapping (minimal needed keys)
    cfg = {}
    cfg["out_dir"] = out_dir
    cfg["seed"] = seed
    cfg["protocol"] = y["protocol"]

    # data
    cfg["data_path"] = y["data"]["path"]

    # cols
    cols = y.get("cols", {})
    cfg["target_col"] = cols.get("target_col", "target")
    cfg["zone_col"] = cols.get("zone_col", "zone_id")
    cfg["time_col"] = cols.get("time_col", "datetime")

    # split
    split = y.get("split", {})
    cfg["train_cut"] = split.get("train_cut")
    cfg["val_cut"] = split.get("val_cut")

    # seq
    seq = y.get("seq", {})
    cfg["lookback"] = int(seq["lookback"])
    cfg["horizon"] = int(seq.get("horizon", 1))
    cfg["feature_cols"] = seq.get("feature_cols", None)
    cfg["include_target_as_input"] = bool(seq.get("include_target_as_input", True))
    cfg["add_missing_mask"] = bool(seq.get("add_missing_mask", True))

    # train
    tr = y.get("train", {})
    cfg["device"] = tr.get("device", None)
    cfg["batch_size"] = int(tr.get("batch_size", 256))
    cfg["num_workers"] = int(tr.get("num_workers", 0))
    cfg["pin_memory"] = bool(tr.get("pin_memory", True))

    return cfg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    y = load_yaml(args.config)
    cfg = flatten_cfg(y, args.out_dir, args.seed)

    pipe = TCNForecastPipeline(cfg)
    data_art = pipe._build_data(cfg)

    batch = next(iter(data_art.train_loader))
    x, y = batch

    print("=== batch preview ===")
    print("X:", x.shape, x.dtype, x.device)
    print("y:", y.shape, y.dtype, y.device)

    # check last feature channel for mask-like values
    last = x[..., -1]
    print("\n=== last feature stats (possible mask) ===")
    print("min/max:", float(last.min()), float(last.max()))
    uniq = torch.unique(last[:2000].round()).cpu().numpy()
    print("unique (rounded, sample<=2000*L):", uniq[:20], " ... total:", len(uniq))

    # also show first timestep last feature for first few samples
    print("\nfirst 10 samples, last feature at t=0:", last[:10, 0].cpu().numpy())

if __name__ == "__main__":
    main()
