# src/experiments/run_transformer_qr.py
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import time

import numpy as np
import torch
import yaml

from src.pipelines.transformer_qr_pipeline import TransformerQRPipeline

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
    # your project uses cfg["protocol"] to decide split mode
    proto = cfg.get("protocol", None)
    return proto == "track2_lofo_time_val"


def _auto_out_dir(cfg: Dict[str, Any], *, tag: str, seed: int, heldout: Optional[int]) -> Path:
    """
    Track1: reports_dir/<tag>/seed_<seed>
    Track2: reports_dir/<tag>/heldout_<heldout>/seed_<seed>
    """
    reports_dir = _resolve_reports_dir(cfg)
    if heldout is None:
        return ensure_dir(reports_dir / tag / f"seed_{seed}")
    return ensure_dir(reports_dir / tag / f"heldout_{heldout}" / f"seed_{seed}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config")

    # out_dir is optional; if not provided, we auto-generate under paths.reports_dir
    ap.add_argument("--out_dir", default=None, help="Override base output directory (optional)")

    # seeds
    ap.add_argument("--seed", type=int, default=None, help="Single seed (legacy)")
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="Multiple seeds, e.g. 42 43 44")

    # heldouts for track2
    ap.add_argument(
        "--heldouts",
        type=int,
        nargs="+",
        default=None,
        help="Track2 only: list of held_out_group values, e.g. 1 2 3 4",
    )

    ap.add_argument("--device", default=None, help="Override device, e.g. cuda/cpu")
    ap.add_argument("--tag", default=None, help="Run tag used in auto-outdir")

    return ap.parse_args()


def _pick_seeds(args: argparse.Namespace) -> List[int]:
    if args.seeds is not None and len(args.seeds) > 0:
        return [int(s) for s in args.seeds]
    if args.seed is not None:
        return [int(args.seed)]
    return [42]


def _pick_heldouts(args: argparse.Namespace, cfg0: Dict[str, Any]) -> List[Optional[int]]:
    """
    Returns [None] for track1 (no heldout loop).
    For track2: returns heldouts list (from args or yaml).
    """
    if not _is_track2(cfg0):
        return [None]

    if args.heldouts is not None and len(args.heldouts) > 0:
        return [int(h) for h in args.heldouts]

    # fallback to yaml
    split = cfg0.get("split", {}) if isinstance(cfg0.get("split", {}), dict) else {}
    if "held_out_group" not in split:
        raise KeyError("Track2 requires split.held_out_group in YAML or pass --heldouts ...")
    return [int(split["held_out_group"])]


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

    # nested loops: heldout -> seed
    for heldout in heldouts:
        for seed in seeds:
            # clone cfg per run (deep-ish copy)
            cfg: Dict[str, Any] = json.loads(json.dumps(cfg0))

            # override device if given
            if args.device is not None:
                cfg["device"] = str(args.device)
                cfg.setdefault("training", {})
                if isinstance(cfg["training"], dict):
                    cfg["training"]["device"] = str(args.device)

            # set seed
            cfg["seed"] = int(seed)

            # set heldout for track2
            if heldout is not None:
                cfg.setdefault("split", {})
                if not isinstance(cfg["split"], dict):
                    cfg["split"] = {}
                cfg["split"]["held_out_group"] = int(heldout)

            # resolve out_dir
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
            logger.info(f"{prefix} Loading pipeline...")
            pipe = TransformerQRPipeline(cfg)

            logger.info(f"{prefix} Running...")

            start_time = time.time()
            out = pipe.run()
            end_time = time.time()
            train_minutes = (end_time - start_time) / 60

            logger.info(f"{prefix} Training time: {train_minutes:.2f} min")

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
                }
            }
            write_json(out_dir / "runtime.json", runtime)

            # pipe.run() already writes metrics.json/model.pt; this is just a convenient summary
            if isinstance(out, dict) and "metrics" in out and isinstance(out["metrics"], dict):
                write_json(out_dir / "metrics_summary.json", out["metrics"])

            logger.info(f"{prefix} Done.")


if __name__ == "__main__":
    main()