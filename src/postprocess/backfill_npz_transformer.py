import argparse
import json
from pathlib import Path

import torch

from src.pipelines.transformer_qr_pipeline import TransformerQRPipeline


def load_runtime_cfg(run_dir: Path) -> dict:
    snap = run_dir / "config_snapshot.json"
    if not snap.exists():
        raise FileNotFoundError(f"Missing config_snapshot.json in {run_dir}")
    d = json.loads(snap.read_text(encoding="utf-8"))
    cfg = d["runtime"]
    cfg["out_dir"] = str(run_dir)  # IMPORTANT: make sure outputs go back into this run dir
    return cfg


@torch.no_grad()
def export_npz_for_run(run_dir: Path) -> None:
    run_dir = Path(run_dir)

    model_path = run_dir / "model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model.pt in {run_dir}")

    cfg = load_runtime_cfg(run_dir)

    # build pipeline + data
    pipe = TransformerQRPipeline(cfg)
    data_art = pipe._build_data(cfg)

    # build model + load weights
    model = pipe.build_model(data_art.input_dim, data_art.output_dim, data_art).to(pipe.device)
    ckpt = torch.load(str(model_path), map_location=pipe.device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # run val/test interval inference
    y_val, _ = pipe._predict_point(model, data_art.val_loader)
    y_test, _ = pipe._predict_point(model, data_art.test_loader)

    coverage = float(cfg.get("interval_coverage", 0.9))
    lo_val, hi_val = pipe._predict_interval(model, data_art.val_loader, coverage=coverage)
    lo_test, hi_test = pipe._predict_interval(model, data_art.test_loader, coverage=coverage)

    # naming consistent with your batch_conformal command
    protocol = str(cfg.get("protocol", ""))
    if protocol == "track2_lofo_time_val":
        val_name = "preds_outer_val_transformer_qr.npz"
        test_name = "preds_outer_test_transformer_qr.npz"
    else:
        val_name = "preds_val_transformer_qr.npz"
        test_name = "preds_test_transformer_qr.npz"

    # use the saver you added to forecast_base.py
    from src.pipelines.forecast_base import save_interval_npz

    save_interval_npz(run_dir / val_name, y_val, lo_val, hi_val)
    save_interval_npz(run_dir / test_name, y_test, lo_test, hi_test)

    print(f"[OK] {run_dir}")
    print(f"     saved: {val_name}")
    print(f"     saved: {test_name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", type=str, required=True,
                    help="e.g. .../transformer_qr_track2_lb168/baseline")
    ap.add_argument("--track", type=int, choices=[1, 2], default=2)
    args = ap.parse_args()

    base = Path(args.base_dir)

    if args.track == 2:
        run_dirs = sorted(base.glob("heldout_*/seed_*"))
    else:
        run_dirs = sorted(base.glob("seed_*"))

    if not run_dirs:
        raise FileNotFoundError(f"No run dirs found under {base}")

    n_ok, n_skip, n_fail = 0, 0, 0
    for rd in run_dirs:
        if not (rd / "model.pt").exists():
            n_skip += 1
            continue
        try:
            export_npz_for_run(rd)
            n_ok += 1
        except Exception as e:
            print(f"[FAIL] {rd}: {e}")
            n_fail += 1

    print(f"\nDone. OK={n_ok}, SKIP(no model)={n_skip}, FAIL={n_fail}")


if __name__ == "__main__":
    main()