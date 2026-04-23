from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Dict

import numpy as np


# -----------------------------
# Discovery
# -----------------------------
def find_seed_files(root: Path, filename: str) -> List[Path]:
    """
    Find files matching: root/**/seed_*/**/filename

    Returns one file per seed folder. If multiple matches exist within one seed folder,
    prefer the one whose path contains 'postprocess'.
    """
    seed_dirs = sorted([p for p in root.rglob("seed_*") if p.is_dir()])
    files: List[Path] = []

    for sd in seed_dirs:
        hits = sorted(sd.rglob(filename))
        if len(hits) == 0:
            continue
        if len(hits) > 1:
            post_hits = [h for h in hits if "postprocess" in str(h).lower()]
            files.append(post_hits[0] if len(post_hits) > 0 else hits[0])
        else:
            files.append(hits[0])

    return files


# -----------------------------
# Loading helpers
# -----------------------------
def squeeze_1d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim == 2 and a.shape[1] == 1:
        return a[:, 0]
    if a.ndim == 1:
        return a
    raise ValueError(f"Expected 1D or Nx1, got shape={a.shape}")


# -----------------------------
# Aggregation core
# -----------------------------
def aggregate_npz(files: List[Path], out_path: Path, keys: List[str], y_key: str) -> None:
    """
    Aggregate multiple seed NPZs by averaging arrays across seeds.

    Robust to different naming conventions:
      - q05_cal <-> lower_cal
      - q95_cal <-> upper_cal
      - q05     <-> lower
      - q95     <-> upper

    Also supports synthesizing q50 if missing, using the midpoint of bounds:
      q50 = 0.5 * (lo + hi)
    Prefer calibrated bounds when synthesizing q50.
    """
    if len(files) == 0:
        raise FileNotFoundError("No seed files found to aggregate.")

    z0 = np.load(files[0], allow_pickle=True)
    if y_key not in z0.files:
        raise KeyError(f"y_key='{y_key}' not found in {files[0]}. Available keys: {list(z0.files)}")
    y = squeeze_1d(z0[y_key])

    def has(z, k: str) -> bool:
        return k in z.files

    def get(z, k: str) -> np.ndarray:
        return squeeze_1d(z[k]).astype(float)

    # Alias candidates (prefer calibrated)
    ALIAS = {
        "q05_cal": ["q05_cal", "lower_cal"],
        "q95_cal": ["q95_cal", "upper_cal"],
        "q05":     ["q05", "lower"],
        "q95":     ["q95", "upper"],
    }

    def fetch_with_alias(z, key: str) -> np.ndarray | None:
        """Return array if key exists or an alias exists; otherwise None."""
        if has(z, key):
            return get(z, key)
        if key in ALIAS:
            for cand in ALIAS[key]:
                if has(z, cand):
                    return get(z, cand)
        return None

    stacked: Dict[str, List[np.ndarray]] = {k: [] for k in keys}

    for fp in files:
        z = np.load(fp, allow_pickle=True)

        if y_key not in z.files:
            raise KeyError(f"y_key='{y_key}' not found in {fp}. Available keys: {list(z.files)}")

        for k in keys:
            arr = fetch_with_alias(z, k)

            # Synthesize q50 if requested and missing
            if arr is None and k == "q50":
                # Prefer calibrated bounds
                lo = fetch_with_alias(z, "q05_cal")
                hi = fetch_with_alias(z, "q95_cal")
                if lo is not None and hi is not None:
                    arr = 0.5 * (lo + hi)
                else:
                    # Fall back to raw bounds
                    lo = fetch_with_alias(z, "q05")
                    hi = fetch_with_alias(z, "q95")
                    if lo is not None and hi is not None:
                        arr = 0.5 * (lo + hi)

            if arr is None:
                raise KeyError(
                    f"Key '{k}' not found in {fp}. Available keys: {list(z.files)}"
                )

            stacked[k].append(arr)

    # Truncate to common min length (safe across slight mismatches)
    min_len = min([len(y)] + [len(a) for k in keys for a in stacked[k]])
    y = y[:min_len]

    out = {y_key: y}
    for k in keys:
        A = np.stack([a[:min_len] for a in stacked[k]], axis=0)  # (nseed, T)
        out[k] = np.mean(A, axis=0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **out)


# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="Root directory containing seed_* folders (possibly nested)")
    ap.add_argument("--filename", type=str, required=True, help="NPZ file name inside each seed folder (searched recursively)")
    ap.add_argument("--out", type=str, default=None, help="Output npz path (default: <root>/<filename_stem>_agg.npz)")
    ap.add_argument("--y_key", type=str, default="y_true", help="Key for ground truth array in the npz")
    ap.add_argument("--keys", type=str, default="q05_cal,q50,q95_cal", help="Comma-separated keys to aggregate")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        raise FileNotFoundError(root)

    keys = [k.strip() for k in args.keys.split(",") if k.strip()]
    files = find_seed_files(root, args.filename)

    if len(files) == 0:
        raise FileNotFoundError(f"No files found under: {root} with pattern **/seed_*/**/{args.filename}")

    out_path = Path(args.out).resolve() if args.out else (root / (Path(args.filename).stem + "_agg.npz"))

    print(f"[INFO] Found {len(files)} seed files:")
    for f in files:
        print("  -", f)

    aggregate_npz(files, out_path, keys=keys, y_key=args.y_key)
    print(f"[OK] Saved aggregated npz -> {out_path}")


if __name__ == "__main__":
    main()