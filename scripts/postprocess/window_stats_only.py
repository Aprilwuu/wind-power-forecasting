from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np


# -----------------------------
# Loading
# -----------------------------
def _squeeze_1d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim == 2 and a.shape[1] == 1:
        return a[:, 0]
    if a.ndim == 1:
        return a
    raise ValueError(f"Expected 1D array (or Nx1), got shape={a.shape}")


def load_npz(path: str) -> Dict[str, np.ndarray]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    z = np.load(p, allow_pickle=True)
    return {k: z[k] for k in z.files}


def _first_key(d: Dict[str, np.ndarray], keys: List[str]) -> Optional[str]:
    for k in keys:
        if k in d:
            return k
    return None


def _get_center(d: Dict[str, np.ndarray], lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """
    Prefer a provided center/median if available; otherwise fallback to (lo+hi)/2.
    """
    k = _first_key(d, ["q50_cal", "q50", "median", "y_pred", "pred", "mu", "mean"])
    if k is None:
        return 0.5 * (lo + hi)
    c = _squeeze_1d(d[k])
    n = min(len(c), len(lo), len(hi))
    return c[:n]


@dataclass
class SeriesPack:
    y: np.ndarray
    lo: np.ndarray
    mid: np.ndarray
    hi: np.ndarray
    name: str


def get_pack_auto(npz: Dict[str, np.ndarray], name: str) -> SeriesPack:
    if "y_true" not in npz:
        raise KeyError(f"[{name}] missing required key 'y_true'. Available: {list(npz.keys())}")

    lo_key = _first_key(npz, ["q05_cal", "lower_cal", "q05", "lower"])
    hi_key = _first_key(npz, ["q95_cal", "upper_cal", "q95", "upper"])

    if lo_key is None or hi_key is None:
        raise KeyError(
            f"[{name}] cannot find lo/hi keys. "
            f"Expected one of q05_cal/lower_cal/q05/lower and q95_cal/upper_cal/q95/upper. "
            f"Available: {list(npz.keys())}"
        )

    y = _squeeze_1d(npz["y_true"])
    lo = _squeeze_1d(npz[lo_key])
    hi = _squeeze_1d(npz[hi_key])

    n = min(len(y), len(lo), len(hi))
    y, lo, hi = y[:n], lo[:n], hi[:n]

    mid = _get_center(npz, lo, hi)
    n2 = min(len(y), len(lo), len(mid), len(hi))
    return SeriesPack(y=y[:n2], lo=lo[:n2], mid=mid[:n2], hi=hi[:n2], name=name)


# -----------------------------
# Alignment
# -----------------------------
def align_per_zone(a: SeriesPack, b: SeriesPack, *, nzones: int, lookback: int) -> Tuple[SeriesPack, SeriesPack, str]:
    nA = len(a.y)
    nB = len(b.y)

    if nzones <= 0:
        raise ValueError("nzones must be positive")

    if nA % nzones != 0 or nB % nzones != 0:
        raise ValueError(f"Cannot per-zone align: lenA={nA}, lenB={nB} not divisible by nzones={nzones}")

    TA = nA // nzones
    TB = nB // nzones
    if TA - TB != lookback:
        raise ValueError(
            f"Cannot per-zone align: per-zone lengths TA={TA}, TB={TB}, expected TA-TB==lookback={lookback}"
        )

    def reshape(x: np.ndarray, T: int) -> np.ndarray:
        return x.reshape(nzones, T)

    def drop_first_lookback(x: np.ndarray) -> np.ndarray:
        return reshape(x, TA)[:, lookback:].reshape(-1)

    a2 = SeriesPack(
        y=drop_first_lookback(a.y),
        lo=drop_first_lookback(a.lo),
        mid=drop_first_lookback(a.mid),
        hi=drop_first_lookback(a.hi),
        name=a.name + f" (aligned: per-zone drop {lookback})",
    )
    b2 = SeriesPack(
        y=b.y.copy(),
        lo=b.lo.copy(),
        mid=b.mid.copy(),
        hi=b.hi.copy(),
        name=b.name + " (aligned: per-zone)",
    )

    mae_val = float(np.mean(np.abs(a2.y - b2.y)))
    msg = f"per-zone alignment ok: nzones={nzones}, lookback={lookback}, per-zone TA={TA}, TB={TB}, y_MAE={mae_val:.6g}"
    return a2, b2, msg


def align_global_shift(a: SeriesPack, b: SeriesPack, *, max_shift: int, eval_len: int = 2000) -> Tuple[SeriesPack, SeriesPack, str]:
    yA = a.y
    yB = b.y
    best = (1e18, 0)

    def score(shift: int) -> float:
        if shift >= 0:
            aa = yA[shift:]
            bb = yB
        else:
            aa = yA
            bb = yB[-shift:]
        n = min(len(aa), len(bb), eval_len)
        if n <= 50:
            return 1e18
        return float(np.mean(np.abs(aa[:n] - bb[:n])))

    for s in range(-max_shift, max_shift + 1):
        sc = score(s)
        if sc < best[0]:
            best = (sc, s)

    _, shift = best

    def apply(pack: SeriesPack, which: str) -> SeriesPack:
        if shift >= 0:
            if which == "A":
                y, lo, mid, hi = pack.y[shift:], pack.lo[shift:], pack.mid[shift:], pack.hi[shift:]
            else:
                y, lo, mid, hi = pack.y, pack.lo, pack.mid, pack.hi
        else:
            if which == "B":
                y, lo, mid, hi = pack.y[-shift:], pack.lo[-shift:], pack.mid[-shift:], pack.hi[-shift:]
            else:
                y, lo, mid, hi = pack.y, pack.lo, pack.mid, pack.hi

        n = min(len(y), len(lo), len(mid), len(hi))
        return SeriesPack(y=y[:n], lo=lo[:n], mid=mid[:n], hi=hi[:n], name=pack.name)

    A2 = apply(a, "A")
    B2 = apply(b, "B")

    n = min(len(A2.y), len(B2.y))
    A2 = SeriesPack(y=A2.y[:n], lo=A2.lo[:n], mid=A2.mid[:n], hi=A2.hi[:n], name=A2.name + f" (aligned: shift {shift})")
    B2 = SeriesPack(y=B2.y[:n], lo=B2.lo[:n], mid=B2.mid[:n], hi=B2.hi[:n], name=B2.name + f" (aligned: shift {shift})")

    mae_val = float(np.mean(np.abs(A2.y - B2.y)))
    msg = f"global-shift alignment: shift={shift}, y_MAE={mae_val:.6g} (searched ±{max_shift})"
    return A2, B2, msg


def auto_align(a: SeriesPack, b: SeriesPack, *, align: str, nzones: int, lookback: int, max_shift: int) -> Tuple[SeriesPack, SeriesPack, str]:
    if align == "none":
        n = min(len(a.y), len(b.y))
        return (
            SeriesPack(y=a.y[:n], lo=a.lo[:n], mid=a.mid[:n], hi=a.hi[:n], name=a.name + " (aligned: none)"),
            SeriesPack(y=b.y[:n], lo=b.lo[:n], mid=b.mid[:n], hi=b.hi[:n], name=b.name + " (aligned: none)"),
            f"no alignment, truncated to n={n}",
        )

    if align in ("per_zone", "auto"):
        try:
            return align_per_zone(a, b, nzones=nzones, lookback=lookback)
        except Exception:
            if align == "per_zone":
                raise

    if align in ("shift", "auto"):
        return align_global_shift(a, b, max_shift=max_shift)

    raise ValueError(f"Unknown align mode: {align}")


# -----------------------------
# Window metrics
# -----------------------------
def coverage_rate(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    return float(np.mean((y >= lo) & (y <= hi)))


def mean_interval_width(lo: np.ndarray, hi: np.ndarray) -> float:
    return float(np.mean(hi - lo))


def rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - pred) ** 2)))


def mae(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y - pred)))


def wis_90(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, alpha: float = 0.10) -> float:
    width = hi - lo
    below = np.maximum(lo - y, 0.0)
    above = np.maximum(y - hi, 0.0)
    score = width + (2.0 / alpha) * below + (2.0 / alpha) * above
    return float(np.mean(score))


def mean_abs_diff(x: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(x))))


def summarize_window(pack: SeriesPack, start: int, window: int) -> Dict[str, float]:
    n = len(pack.y)
    start = max(0, min(start, n - window))
    end = start + window

    y = pack.y[start:end]
    lo = pack.lo[start:end]
    mid = pack.mid[start:end]
    hi = pack.hi[start:end]
    width = hi - lo

    return {
        "mae": mae(y, mid),
        "rmse": rmse(y, mid),
        "coverage": coverage_rate(y, lo, hi),
        "avg_width": mean_interval_width(lo, hi),
        "wis": wis_90(y, lo, hi, alpha=0.10),
        "center_diff": mean_abs_diff(mid),
        "width_diff": mean_abs_diff(width),
        "max_abs_err": float(np.max(np.abs(y - mid))),
    }


def collect_window_summary(
    comparison: str,
    model_label: str,
    pack: SeriesPack,
    start: int,
    window: int,
) -> Dict[str, float]:
    s = summarize_window(pack, start, window)
    return {
        "comparison": comparison,
        "model": model_label,
        "window_start": int(start),
        "window_len": int(window),
        "mae": s["mae"],
        "rmse": s["rmse"],
        "coverage": s["coverage"],
        "avg_width": s["avg_width"],
        "wis": s["wis"],
        "center_diff": s["center_diff"],
        "width_diff": s["width_diff"],
        "max_abs_err": s["max_abs_err"],
    }


def save_stats_csv(path: str, rows: List[Dict[str, float]]) -> None:
    outp = Path(path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with outp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_row(row: Dict[str, float]) -> None:
    print(
        f"[{row['comparison']}] {row['model']}: "
        f"start={row['window_start']}, len={row['window_len']}, "
        f"MAE={row['mae']:.4f}, RMSE={row['rmse']:.4f}, "
        f"Coverage={row['coverage']:.4f}, AvgWidth={row['avg_width']:.4f}, "
        f"WIS={row['wis']:.4f}, CenterDiff={row['center_diff']:.4f}, "
        f"WidthDiff={row['width_diff']:.4f}, MaxAbsErr={row['max_abs_err']:.4f}"
    )


# -----------------------------
# Picking a window
# -----------------------------
def pick_start(a: SeriesPack, b: SeriesPack, window: int, pick: str) -> int:
    n = min(len(a.y), len(b.y))
    if window >= n:
        return 0

    if pick == "first":
        return 0

    if pick == "gap":
        err = np.abs(a.mid[:n] - a.y[:n]) + np.abs(b.mid[:n] - b.y[:n])
        c = np.convolve(err, np.ones(window), mode="valid")
        return int(np.argmax(c))

    if pick == "A_worst":
        miss = np.maximum(a.lo[:n] - a.y[:n], 0) + np.maximum(a.y[:n] - a.hi[:n], 0)
        c = np.convolve(miss, np.ones(window), mode="valid")
        return int(np.argmax(c))

    if pick == "B_worst":
        miss = np.maximum(b.lo[:n] - b.y[:n], 0) + np.maximum(b.y[:n] - b.hi[:n], 0)
        c = np.convolve(miss, np.ones(window), mode="valid")
        return int(np.argmax(c))

    raise ValueError(f"Unknown pick: {pick}")


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--lgbm_cal", type=str, required=True)
    ap.add_argument("--tqr_cal", type=str, required=True)
    ap.add_argument("--tbeta_cal", type=str, required=True)
    ap.add_argument("--tcn_noise_cal", type=str, default=None)

    ap.add_argument("--out_csv", type=str, required=True)

    ap.add_argument("--window", type=int, default=168)
    ap.add_argument("--pick", type=str, default="gap", choices=["gap", "first", "A_worst", "B_worst"])
    ap.add_argument("--align", type=str, default="auto", choices=["auto", "per_zone", "shift", "none"])
    ap.add_argument("--nzones", type=int, default=10)
    ap.add_argument("--lookback", type=int, default=168)
    ap.add_argument("--max_shift", type=int, default=400)

    ap.add_argument("--same_window", action="store_true",
                    help="Use the same window start from fig1 for fig2 and best-vs-best.")

    # optional: manually specify starts if you already know the figure windows
    ap.add_argument("--start_fig1", type=int, default=None)
    ap.add_argument("--start_fig2", type=int, default=None)
    ap.add_argument("--start_best", type=int, default=None)

    args = ap.parse_args()

    rows: List[Dict[str, float]] = []

    # load
    L = get_pack_auto(load_npz(args.lgbm_cal), "LightGBM-QR (cal)")
    Q = get_pack_auto(load_npz(args.tqr_cal), "Transformer-QR (cal)")
    B = get_pack_auto(load_npz(args.tbeta_cal), "Transformer-Beta (cal)")

    T: Optional[SeriesPack] = None
    if args.tcn_noise_cal is not None:
        T = get_pack_auto(load_npz(args.tcn_noise_cal), "TCN-MC-noise (cal)")

    # -------------------------
    # Comparison 1: LGBM vs TQR
    # -------------------------
    L2, Q2, msg1 = auto_align(
        L, Q,
        align=args.align,
        nzones=args.nzones,
        lookback=args.lookback,
        max_shift=args.max_shift
    )
    print("[ALIGN-fig1]", msg1)

    n1 = min(len(L2.y), len(Q2.y))
    L2 = SeriesPack(y=L2.y[:n1], lo=L2.lo[:n1], mid=L2.mid[:n1], hi=L2.hi[:n1], name=L2.name)
    Q2 = SeriesPack(y=Q2.y[:n1], lo=Q2.lo[:n1], mid=Q2.mid[:n1], hi=Q2.hi[:n1], name=Q2.name)

    start1 = args.start_fig1 if args.start_fig1 is not None else pick_start(L2, Q2, args.window, args.pick)

    row = collect_window_summary("Tree_vs_Deep", "LightGBM-QR", L2, start1, args.window)
    rows.append(row)
    print_row(row)

    row = collect_window_summary("Tree_vs_Deep", "Transformer-QR", Q2, start1, args.window)
    rows.append(row)
    print_row(row)

    # -------------------------
    # Comparison 2: TQR vs Beta
    # -------------------------
    Q3, B2, msg2 = auto_align(
        Q, B,
        align=args.align,
        nzones=args.nzones,
        lookback=args.lookback,
        max_shift=args.max_shift
    )
    print("[ALIGN-fig2]", msg2)

    n2 = min(len(Q3.y), len(B2.y))
    Q3 = SeriesPack(y=Q3.y[:n2], lo=Q3.lo[:n2], mid=Q3.mid[:n2], hi=Q3.hi[:n2], name=Q3.name)
    B2 = SeriesPack(y=B2.y[:n2], lo=B2.lo[:n2], mid=B2.mid[:n2], hi=B2.hi[:n2], name=B2.name)

    if args.start_fig2 is not None:
        start2 = args.start_fig2
    else:
        start2 = pick_start(Q3, B2, args.window, args.pick)
        if args.same_window:
            start2 = min(start1, max(0, min(len(Q3.y), len(B2.y)) - args.window))

    row = collect_window_summary("Uncertainty_Strategy", "Transformer-QR", Q3, start2, args.window)
    rows.append(row)
    print_row(row)

    row = collect_window_summary("Uncertainty_Strategy", "Transformer-Beta", B2, start2, args.window)
    rows.append(row)
    print_row(row)

    # -------------------------
    # Comparison 3: LGBM vs TCN
    # -------------------------
    if T is not None:
        Lb, Tb, msgb = auto_align(
            L, T,
            align=args.align,
            nzones=args.nzones,
            lookback=args.lookback,
            max_shift=args.max_shift
        )
        print("[ALIGN-best]", msgb)

        nb = min(len(Lb.y), len(Tb.y))
        Lb = SeriesPack(y=Lb.y[:nb], lo=Lb.lo[:nb], mid=Lb.mid[:nb], hi=Lb.hi[:nb], name=Lb.name)
        Tb = SeriesPack(y=Tb.y[:nb], lo=Tb.lo[:nb], mid=Tb.mid[:nb], hi=Tb.hi[:nb], name=Tb.name)

        if args.start_best is not None:
            startb = args.start_best
        else:
            startb = pick_start(Lb, Tb, args.window, args.pick)
            if args.same_window:
                startb = min(start1, max(0, min(len(Lb.y), len(Tb.y)) - args.window))

        row = collect_window_summary("Best_vs_Best", "LightGBM-QR", Lb, startb, args.window)
        rows.append(row)
        print_row(row)

        row = collect_window_summary("Best_vs_Best", "TCN-MC-noise", Tb, startb, args.window)
        rows.append(row)
        print_row(row)

    save_stats_csv(args.out_csv, rows)
    print(f"[OK] saved stats -> {args.out_csv}")


if __name__ == "__main__":
    main()
