from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import matplotlib.pyplot as plt


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


def infer_mode(npz: Dict[str, np.ndarray]) -> str:
    if any(k in npz for k in ["q05_cal", "q95_cal", "lower_cal", "upper_cal"]):
        return "cal"
    return "raw"


def _get_center(
    primary_npz: Dict[str, np.ndarray],
    lo: np.ndarray,
    hi: np.ndarray,
    raw_npz: Optional[Dict[str, np.ndarray]] = None,
) -> np.ndarray:
    """
    Priority for center:
      1) calibrated/raw file's q50_cal/q50/median/y_pred/pred/mu/mean
      2) raw file's q50/median/y_pred/pred/mu/mean
      3) fallback to midpoint
    """
    k = _first_key(primary_npz, ["q50_cal", "q50", "median", "y_pred", "pred", "mu", "mean"])
    if k is not None:
        c = _squeeze_1d(primary_npz[k])
        n = min(len(c), len(lo), len(hi))
        return c[:n]

    if raw_npz is not None:
        k2 = _first_key(raw_npz, ["q50", "median", "y_pred", "pred", "mu", "mean"])
        if k2 is not None:
            c = _squeeze_1d(raw_npz[k2])
            n = min(len(c), len(lo), len(hi))
            return c[:n]

    return 0.5 * (lo + hi)


@dataclass
class SeriesPack:
    y: np.ndarray
    lo: np.ndarray
    mid: np.ndarray
    hi: np.ndarray
    name: str


def get_pack_auto(
    npz: Dict[str, np.ndarray],
    name: str,
    raw_npz_for_center: Optional[Dict[str, np.ndarray]] = None,
) -> SeriesPack:
    """
    Auto-detect lo/hi keys across different model outputs.

    Supported patterns:
      - QR style: q05_cal / q95_cal (preferred) or q05 / q95
      - Beta style: lower_cal / upper_cal (preferred) or lower / upper

    Center:
      - q50/q50_cal if available
      - else mu/mean if available
      - else fallback to raw file center if provided
      - else midpoint

    y:
      - y_true (required)
    """
    if npz is None:
        raise ValueError(f"[{name}] npz is None")

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

    mid = _get_center(npz, lo, hi, raw_npz=raw_npz_for_center)
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

    mae = float(np.mean(np.abs(a2.y - b2.y)))
    msg = f"per-zone alignment ok: nzones={nzones}, lookback={lookback}, per-zone TA={TA}, TB={TB}, y_MAE={mae:.6g}"
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
    A2 = SeriesPack(
        y=A2.y[:n], lo=A2.lo[:n], mid=A2.mid[:n], hi=A2.hi[:n], name=A2.name + f" (aligned: shift {shift})"
    )
    B2 = SeriesPack(
        y=B2.y[:n], lo=B2.lo[:n], mid=B2.mid[:n], hi=B2.hi[:n], name=B2.name + f" (aligned: shift {shift})"
    )

    mae = float(np.mean(np.abs(A2.y - B2.y)))
    msg = f"global-shift alignment: shift={shift}, y_MAE={mae:.6g} (searched ±{max_shift})"
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


def align_to_reference(
    ref: SeriesPack,
    others: List[SeriesPack],
    *,
    align: str,
    nzones: int,
    lookback: int,
    max_shift: int,
) -> Tuple[SeriesPack, List[SeriesPack], List[str]]:
    """
    Align each model to ref independently, then truncate all to a common length.
    """
    ref_current = ref
    aligned_others: List[SeriesPack] = []
    msgs = [f"reference={ref.name}"]

    for pack in others:
        ref_aligned, pack_aligned, msg = auto_align(
            ref_current,
            pack,
            align=align,
            nzones=nzones,
            lookback=lookback,
            max_shift=max_shift,
        )
        ref_current = ref_aligned
        aligned_others.append(pack_aligned)
        msgs.append(f"{pack.name}: {msg}")

    common_n = min([len(ref_current.y)] + [len(p.y) for p in aligned_others])

    ref_final = SeriesPack(
        y=ref_current.y[:common_n],
        lo=ref_current.lo[:common_n],
        mid=ref_current.mid[:common_n],
        hi=ref_current.hi[:common_n],
        name=ref_current.name,
    )
    others_final = [
        SeriesPack(y=p.y[:common_n], lo=p.lo[:common_n], mid=p.mid[:common_n], hi=p.hi[:common_n], name=p.name)
        for p in aligned_others
    ]
    return ref_final, others_final, msgs


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
        miss = np.maximum(b.lo[:n] - b.y[:n], 0) + np.maximum(a.y[:n] - b.hi[:n], 0)
        c = np.convolve(miss, np.ones(window), mode="valid")
        return int(np.argmax(c))

    raise ValueError(f"Unknown pick: {pick}")


def pick_start_multi(ref: SeriesPack, packs: List[SeriesPack], window: int, pick: str) -> int:
    n = min([len(ref.y)] + [len(p.y) for p in packs])
    if window >= n:
        return 0

    if pick == "first":
        return 0

    err = np.zeros(n, dtype=float)

    if pick == "gap":
        for p in packs:
            err += np.abs(p.mid[:n] - ref.y[:n])
        c = np.convolve(err, np.ones(window), mode="valid")
        return int(np.argmax(c))

    for p in packs:
        miss = np.maximum(p.lo[:n] - ref.y[:n], 0) + np.maximum(ref.y[:n] - p.hi[:n], 0)
        err += miss
    c = np.convolve(err, np.ones(window), mode="valid")
    return int(np.argmax(c))


# -----------------------------
# Plotting
# -----------------------------
def plot_case(out_png: str, a: SeriesPack, b: SeriesPack, *, window: int, start: int, title: str) -> None:
    n = min(len(a.y), len(b.y))
    start = max(0, min(start, n - window))
    end = start + window

    x = np.arange(window)
    y = a.y[start:end]

    a_lo, a_mid, a_hi = a.lo[start:end], a.mid[start:end], a.hi[start:end]
    b_lo, b_mid, b_hi = b.lo[start:end], b.mid[start:end], b.hi[start:end]

    plt.figure(figsize=(10.5, 3.8))
    plt.plot(x, y, marker="o", markersize=2.5, linewidth=1.0, label="y_true")

    plt.fill_between(x, a_lo, a_hi, alpha=0.12, label=f"{a.name} 90% PI")
    plt.plot(x, a_mid, linewidth=1.2, label=f"{a.name} center")

    plt.fill_between(x, b_lo, b_hi, alpha=0.12, label=f"{b.name} 90% PI")
    plt.plot(x, b_mid, linewidth=1.2, label=f"{b.name} center")

    plt.title(title)
    plt.xlabel(f"time within window (start={start}, len={window})")
    plt.ylabel("normalized power")
    plt.legend(ncol=2, fontsize=8)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    outp = Path(out_png)
    outp.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outp, dpi=200, bbox_inches="tight")
    plt.close()


def plot_case_multi(out_png: str, ref: SeriesPack, packs: List[SeriesPack], *, window: int, start: int, title: str) -> None:
    n = min([len(ref.y)] + [len(p.y) for p in packs])
    start = max(0, min(start, n - window))
    end = start + window

    x = np.arange(window)

    plt.figure(figsize=(15.5, 6.5))

    # y_true
    plt.plot(
        x,
        ref.y[start:end],
        marker="o",
        markersize=3.0,
        linewidth=1.2,
        label="y_true",
        color="black",
    )

    # model colors
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]

    for i, p in enumerate(packs):
        c = colors[i % len(colors)]

        plt.fill_between(
            x,
            p.lo[start:end],
            p.hi[start:end],
            alpha=0.08,
            color=c
        )

        plt.plot(
            x,
            p.mid[start:end],
            linewidth=1.8,
            color=c,
            label=p.name
        )

    plt.title(title)
    plt.xlabel(f"time within window (start={start}, len={window})")
    plt.ylabel("normalized power")
    plt.legend(ncol=2, fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    outp = Path(out_png)
    outp.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outp, dpi=200, bbox_inches="tight")
    plt.close()


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()

    # main inputs
    ap.add_argument("--lgbm", type=str, required=False, default=None)
    ap.add_argument("--tqr", type=str, required=False, default=None)
    ap.add_argument("--tbeta", type=str, required=False, default=None)
    ap.add_argument("--tcn_noise", type=str, required=False, default=None)

    # optional raw files used only for center fallback
    ap.add_argument("--lgbm_raw", type=str, required=False, default=None)
    ap.add_argument("--tqr_raw", type=str, required=False, default=None)
    ap.add_argument("--tbeta_raw", type=str, required=False, default=None)
    ap.add_argument("--tcn_noise_raw", type=str, required=False, default=None)

    ap.add_argument("--out_fig1", type=str, required=False)
    ap.add_argument("--out_fig2", type=str, required=False)
    ap.add_argument("--out_best", type=str, default=None)
    ap.add_argument("--out_all", type=str, default=None)

    ap.add_argument("--out_tqr_single", type=str, default=None)
    ap.add_argument("--out_tbeta_single", type=str, default=None)
    ap.add_argument("--out_tcn_single", type=str, default=None)

    ap.add_argument("--window", type=int, default=168)
    ap.add_argument("--pick", type=str, default="gap", choices=["gap", "first", "A_worst", "B_worst"])
    ap.add_argument("--align", type=str, default="auto", choices=["auto", "per_zone", "shift", "none"])
    ap.add_argument("--nzones", type=int, default=10)
    ap.add_argument("--lookback", type=int, default=168)
    ap.add_argument("--max_shift", type=int, default=400)

    ap.add_argument("--title1", type=str, default="Tree vs Deep")
    ap.add_argument("--title2", type=str, default="Uncertainty strategy")
    ap.add_argument("--title_best", type=str, default="Different modeling strategies")
    ap.add_argument("--title_all", type=str, default="All models")
    ap.add_argument("--title_tqr", type=str, default="Transformer-QR")
    ap.add_argument("--title_beta", type=str, default="Transformer-Beta")
    ap.add_argument("--title_tcn", type=str, default="TCN-MC-noise")

    ap.add_argument("--same_window", action="store_true")

    args = ap.parse_args()

    def _load_optional(path: Optional[str]) -> Optional[Dict[str, np.ndarray]]:
        return load_npz(path) if path else None

    def _make_name(base: str, primary_npz: Dict[str, np.ndarray]) -> str:
        return f"{base} ({infer_mode(primary_npz)})"

    # load primary files
    L_npz = _load_optional(args.lgbm)
    Q_npz = _load_optional(args.tqr)
    B_npz = _load_optional(args.tbeta)
    T_npz = _load_optional(args.tcn_noise)

    # load raw files for center fallback
    L_raw = _load_optional(args.lgbm_raw)
    Q_raw = _load_optional(args.tqr_raw)
    B_raw = _load_optional(args.tbeta_raw)
    T_raw = _load_optional(args.tcn_noise_raw)

    L = get_pack_auto(L_npz, _make_name("LightGBM-QR", L_npz), raw_npz_for_center=L_raw) if L_npz is not None else None
    Q = get_pack_auto(Q_npz, _make_name("Transformer-QR", Q_npz), raw_npz_for_center=Q_raw) if Q_npz is not None else None
    B = get_pack_auto(B_npz, _make_name("Transformer-Beta", B_npz), raw_npz_for_center=B_raw) if B_npz is not None else None
    T = get_pack_auto(T_npz, _make_name("TCN-MC-noise", T_npz), raw_npz_for_center=T_raw) if T_npz is not None else None

    # fig1: L vs Q
    if args.out_fig1:
        if L is None or Q is None:
            raise ValueError("out_fig1 requires both --lgbm and --tqr")

        L2, Q2, msg1 = auto_align(L, Q, align=args.align, nzones=args.nzones, lookback=args.lookback, max_shift=args.max_shift)
        print("[ALIGN-fig1]", msg1)

        n1 = min(len(L2.y), len(Q2.y))
        L2 = SeriesPack(y=L2.y[:n1], lo=L2.lo[:n1], mid=L2.mid[:n1], hi=L2.hi[:n1], name=L2.name)
        Q2 = SeriesPack(y=Q2.y[:n1], lo=Q2.lo[:n1], mid=Q2.mid[:n1], hi=Q2.hi[:n1], name=Q2.name)

        start1 = pick_start(L2, Q2, args.window, args.pick)
        plot_case(args.out_fig1, L2, Q2, window=args.window, start=start1, title=args.title1)
        print(f"[OK] saved fig1 -> {args.out_fig1} (start={start1}, window={args.window})")

    # fig2: Q vs B
    if args.out_fig2:
        if Q is None or B is None:
            raise ValueError("out_fig2 requires both --tqr and --tbeta")

        Q2, B2, msg2 = auto_align(Q, B, align=args.align, nzones=args.nzones, lookback=args.lookback, max_shift=args.max_shift)
        print("[ALIGN-fig2]", msg2)

        n2 = min(len(Q2.y), len(B2.y))
        Q2 = SeriesPack(y=Q2.y[:n2], lo=Q2.lo[:n2], mid=Q2.mid[:n2], hi=Q2.hi[:n2], name=Q2.name)
        B2 = SeriesPack(y=B2.y[:n2], lo=B2.lo[:n2], mid=B2.mid[:n2], hi=B2.hi[:n2], name=B2.name)

        start2 = pick_start(Q2, B2, args.window, args.pick)
        plot_case(args.out_fig2, Q2, B2, window=args.window, start=start2, title=args.title2)
        print(f"[OK] saved fig2 -> {args.out_fig2} (start={start2}, window={args.window})")

    # best: L vs T
    if args.out_best:
        if L is None or T is None:
            raise ValueError("out_best requires both --lgbm and --tcn_noise")

        Lb, Tb, msgb = auto_align(L, T, align=args.align, nzones=args.nzones, lookback=args.lookback, max_shift=args.max_shift)
        print("[ALIGN-best]", msgb)

        nb = min(len(Lb.y), len(Tb.y))
        Lb = SeriesPack(y=Lb.y[:nb], lo=Lb.lo[:nb], mid=Lb.mid[:nb], hi=Lb.hi[:nb], name=Lb.name)
        Tb = SeriesPack(y=Tb.y[:nb], lo=Tb.lo[:nb], mid=Tb.mid[:nb], hi=Tb.hi[:nb], name=Tb.name)

        startb = pick_start(Lb, Tb, args.window, args.pick)
        plot_case(args.out_best, Lb, Tb, window=args.window, start=startb, title=args.title_best)
        print(f"[OK] saved best -> {args.out_best} (start={startb}, window={args.window})")

    # all models on one figure
    if args.out_all:
        packs_available = []
        if L is not None:
            packs_available.append(L)
        if Q is not None:
            packs_available.append(Q)
        if B is not None:
            packs_available.append(B)
        if T is not None:
            packs_available.append(T)

        if len(packs_available) < 2:
            raise ValueError("out_all requires at least two models")

        ref = packs_available[0]
        others = packs_available[1:]

        ref2, others2, msgs = align_to_reference(
            ref,
            others,
            align=args.align,
            nzones=args.nzones,
            lookback=args.lookback,
            max_shift=args.max_shift,
        )
        for m in msgs:
            print("[ALIGN-all]", m)

        all_packs = [ref2] + others2
        start_all = pick_start_multi(ref2, all_packs, args.window, args.pick)
        plot_case_multi(args.out_all, ref2, all_packs, window=args.window, start=start_all, title=args.title_all)
        print(f"[OK] saved all -> {args.out_all} (start={start_all}, window={args.window})")

    if args.out_tqr_single:
        if Q is None:
            raise ValueError("out_tqr_single requires --tqr")
        plot_case(args.out_tqr_single, Q, Q, window=args.window, start=0, title=args.title_tqr)

    if args.out_tbeta_single:
        if B is None:
            raise ValueError("out_tbeta_single requires --tbeta")
        plot_case(args.out_tbeta_single, B, B, window=args.window, start=0, title=args.title_beta)

    if args.out_tcn_single:
        if T is None:
            raise ValueError("out_tcn_single requires --tcn_noise")
        plot_case(args.out_tcn_single, T, T, window=args.window, start=0, title=args.title_tcn)


if __name__ == "__main__":
    main()