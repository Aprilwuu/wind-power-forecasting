# scripts/postprocess/run_conformal_batch.py
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple


# ============================
# User config
# ============================
DATA_ROOT = Path(r"E:\Projects\wind-power-forecasting\data\featured")

# 你要跑的 track 文件夹（按你的实际目录名改/加）
TRACK_DIRS = [
    "tcn_track1_mc500",
    "tcn_track2_mc500",
]

# conformal 参数（一般不用改）
TARGET_PICP = 0.90
LO_KEY = "q05"
HI_KEY = "q95"
Y_KEY = "y_true"
SUFFIX = "_cal"
CLIP_MIN = 0.0
CLIP_MAX = 1.0

# 文件名配对规则： (calibration_file_name, apply_file_name)
# Track 1 常见: preds_val_mc.npz -> preds_test_mc.npz
# Track 2 常见: preds_inner_val_mc.npz -> preds_outer_test_mc.npz
PAIR_RULES: List[Tuple[str, str]] = [
    ("preds_val_mc.npz", "preds_test_mc.npz"),
    ("preds_inner_val_mc.npz", "preds_outer_test_mc.npz"),
    # 如果你的 track2 test 命名不同，可以在这里再加一条
    ("preds_inner_val_mc.npz", "preds_test_mc.npz"),
]

# 如果你只想跑某一个 track，把另一个注释掉即可
# ============================


def project_root() -> Path:
    # .../scripts/postprocess/run_conformal_batch.py -> project root is parents[2]
    return Path(__file__).resolve().parents[2]


def script_path() -> Path:
    return project_root() / "scripts" / "postprocess" / "run_conformal.py"


def build_env() -> Dict[str, str]:
    env = os.environ.copy()
    pr = str(project_root())
    env["PYTHONPATH"] = pr + (os.pathsep + env["PYTHONPATH"] if "PYTHONPATH" in env else "")
    return env


def find_pairs(track_baseline_dir: Path) -> List[Tuple[Path, Path]]:
    """
    Recursively find (cal, apply) pairs under a track baseline directory.
    Deduplicate pairs if multiple rules match.
    """
    found: Dict[Tuple[str, str], Tuple[Path, Path]] = {}

    for cal_name, apply_name in PAIR_RULES:
        for cal_path in track_baseline_dir.rglob(cal_name):
            apply_path = cal_path.with_name(apply_name)
            if apply_path.exists():
                key = (str(cal_path), str(apply_path))
                found[key] = (cal_path, apply_path)

    pairs = list(found.values())
    pairs.sort(key=lambda x: (str(x[0]), str(x[1])))
    return pairs


def run_one(cal_path: Path, apply_path: Path, env: Dict[str, str]) -> None:
    cmd = [
        sys.executable,
        str(script_path()),
        "--cal",
        str(cal_path),
        "--apply",
        str(apply_path),
        "--target_picp",
        f"{TARGET_PICP}",
        "--lo",
        LO_KEY,
        "--hi",
        HI_KEY,
        "--y",
        Y_KEY,
        "--suffix",
        SUFFIX,
        "--clip_min",
        f"{CLIP_MIN}",
        "--clip_max",
        f"{CLIP_MAX}",
    ]

    print(f"[RUN]\n  cal  : {cal_path}\n  apply: {apply_path}")
    subprocess.run(cmd, check=True, env=env)


def main() -> None:
    env = build_env()

    print(f"Project root : {project_root()}")
    print(f"Data root    : {DATA_ROOT}")
    print(f"Run script   : {script_path()}")
    print(f"Target PICP  : {TARGET_PICP}\n")

    if not script_path().exists():
        raise FileNotFoundError(f"Cannot find: {script_path()}")

    for track_dir in TRACK_DIRS:
        track_root = DATA_ROOT / track_dir / "baseline"
        if not track_root.exists():
            print(f"[SKIP] Not found: {track_root}")
            continue

        pairs = find_pairs(track_root)
        print(f"=== {track_dir} ===")
        print(f"Baseline dir: {track_root}")
        print(f"Found pairs : {len(pairs)}\n")

        if not pairs:
            print("[WARN] No file pairs found. Check PAIR_RULES or filenames.\n")
            continue

        for cal_path, apply_path in pairs:
            run_one(cal_path, apply_path, env)

        print(f"\n[DONE] {track_dir}\n")


if __name__ == "__main__":
    main()