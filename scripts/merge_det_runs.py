import json
import re
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(r"E:\Projects\wind-power-forecasting\data\featured\tcn_track2_det")
OUT_JSON = Path(r"E:\Projects\wind-power-forecasting\reports\experiments\tcn_track2_det\merged_runs.json")
OUT_CSV  = Path(r"E:\Projects\wind-power-forecasting\reports\experiments\tcn_track2_det\merged_runs.csv")

# 你可以按需要扩展候选人列表；也可以不限制，直接扫所有 candidate
CANDIDATES = None  # e.g., {"baseline", "tuned"} or None

# 匹配 ...\<candidate>\heldout_(\d+)\seed_(\d+)\metrics.json
pat = re.compile(r".*\\(?P<cand>[^\\]+)\\heldout_(?P<g>\d+)\\seed_(?P<s>\d+)\\metrics\.json$", re.I)

def safe_get(d, keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def stat_block(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "mean": float(mean(vals)),
        "std": float(pstdev(vals)) if len(vals) > 1 else 0.0,
        "min": float(min(vals)),
        "max": float(max(vals)),
    }

runs = []
for p in ROOT.rglob("metrics.json"):
    m = pat.match(str(p))
    if not m:
        continue
    cand = m.group("cand")
    if CANDIDATES is not None and cand not in CANDIDATES:
        continue

    heldout = int(m.group("g"))
    seed = int(m.group("s"))

    metrics = json.loads(p.read_text(encoding="utf-8"))
    # 下面这些 key 需要和你 metrics.json 的实际结构对齐：
    # 我先按常见命名写：inner_val_rmse / outer_test_rmse 等
    rec = {
    "candidate": cand,
    "held_out_group": heldout,
    "seed": seed,

    "inner_val_rmse": safe_get(metrics, ["inner_val", "rmse"]),
    "inner_val_mae":  safe_get(metrics, ["inner_val", "mae"]),
    "inner_val_r2":   safe_get(metrics, ["inner_val", "r2"]),

    "outer_test_rmse": safe_get(metrics, ["outer_test", "rmse"]),
    "outer_test_mae":  safe_get(metrics, ["outer_test", "mae"]),
    "outer_test_r2":   safe_get(metrics, ["outer_test", "r2"]),

    "pipeline_metrics_path": str(p),
    "pipeline_out_dir": str(p.parent),
}


    # 如果你的 metrics.json 结构是嵌套的（比如 metrics["inner_val"]["rmse"]），改成：
    # rec["inner_val_rmse"] = safe_get(metrics, ["inner_val", "rmse"])
    # rec["outer_test_rmse"] = safe_get(metrics, ["outer_test", "rmse"])

    runs.append(rec)

# 去重（同 cand/heldout/seed 只保留一条，避免你重复跑产生重复文件）
uniq = {}
for r in runs:
    k = (r["candidate"], r["held_out_group"], r["seed"])
    uniq[k] = r
runs = sorted(uniq.values(), key=lambda x: (x["candidate"], x["seed"], x["held_out_group"]))

# 汇总 by_candidate
by_candidate = {}
for r in runs:
    cand = r["candidate"]
    by_candidate.setdefault(cand, []).append(r)

summary = {}
for cand, rr in by_candidate.items():
    summary[cand] = {
        "n_runs": len(rr),
        "inner_val": {
            "rmse": stat_block([x["inner_val_rmse"] for x in rr]),
            "mae":  stat_block([x["inner_val_mae"] for x in rr]),
            "r2":   stat_block([x["inner_val_r2"] for x in rr]),
        },
        "outer_test": {
            "rmse": stat_block([x["outer_test_rmse"] for x in rr]),
            "mae":  stat_block([x["outer_test_mae"] for x in rr]),
            "r2":   stat_block([x["outer_test_r2"] for x in rr]),
        },
    }

out = {
    "experiment": "tcn_track2_det",
    "protocol": "track2_lofo",
    "artifact_exp_dir": str(ROOT),
    "n_runs": len(runs),
    "runs": runs,
    "by_candidate": summary,
}

OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"[OK] merged json -> {OUT_JSON}  (runs={len(runs)})")

# 额外导出 CSV 方便你用 Excel / pandas 看
try:
    import csv
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(runs[0].keys()) if runs else [])
        w.writeheader()
        for r in runs:
            w.writerow(r)
    print(f"[OK] merged csv  -> {OUT_CSV}")
except Exception as e:
    print("[WARN] CSV export failed:", e)
