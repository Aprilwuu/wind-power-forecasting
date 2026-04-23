import yaml
import itertools
import os
from pathlib import Path
from src.pipelines.deterministic_lgbm_pipeline import run_pipeline_any

def main(config_path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    grid = cfg["grid"]
    fixed = cfg["fixed_params"]
    seeds = cfg["seeds"]
    runtime = cfg["runtime"]
    features = cfg["features"]
    data = cfg["data"]

    keys = list(grid.keys())
    values = list(grid.values())

    for combo in itertools.product(*values):
        combo_dict = dict(zip(keys, combo))
        combo_name = "_".join(f"{k}-{v}" for k,v in combo_dict.items())

        for seed in seeds:
            params = fixed.copy()
            params.update(combo_dict)
            params["seed"] = seed

            out_dir = Path(runtime["out_dir"]) / combo_name / f"seed_{seed}"
            os.makedirs(out_dir, exist_ok=True)

            run_pipeline_any(
                data_path=runtime["data_path"],
                out_dir=str(out_dir),
                train_cut=runtime["train_cut"],
                val_cut=runtime["val_cut"],
                lags=features["lags"],
                rolling_windows=features["rolling_windows"],
                target_col=data["target_col"],
                zone_col=data["zone_col"],
                time_col=data["time_col"],
                keep_zone=data["keep_zone"],
                lgbm_params=params,
                seed=seed
            )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)
