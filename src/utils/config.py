import yaml
from pathlib import Path

def load_config(path: str | Path) -> dict:
    """Load YAML configuration file and return a dictionary."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        cfg = {}

    if not isinstance(cfg, dict):
        raise TypeError(
            f"Config must be a dict, but got {type(cfg).__name__}. "
            "Top-level YAML must be key-value pairs."
        )

    return cfg