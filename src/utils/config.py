import yaml

def load_config(path: str) -> dict:
    """Load YAML configuration file and return a dictionary."""
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config