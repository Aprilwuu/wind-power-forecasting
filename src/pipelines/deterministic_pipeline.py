#  src/pipelines/deterministic_pipeline.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Sequence, Dict, Any, Tuple

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

from pipelines.tcn_mcdropout_pipeline import BasePipeline


@dataclass
class DeterministicConfig:
    # data
    data_path: str | None = None              # e.g., "data/gefcom.csv"
    target_col: "target_col"
    feature_cos: Sequence[str] | None = None
    text_size: float = 0.2
    val_size: float = 0.1
    shuffle: bool = False
    seed: int = 42

    # model
    model_type: str = "naive"