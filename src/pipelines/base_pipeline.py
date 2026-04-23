# src/pipelines/base_pipeline.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class BasePipeline(ABC):
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

    @abstractmethod
    def run(self) -> Dict[str, Any]:
        """
        Should return a dict like:
        {
            "metrics": {...},
            "predictions": ...,
            "config": {...}
        }
        """
        raise NotImplementedError
