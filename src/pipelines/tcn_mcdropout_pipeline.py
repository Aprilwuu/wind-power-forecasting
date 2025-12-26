#  src/pipelines/base_pipeline.py
from abc import ABC, abstractmethod

class BasePipeline(ABC):

    @abstractmethod
    def run(self) -> dict:
        """
        Should return a dict like:
        {
            "metrics":{...},
            "predictions":...,
            "config":{...}
        }
        """
        pass