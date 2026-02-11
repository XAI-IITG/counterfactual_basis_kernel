from abc import ABC, abstractmethod
import torch

class CounterfactualExplainer(ABC):
    def __init__(self, model, **kwargs):
        self.model = model

    @abstractmethod
    def generate(self, query_instance: torch.Tensor, target_rul: float) -> torch.Tensor:
        """
        Args:
            query_instance: Input time series (T, D)
            target_rul: Desired outcome
        Returns:
            cf_instance: Counterfactual time series (T, D)
        """
        pass