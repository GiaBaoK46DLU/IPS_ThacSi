"""Algorithm is an IPS lifecycle, never a torch.nn.Module interface."""
from __future__ import annotations
from abc import ABC, abstractmethod

from ..data_access import FingerprintInputs, FingerprintSplit
from ..experiments.contracts import AlgorithmExecution, ExperimentConfig, IPSPredictions, ModelDiagram


class Algorithm(ABC):
    def __init__(self, config: ExperimentConfig):
        self.config = config

    @abstractmethod
    def prepare(self, train: FingerprintSplit) -> None:
        """Fit branch-specific preprocessing on train only. No validation/test argument."""

    @abstractmethod
    def execute(self, validation: FingerprintSplit | None) -> AlgorithmExecution:
        """Train, mine, fine-tune or freeze an inference-only pipeline.

        Validation is for selecting/evaluating candidates, not fitting transforms.
        Return finalized=True only after freezing all choices used by predict.
        """

    @abstractmethod
    def predict(self, inputs: FingerprintInputs) -> IPSPredictions:
        """No truth labels/metadata. Return exactly the requested sample order."""

    def get_diagram(self) -> ModelDiagram | None:
        return None
