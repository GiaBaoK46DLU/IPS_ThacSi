"""Team-owned LLM contract. No invented RSSI-to-IPS prediction implementation."""
from abc import abstractmethod
from ..base import Algorithm
from ...data_access import FingerprintInputs
from ...experiments.contracts import IPSPredictions


class LLMAlgorithm(Algorithm):
    @abstractmethod
    def encode_rssi(self, inputs: FingerprintInputs):
        """Define AP/missing-signal serialization and the token budget."""

    @abstractmethod
    def decode_position(self, response, inputs: FingerprintInputs) -> IPSPredictions:
        """Define label mapping, coordinate units and invalid-response handling."""
