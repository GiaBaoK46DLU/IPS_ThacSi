"""Team-owned mining contract; discovered itemsets alone are not IPS predictions."""
from abc import abstractmethod
from ..base import Algorithm
from ...data_access import FingerprintInputs
from ...experiments.contracts import IPSPredictions


class HUIMAlgorithm(Algorithm):
    @abstractmethod
    def encode_transactions(self, inputs: FingerprintInputs):
        """Define AP items, train-fitted bins and the utility function."""

    @abstractmethod
    def itemsets_to_position(self, inputs: FingerprintInputs) -> IPSPredictions:
        """Define how mined itemsets produce class IDs and/or metre coordinates."""
