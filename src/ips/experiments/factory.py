"""Registry factory at the IPS algorithm level; no fake LLM/HUIM registration."""
import copy
import inspect

from ..algorithms.base import Algorithm
from .contracts import ExperimentConfig


class UnknownAlgorithmError(ValueError):
    pass


class AlgorithmNotImplementedError(NotImplementedError):
    pass


class AlgorithmFactory:
    unavailable = {"llm": "LLM adapter has no RSSI-to-position implementation yet",
                   "huim": "HUIM adapter has no itemset-to-position implementation yet",
                   "fastkan": "FastKAN is not implemented; current KAN uses efficient-kan B-splines",
                   "s_kan": "S-KAN is not implemented",
                   "fast_kan_mixer": "FastKAN-Mixer is not implemented",
                   "s_kan_mixer": "S-KAN-Mixer is not implemented"}

    def __init__(self):
        self._registry: dict[str, type[Algorithm]] = {}

    def register(self, algorithm_id: str, implementation: type[Algorithm]) -> None:
        if not isinstance(algorithm_id, str) or not algorithm_id.strip():
            raise ValueError("Registry ID must be a nonempty string.")
        if (not isinstance(implementation, type) or not issubclass(implementation, Algorithm) or
                inspect.isabstract(implementation)):
            raise TypeError("Register a concrete Algorithm subclass, not an abstract adapter or nn.Module.")
        if algorithm_id in self._registry:
            raise ValueError(f"Algorithm ID already registered: {algorithm_id}")
        self._registry[algorithm_id] = implementation

    def available_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._registry))

    def create(self, algorithm_id: str, config: ExperimentConfig) -> Algorithm:
        if algorithm_id != config.algorithm_id:
            raise ValueError("Factory ID differs from ExperimentConfig.algorithm_id.")
        implementation = self._registry.get(algorithm_id)
        if implementation is None:
            if algorithm_id in ("mlp", "mlp_mixer"):
                raise UnknownAlgorithmError("MLP/MLP-Mixer are personal study baselines, excluded from the shared Factory. Choose kan_mixer.")
            if algorithm_id in self.unavailable:
                raise AlgorithmNotImplementedError(self.unavailable[algorithm_id])
            raise UnknownAlgorithmError(f"Unregistered algorithm_id '{algorithm_id}'. Available: {', '.join(self.available_ids())}")
        return implementation(copy.deepcopy(config))


def default_factory() -> AlgorithmFactory:
    from ..algorithms.kan import KANMixerAlgorithm
    factory = AlgorithmFactory()
    factory.register("kan_mixer", KANMixerAlgorithm)
    return factory
