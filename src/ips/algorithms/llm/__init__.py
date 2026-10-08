"""Extension point only, intentionally not in the default registry."""
from .adapter import LLMAlgorithm

__all__ = ["LLMAlgorithm"]
