"""Cấu hình và kết quả IPS, không phụ thuộc torch.nn.Module."""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from ..data import OUTPUT, ROOT
from ..data_access import FingerprintInputs


@dataclass(frozen=True)
class ExperimentConfig:
    algorithm_id: str
    output_dir: Path
    params: dict[str, Any] = field(default_factory=dict)
    split_dir: Path = OUTPUT
    task: Literal["auto", "position", "classification", "coordinates"] = "auto"
    seed: int = 42
    device: str = "cpu"
    smoke: bool = False
    evaluation_splits: tuple[str, ...] = ("validation",)
    export_plots: bool = True

    def __post_init__(self):
        if not isinstance(self.algorithm_id, str) or not self.algorithm_id.strip():
            raise ValueError("algorithm_id must be a nonempty string.")
        if not isinstance(self.params, dict):
            raise ValueError("params must be an object owned by the selected algorithm.")
        try:
            json.dumps(self.params, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError("params must contain finite JSON-serializable values for reproducible artifacts.") from error
        if self.task not in ("auto", "position", "classification", "coordinates"):
            raise ValueError("Unsupported IPS task.")
        splits = tuple(self.evaluation_splits)
        if not splits or len(set(splits)) != len(splits) or set(splits) - {"validation", "test"}:
            raise ValueError("evaluation_splits must contain distinct validation/test entries.")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer.")
        if any(type(v) is not bool for v in (self.smoke, self.export_plots)):
            raise ValueError("smoke and export_plots must be booleans.")
        if self.smoke and "test" in splits:
            raise ValueError("Smoke runs cannot evaluate the official test.")
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("device must be a nonempty string.")
        for name in ("output_dir", "split_dir"):
            value = Path(getattr(self, name))
            object.__setattr__(self, name, (value if value.is_absolute() else ROOT/value).resolve())
        object.__setattr__(self, "evaluation_splits", splits)
        object.__setattr__(self, "params", copy.deepcopy(self.params))

    @classmethod
    def from_json(cls, path: Path) -> ExperimentConfig:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Experiment configuration must be a JSON object.")
        unknown = set(payload) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown experiment fields: {sorted(unknown)}")
        try:
            return cls(**payload)
        except TypeError as error:
            raise ValueError(f"Invalid experiment configuration: {error}") from error

    def snapshot(self) -> dict[str, Any]:
        return {"algorithm_id": self.algorithm_id, "output_dir": str(self.output_dir),
                "split_dir": str(self.split_dir), "task": self.task, "params": copy.deepcopy(self.params),
                "seed": self.seed, "device": self.device, "smoke": self.smoke,
                "evaluation_splits": list(self.evaluation_splits), "export_plots": self.export_plots}


@dataclass
class IPSPredictions:
    """Original coordinate metres; no normalized xy, logits or true labels."""
    row_indices: np.ndarray
    source_id: str
    xy: np.ndarray | None = None
    class_ids: np.ndarray | None = None
    buildings: np.ndarray | None = None
    floors: np.ndarray | None = None
    confidence: np.ndarray | None = None

    def validate(self, inputs: FingerprintInputs, classes: list[dict]) -> IPSPredictions:
        count = len(inputs)
        rows = np.asarray(self.row_indices)
        if (self.source_id != inputs.source_id or not np.issubdtype(rows.dtype, np.integer) or
                not np.array_equal(rows, inputs.row_indices)):
            raise ValueError("Prediction source/row order differs from requested fingerprints.")
        if all(v is None for v in (self.xy, self.class_ids, self.buildings, self.floors)):
            raise ValueError("Algorithm returned no IPS output.")
        if self.xy is not None:
            self.xy = np.asarray(self.xy, dtype=np.float64)
            if self.xy.shape != (count, 2) or not np.isfinite(self.xy).all():
                raise ValueError("Predicted xy must be finite original-coordinate [N,2].")
        for name in ("class_ids", "buildings", "floors"):
            value = getattr(self, name)
            if value is not None:
                value = np.asarray(value)
                if value.shape != (count,) or not np.issubdtype(value.dtype, np.integer):
                    raise ValueError(f"{name} must be an integer vector[N].")
                setattr(self, name, value.astype(np.int64))
        pairs = np.asarray([(c["building"], c["floor"]) for c in classes], dtype=np.int64)
        if self.class_ids is not None:
            if ((self.class_ids < 0) | (self.class_ids >= len(classes))).any():
                raise ValueError("Predicted class ID outside shared mapping.")
            for axis, name in enumerate(("buildings", "floors")):
                expected = pairs[self.class_ids, axis]
                if getattr(self, name) is not None and not np.array_equal(getattr(self, name), expected):
                    raise ValueError("Predicted class and building/floor disagree.")
                setattr(self, name, expected)
        elif self.buildings is not None and self.floors is not None:
            mapping = {tuple(p): i for i, p in enumerate(pairs)}
            try:
                self.class_ids = np.asarray([mapping[(int(b), int(f))]
                    for b, f in zip(self.buildings, self.floors)], dtype=np.int64)
            except KeyError as error:
                raise ValueError("Predicted building/floor pair is not in shared mapping.") from error
        for axis, name in enumerate(("buildings", "floors")):
            if getattr(self, name) is not None and not np.isin(getattr(self, name), pairs[:, axis]).all():
                raise ValueError(f"{name} outside shared label domain.")
        if self.confidence is not None:
            self.confidence = np.asarray(self.confidence, dtype=np.float64)
            if (self.confidence.shape != (count,) or not np.isfinite(self.confidence).all() or
                    ((self.confidence < 0) | (self.confidence > 1)).any() or
                    all(v is None for v in (self.class_ids, self.buildings, self.floors))):
                raise ValueError("Confidence must be finite [0,1] for an actual classification output.")
        self.row_indices = rows.astype(np.int64)
        return self


@dataclass(frozen=True)
class ModelDiagram:
    content: str
    format: Literal["mermaid", "text"] = "text"

    def __post_init__(self):
        if self.format not in ("mermaid", "text") or not isinstance(self.content, str) or not self.content.strip():
            raise ValueError("Diagram must have nonempty content and mermaid/text format.")


@dataclass
class AlgorithmExecution:
    finalized: bool = False  # predict/test only after execute freezes the selected state
    run_type: str = "execution"
    history: list[dict[str, Any]] | None = None
    artifacts: dict[str, Path] = field(default_factory=dict)
    resources: dict[str, Any] | None = None


@dataclass
class RunResult:
    algorithm_id: str
    predictions: dict[str, IPSPredictions]
    metrics: dict[str, dict[str, Any]]
    matrices: dict[str, dict[str, list[list[int]]]]
    runtime: dict[str, float]
    log_path: Path
    config_snapshot: dict[str, Any]
    run_type: str
    history: list[dict[str, Any]] | None = None
    artifacts: dict[str, Path] = field(default_factory=dict)
    resources: dict[str, Any] | None = None
    diagram: ModelDiagram | None = None

    def summary(self) -> dict[str, Any]:
        return {"algorithm_id": self.algorithm_id, "run_type": self.run_type,
                "predictions": {split: {"count": len(p.row_indices), "source_id": p.source_id,
                    "outputs": [name for name in ("xy", "class_ids", "buildings", "floors", "confidence")
                                if getattr(p, name) is not None]} for split, p in self.predictions.items()},
                "metrics": self.metrics, "matrices": self.matrices, "runtime": self.runtime,
                "log_path": str(self.log_path), "config_snapshot": self.config_snapshot,
                "history": self.history, "resources": self.resources,
                "diagram": None if self.diagram is None else {"format": self.diagram.format},
                "artifacts": {k: str(v) for k, v in self.artifacts.items()}}
