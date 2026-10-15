"""Team-owned LLM contract and concrete AutoLoC implementation."""
from __future__ import annotations

import copy
from abc import abstractmethod
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pandas as pd

from ..base import Algorithm
from ...data_access.contracts import FingerprintInputs, FingerprintSplit
from ...experiments.contracts import AlgorithmExecution, ExperimentConfig, IPSPredictions
from .autoloc import AutoLoCEngine, AutoLoCPrediction


class LLMAlgorithm(Algorithm):
    """Abstract base class for LLM-based IPS algorithms."""

    @abstractmethod
    def encode_rssi(self, inputs: FingerprintInputs):
        """Define AP/missing-signal serialization and the token budget."""

    @abstractmethod
    def decode_position(self, response, inputs: FingerprintInputs) -> IPSPredictions:
        """Define label mapping, coordinate units and invalid-response handling."""


class LLMAutoLoCAlgorithm(LLMAlgorithm):
    """Concrete AutoLoC zero-shot IPS implementation integrated with local Ollama.

    Implements:
    - 15-D PCA latent space
    - Diversity-Aware Top-5 KNN vector memory retrieval
    - Prompt synthesis with Query Top-3 WAPs + Top-5 neighbors
    - Local LLM inference (Ollama)
    - Fallback decoder
    """

    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        params = self.config.params or {}
        self.model_name = params.get("model", "llama3.2:1b")
        self.ollama_url = params.get("ollama_url", "http://localhost:11434/api/generate")
        self.n_components = params.get("n_components", 15)
        self.n_neighbors = params.get("n_neighbors", 5)
        self.max_workers = params.get("max_workers", 4)
        self.seed = self.config.seed

        self.engine = AutoLoCEngine(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            model_name=self.model_name,
            ollama_url=self.ollama_url,
            seed=self.seed,
        )

        self._train: Optional[FingerprintSplit] = None
        self._classes: Optional[List[dict]] = None
        self._wap_cols: List[str] = []

    def prepare(self, train: FingerprintSplit) -> None:
        """Fit StandardScaler, 15-D PCA and build Vector Memory on train only."""
        if train.name != "train":
            raise ValueError("AutoLoC preparation requires the train split.")
        self._train = train
        self._classes = train.info.classes

        # Reconstruct DataFrame with WAP cols and metadata for the AutoLoCEngine
        n_waps = train.inputs.rssi.shape[1]
        self._wap_cols = [f"WAP{i+1:03d}" for i in range(n_waps)]

        buildings = np.array([self._classes[cid]["building"] for cid in train.class_ids], dtype=int)
        floors = np.array([self._classes[cid]["floor"] for cid in train.class_ids], dtype=int)
        
        # Build DataFrame
        df_dict = {col: train.inputs.rssi[:, i] for i, col in enumerate(self._wap_cols)}
        df_dict["BUILDINGID"] = buildings
        df_dict["FLOOR"] = floors
        df_dict["SPACEID"] = np.zeros(len(buildings), dtype=int)  # fallback space if not tracked
        df_dict["LATITUDE"] = train.xy[:, 1]
        df_dict["LONGITUDE"] = train.xy[:, 0]

        train_df = pd.DataFrame(df_dict)
        self.engine.fit_memory(train_df, wap_cols=self._wap_cols)

    def execute(self, validation: FingerprintSplit | None) -> AlgorithmExecution:
        """Freeze inference pipeline and store model configuration artifacts."""
        if not self.engine.is_fitted:
            raise RuntimeError("Call prepare(train) before execute().")

        output = self.config.output_dir / "algorithm"
        output.mkdir(parents=True, exist_ok=True)

        info_path = output / "autoloc_config.json"
        import json
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model": self.model_name,
                    "ollama_url": self.ollama_url,
                    "n_components": self.n_components,
                    "n_neighbors": self.n_neighbors,
                    "seed": self.seed,
                },
                f,
                indent=2,
            )

        artifacts = {"config": info_path}
        return AlgorithmExecution(
            finalized=True,
            run_type="autoloc-zero-shot-rag",
            history=None,
            artifacts=artifacts,
            resources={"model": self.model_name, "backend": "ollama"},
        )

    def encode_rssi(self, inputs: FingerprintInputs):
        """Encode query fingerprints into prompts for LLM."""
        prompts = []
        clean_rssi = self.engine.preprocess_rssi(inputs.rssi)
        for i in range(len(inputs)):
            top_waps = self.engine.get_query_strongest_waps(inputs.rssi[i], top_k=3)
            neighbors, _ = self.engine.retrieve_neighbors(clean_rssi[i])
            prompt = self.engine.format_prompt(top_waps, neighbors)
            prompts.append((prompt, neighbors))
        return prompts

    def decode_position(self, responses: List[AutoLoCPrediction], inputs: FingerprintInputs) -> IPSPredictions:
        """Map AutoLoC predictions into standard IPSPredictions."""
        pred_coords = np.array([[p.predicted_lon, p.predicted_lat] for p in responses], dtype=np.float64)
        pred_bld = np.array([p.predicted_building for p in responses], dtype=np.int64)
        pred_flr = np.array([p.predicted_floor for p in responses], dtype=np.int64)

        mapping = {(c["building"], c["floor"]): i for i, c in enumerate(self._classes or [])}
        class_ids = []
        for b, f in zip(pred_bld, pred_flr):
            cid = mapping.get((int(b), int(f)), 0)
            class_ids.append(cid)
        class_ids = np.array(class_ids, dtype=np.int64)

        return IPSPredictions(
            row_indices=inputs.row_indices.copy(),
            source_id=inputs.source_id,
            xy=pred_coords,
            class_ids=class_ids,
            buildings=pred_bld,
            floors=pred_flr,
        )

    def predict(self, inputs: FingerprintInputs) -> IPSPredictions:
        """Predict building, floor, and coordinates for input fingerprints."""
        if not self.engine.is_fitted:
            raise RuntimeError("Call execute() before predict().")

        # Run concurrent batch prediction
        predictions = self.engine.predict_batch(
            inputs.rssi,
            max_workers=self.max_workers,
        )

        return self.decode_position(predictions, inputs)


class HybridRFAutoLoCAlgorithm(LLMAutoLoCAlgorithm):
    """Hybrid Cascaded Random Forest + Floor-Locked AutoLoC LLM Algorithm."""

    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        from .hybrid_autoloc import HybridRFAutoLoCEngine
        params = self.config.params or {}
        self.engine = HybridRFAutoLoCEngine(
            n_estimators=params.get("n_estimators", 50),
            n_components=params.get("n_components", 15),
            n_neighbors=params.get("n_neighbors", 5),
            knn_k=params.get("knn_k", 9),
            model_name=self.model_name,
            ollama_url=self.ollama_url,
            seed=self.seed,
        )

    def prepare(self, train: FingerprintSplit) -> None:
        if train.name != "train":
            raise ValueError("Hybrid RF+AutoLoC preparation requires the train split.")
        self._train = train
        self._classes = train.info.classes

        n_waps = train.inputs.rssi.shape[1]
        self._wap_cols = [f"WAP{i+1:03d}" for i in range(n_waps)]

        buildings = np.array([self._classes[cid]["building"] for cid in train.class_ids], dtype=int)
        floors = np.array([self._classes[cid]["floor"] for cid in train.class_ids], dtype=int)

        df_dict = {col: train.inputs.rssi[:, i] for i, col in enumerate(self._wap_cols)}
        df_dict["BUILDINGID"] = buildings
        df_dict["FLOOR"] = floors
        df_dict["SPACEID"] = np.zeros(len(buildings), dtype=int)
        df_dict["LATITUDE"] = train.xy[:, 1]
        df_dict["LONGITUDE"] = train.xy[:, 0]

        train_df = pd.DataFrame(df_dict)
        self.engine.fit(train_df, wap_cols=self._wap_cols)

