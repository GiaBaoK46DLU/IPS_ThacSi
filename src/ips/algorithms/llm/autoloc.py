"""AutoLoC: Large Language Model-based Zero-Shot Indoor Positioning System.

Implements the methodology from the AutoLoC paper:
1. RSSI missing-signal handling (-105 dBm threshold)
2. Standardization + 15-D PCA latent space
3. Diversity-Aware Vector Memory Retrieval (Top-5 Euclidean KNN)
4. Context prompt synthesis (Top-5 neighbors + Top-3 strongest WAPs)
5. Physical continuity and vertical leakage conflict resolution via Local LLM (Ollama)
6. Robust JSON decoding with inverse-distance weighted KNN fallback
"""
from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


@dataclass
class NeighborRecord:
    index: int
    distance: float
    building: int
    floor: int
    space_id: int
    latitude: float
    longitude: float


@dataclass
class AutoLoCPrediction:
    predicted_building: int
    predicted_floor: int
    predicted_space: int
    predicted_lat: float
    predicted_lon: float
    raw_response: str = ""
    fallback_used: bool = False
    latency_sec: float = 0.0


@dataclass
class ScarcityResult:
    split_ratio: float
    train_size: int
    test_size: int
    building_acc: float
    floor_acc: float
    combined_acc: float
    space_acc: float
    mean_error_meters: float
    median_error_meters: float
    fallback_rate: float
    total_time_sec: float


class AutoLoCEngine:
    """Core AutoLoC Engine integrating PCA latent retrieval with Local LLM reasoning."""

    PROMPT_TEMPLATE = """AutoLoC IPS Reasoning:
You are an expert Indoor Positioning System (IPS) reasoner.
Analyze physical signal continuity (stronger RSSI = closer to transmitter) and resolve vertical floor leakage conflicts (signals penetrating ceilings/floors between adjacent floors).

Query Strongest Detected WAPs:
{query_waps}

Top-5 Latent Retrieved Neighbors from Vector Memory:
{neighbors_block}

Task:
Synthesize the neighbor spatial consensus, resolve any vertical floor leakage conflicts, and predict the query location.
Output STRICTLY a single JSON object with these exact keys and types:
{{"predicted_building": <int>, "predicted_floor": <int>, "predicted_space": <int>, "predicted_lat": <float>, "predicted_lon": <float>}}"""

    def __init__(
        self,
        n_components: int = 15,
        n_neighbors: int = 5,
        missing_val: float = 100.0,
        replace_val: float = -105.0,
        model_name: str = "llama3.2:1b",
        ollama_url: str = "http://localhost:11434/api/generate",
        seed: int = 42,
    ):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.missing_val = missing_val
        self.replace_val = replace_val
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.seed = seed

        self.scaler = StandardScaler()
        self.pca = PCA(n_components=self.n_components, random_state=self.seed)
        self.knn = NearestNeighbors(n_neighbors=self.n_neighbors, metric="euclidean")

        self.memory_latent: Optional[np.ndarray] = None
        self.memory_meta: Optional[pd.DataFrame] = None
        self.wap_cols: List[str] = []
        self.is_fitted = False

    def preprocess_rssi(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Replace missing/weak signals (+100 dBm) with replace_val (-105 dBm)."""
        if isinstance(X, pd.DataFrame):
            arr = X.to_numpy(dtype=np.float32)
        else:
            arr = np.asarray(X, dtype=np.float32).copy()

        # Handle +100 as missing
        arr[arr == self.missing_val] = self.replace_val
        # Also clamp any signals <= replace_val
        arr[arr < self.replace_val] = self.replace_val
        return arr

    def fit_memory(
        self,
        train_df: pd.DataFrame,
        wap_cols: Optional[List[str]] = None,
    ) -> None:
        """Fit StandardScaler, 15-D PCA, and build KNN index on the memory database."""
        if wap_cols is None:
            self.wap_cols = [c for c in train_df.columns if c.startswith("WAP")]
        else:
            self.wap_cols = list(wap_cols)

        X_raw = train_df[self.wap_cols].values
        X_clean = self.preprocess_rssi(X_raw)

        # 1. Standardization
        X_scaled = self.scaler.fit_transform(X_clean)

        # 2. 15-D PCA Latent Space
        self.memory_latent = self.pca.fit_transform(X_scaled)

        # 3. Vector Memory KNN Index
        self.knn.fit(self.memory_latent)

        # Store metadata
        meta_cols = ["BUILDINGID", "FLOOR", "SPACEID", "LATITUDE", "LONGITUDE"]
        self.memory_meta = train_df[meta_cols].copy().reset_index(drop=True)
        self.is_fitted = True

    def get_query_strongest_waps(self, query_rssi: np.ndarray, top_k: int = 3) -> str:
        """Find the Top-K strongest detected WAPs for the query."""
        # Detectable signals are > replace_val (-105) and != 100
        valid_mask = (query_rssi > self.replace_val) & (query_rssi != self.missing_val)
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) == 0:
            return "No active WAPs detected"

        # Sort by RSSI descending
        sorted_valid = valid_indices[np.argsort(-query_rssi[valid_indices])]
        top_indices = sorted_valid[:top_k]

        items = []
        for idx in top_indices:
            wap_name = self.wap_cols[idx] if idx < len(self.wap_cols) else f"WAP_{idx}"
            items.append(f"{wap_name}: {query_rssi[idx]:.1f}dBm")
        return ", ".join(items)

    def retrieve_neighbors(self, query_clean_rssi: np.ndarray) -> Tuple[List[NeighborRecord], np.ndarray]:
        """Project query into 15-D PCA latent space and retrieve Top-K nearest neighbors."""
        if not self.is_fitted or self.memory_meta is None:
            raise RuntimeError("AutoLoCEngine must be fitted before retrieval.")

        q_scaled = self.scaler.transform(query_clean_rssi.reshape(1, -1))
        q_latent = self.pca.transform(q_scaled)

        distances, indices = self.knn.kneighbors(q_latent)
        distances = distances[0]
        indices = indices[0]

        neighbors = []
        for dist, idx in zip(distances, indices):
            row = self.memory_meta.iloc[idx]
            neighbors.append(
                NeighborRecord(
                    index=int(idx),
                    distance=float(dist),
                    building=int(row["BUILDINGID"]),
                    floor=int(row["FLOOR"]),
                    space_id=int(row["SPACEID"]),
                    latitude=float(row["LATITUDE"]),
                    longitude=float(row["LONGITUDE"]),
                )
            )
        return neighbors, q_latent

    def format_prompt(self, top_waps_str: str, neighbors: List[NeighborRecord]) -> str:
        """Format the RAG context and instructions for the LLM."""
        lines = []
        for i, n in enumerate(neighbors, 1):
            lines.append(
                f"N{i}: dist={n.distance:.2f}, Bld={n.building}, Flr={n.floor}, "
                f"Spc={n.space_id}, Lat={n.latitude:.2f}, Lon={n.longitude:.2f}"
            )
        neighbors_block = "\n".join(lines)
        return self.PROMPT_TEMPLATE.format(query_waps=top_waps_str, neighbors_block=neighbors_block)

    def call_ollama(self, prompt: str, timeout: float = 30.0) -> str:
        """Call local Ollama API for zero-shot reasoning."""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "num_predict": 70,
                "temperature": 0.0,
            },
        }
        resp = requests.post(self.ollama_url, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json().get("response", "")

    def fallback_knn_prediction(self, neighbors: List[NeighborRecord]) -> AutoLoCPrediction:
        """Inverse-distance weighted KNN fallback if LLM response is malformed."""
        eps = 1e-5
        weights = [1.0 / (n.distance + eps) for n in neighbors]
        total_w = sum(weights)

        # Weighted coordinates
        lat = sum(n.latitude * w for n, w in zip(neighbors, weights)) / total_w
        lon = sum(n.longitude * w for n, w in zip(neighbors, weights)) / total_w

        # Nearest neighbor for building, floor, space
        best_n = neighbors[0]
        return AutoLoCPrediction(
            predicted_building=best_n.building,
            predicted_floor=best_n.floor,
            predicted_space=best_n.space_id,
            predicted_lat=lat,
            predicted_lon=lon,
            raw_response="FALLBACK_KNN",
            fallback_used=True,
        )

    def parse_llm_response(self, raw: str, neighbors: List[NeighborRecord]) -> AutoLoCPrediction:
        """Parse strict JSON response with regex and fallback protection."""
        clean = raw.strip()
        # Find json block if wrapped in markdown
        match = re.search(r"\{.*?\}", clean, re.DOTALL)
        if match:
            clean = match.group(0)

        try:
            data = json.loads(clean)
            bld = int(data["predicted_building"])
            flr = int(data["predicted_floor"])
            spc = int(data.get("predicted_space", neighbors[0].space_id))
            lat = float(data["predicted_lat"])
            lon = float(data["predicted_lon"])

            # Basic sanity check on lat/lon range
            if not np.isfinite(lat) or not np.isfinite(lon):
                raise ValueError("Infinite lat/lon")

            return AutoLoCPrediction(
                predicted_building=bld,
                predicted_floor=flr,
                predicted_space=spc,
                predicted_lat=lat,
                predicted_lon=lon,
                raw_response=raw,
                fallback_used=False,
            )
        except Exception:
            fb = self.fallback_knn_prediction(neighbors)
            fb.raw_response = raw
            return fb

    def predict_single(self, query_raw_rssi: np.ndarray) -> AutoLoCPrediction:
        """Run the complete AutoLoC pipeline for a single query."""
        t0 = time.time()
        q_clean = self.preprocess_rssi(query_raw_rssi)
        top_waps = self.get_query_strongest_waps(query_raw_rssi, top_k=3)
        neighbors, _ = self.retrieve_neighbors(q_clean)
        prompt = self.format_prompt(top_waps, neighbors)

        try:
            raw_text = self.call_ollama(prompt)
            pred = self.parse_llm_response(raw_text, neighbors)
        except Exception as e:
            pred = self.fallback_knn_prediction(neighbors)
            pred.raw_response = f"ERROR: {str(e)}"

        pred.latency_sec = time.time() - t0
        return pred

    def predict_batch(
        self,
        X_queries: np.ndarray,
        max_workers: int = 4,
        on_progress: Optional[callable] = None,
    ) -> List[AutoLoCPrediction]:
        """Run batch inference concurrently across local Ollama instances."""
        n = len(X_queries)
        results: List[Optional[AutoLoCPrediction]] = [None] * n

        def _worker(i: int):
            return i, self.predict_single(X_queries[i])

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_worker, i) for i in range(n)]
            completed = 0
            for fut in as_completed(futures):
                idx, pred = fut.result()
                results[idx] = pred
                completed += 1
                if on_progress:
                    on_progress(completed, n)

        return [r for r in results if r is not None]


def compute_metrics(
    predictions: List[AutoLoCPrediction],
    ground_truth_df: pd.DataFrame,
) -> Dict[str, float]:
    """Calculate Classification Accuracy and 2D Mean Positioning Error."""
    n = len(predictions)
    true_bld = ground_truth_df["BUILDINGID"].to_numpy(dtype=int)
    true_flr = ground_truth_df["FLOOR"].to_numpy(dtype=int)
    true_spc = ground_truth_df["SPACEID"].to_numpy(dtype=int)
    true_lat = ground_truth_df["LATITUDE"].to_numpy(dtype=float)
    true_lon = ground_truth_df["LONGITUDE"].to_numpy(dtype=float)

    pred_bld = np.array([p.predicted_building for p in predictions], dtype=int)
    pred_flr = np.array([p.predicted_floor for p in predictions], dtype=int)
    pred_spc = np.array([p.predicted_space for p in predictions], dtype=int)
    pred_lat = np.array([p.predicted_lat for p in predictions], dtype=float)
    pred_lon = np.array([p.predicted_lon for p in predictions], dtype=float)

    bld_acc = float(np.mean(pred_bld == true_bld) * 100.0)
    flr_acc = float(np.mean(pred_flr == true_flr) * 100.0)
    comb_acc = float(np.mean((pred_bld == true_bld) & (pred_flr == true_flr)) * 100.0)

    # Space accuracy (check if ground truth has non-zero space labels)
    if (true_spc > 0).any():
        space_acc = float(np.mean(pred_spc == true_spc) * 100.0)
    else:
        space_acc = 0.0  # masked in official validation set

    # Euclidean 2D distance error in meters
    errors = np.sqrt((pred_lat - true_lat) ** 2 + (pred_lon - true_lon) ** 2)
    mean_error = float(np.mean(errors))
    median_error = float(np.median(errors))
    fallback_rate = float(np.mean([p.fallback_used for p in predictions]) * 100.0)

    return {
        "building_acc": bld_acc,
        "floor_acc": flr_acc,
        "combined_acc": comb_acc,
        "space_acc": space_acc,
        "mean_error_meters": mean_error,
        "median_error_meters": median_error,
        "fallback_rate": fallback_rate,
    }

