"""Hybrid RF + AutoLoC: Cascaded Random Forest + Floor-Locked Latent RAG with Local LLM.

Combines the macro-level classification strength of Hierarchical Random Forest
with the micro-level spatial reasoning of AutoLoC (15-D PCA Latent Space + RAG + Local LLM via Ollama).

Mathematical Pipeline:
1. Dual Signal Preprocessing:
   - Positive Shift & Clip: X' = max(0, X + 110), and non-linear power scaling X'_pow = (X')^1.5
     to amplify strong line-of-sight APs and suppress weak cross-floor leakage.
   - Standardized AutoLoC Space: Replace +100 with -105 dBm -> Z-score -> 15-D PCA Latent Space.
2. Cascaded Macro-Locking:
   - RF_building(X') -> B_hat
   - RF_floor([X'^1.5, B_hat]) -> F_hat
3. Floor-Locked Dual-Space RAG Retrieval:
   - Restricts Vector Memory to subspace M_(B_hat, F_hat) = {m | B_m == B_hat and F_m == F_hat}.
   - Computes composite distance combining normalized Manhattan L1 and 15-D PCA Euclidean distance.
   - Applies non-linear inverse distance weighting w_k = (d_L1 + 1)^(-2.0) and spatial outlier
     attenuation exp(-||p_k - p_med||_2 / sigma_s).
4. Confidence-Gated Local LLM Reasoning (Ollama):
   - Computes spatial dispersion sigma_xy of the Top-5 Floor-Locked neighbors.
   - For high-dispersion / ambiguous queries (sigma_xy > gate_threshold), invokes Local LLM
     (Ollama llama3.2:1b / phi3:mini) with Query Top-3 strongest WAPs and Floor-Locked neighbors
     to resolve multi-room ambiguity within the locked floor.
"""
from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from .autoloc import AutoLoCPrediction, NeighborRecord


@dataclass
class ComparisonMetrics:
    method_name: str
    split_name: str
    split_ratio: float
    train_size: int
    test_size: int
    building_acc: float
    floor_acc: float
    combined_acc: float
    space_acc: float
    mean_error_meters: float
    median_error_meters: float
    p90_error_meters: float
    conditional_mean_error: float
    fallback_rate: float
    total_time_sec: float


class HybridRFAutoLoCEngine:
    """Hybrid Cascaded RF + Floor-Locked AutoLoC LLM Engine."""

    HYBRID_PROMPT_TEMPLATE = """AutoLoC Floor-Locked IPS Reasoning:
Macro-Classifier Locked State: Building={locked_bld}, Floor={locked_flr}
Query Top-3 Strongest WAPs: {query_waps}
Top-5 Floor-Locked Neighbors (same Building & Floor):
{neighbors_block}
Spatial Prior Anchor: Spc={prior_space}, Lat={prior_lat:.2f}, Lon={prior_lon:.2f}

Task: Analyze physical signal continuity among the floor-locked neighbors and refine the 2D coordinates (within neighbor bounds). Keep Building={locked_bld} and Floor={locked_flr}. Output strictly JSON:
{{"predicted_building": {locked_bld}, "predicted_floor": {locked_flr}, "predicted_space": {prior_space}, "predicted_lat": {prior_lat:.2f}, "predicted_lon": {prior_lon:.2f}}}"""

    def __init__(
        self,
        n_estimators: int = 100,
        n_components: int = 15,
        n_neighbors: int = 5,
        knn_k: int = 7,
        power_gamma: float = 1.5,
        idw_exponent: float = 2.0,
        spatial_sigma: float = 25.0,
        llm_gate_threshold: float = 50.0,
        model_name: str = "qwen2.5:3b",
        ollama_url: str = "http://localhost:11434/api/generate",
        seed: int = 42,
    ):
        self.n_estimators = n_estimators
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.knn_k = knn_k
        self.power_gamma = power_gamma
        self.idw_exponent = idw_exponent
        self.spatial_sigma = spatial_sigma
        self.llm_gate_threshold = llm_gate_threshold
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.seed = seed

        # Stage 1: Cascaded RF Classifiers
        self.clf_building: Optional[RandomForestClassifier] = None
        self.clf_floor: Optional[RandomForestClassifier] = None

        # Stage 2: Latent PCA & Scaler
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=self.n_components, random_state=self.seed)

        # Memory structures
        self.wap_cols: List[str] = []
        self.memory_pos_rssi: Optional[np.ndarray] = None
        self.memory_latent: Optional[np.ndarray] = None
        self.memory_meta: Optional[pd.DataFrame] = None
        self.floor_groups: Dict[Tuple[int, int], np.ndarray] = {}
        self.is_fitted = False

    @staticmethod
    def preprocess_positive_shift(X: np.ndarray) -> np.ndarray:
        """Transform RSSI via Shift & Clip: X' = max(0, X + 110)."""
        arr = np.asarray(X, dtype=np.float32).copy()
        arr = np.where(arr == 100, -110.0, arr)
        arr = np.clip(arr + 110.0, a_min=0.0, a_max=None)
        return arr

    @staticmethod
    def preprocess_autoloc_standard(X: np.ndarray) -> np.ndarray:
        """Transform RSSI for AutoLoC PCA: replace +100 with -105 dBm."""
        arr = np.asarray(X, dtype=np.float32).copy()
        arr[arr == 100] = -105.0
        arr[arr < -105.0] = -105.0
        return arr

    def fit(self, train_df: pd.DataFrame, wap_cols: Optional[List[str]] = None) -> None:
        """Fit Cascaded RF Classifiers, 15-D PCA Latent Space, and Partitioned Floor Memory."""
        if wap_cols is None:
            self.wap_cols = [c for c in train_df.columns if c.startswith("WAP")]
        else:
            self.wap_cols = list(wap_cols)

        X_raw = train_df[self.wap_cols].to_numpy(dtype=np.float32)

        # 1. Positive shifted features for Cascaded RF & L1 retrieval
        X_pos = self.preprocess_positive_shift(X_raw)
        X_pow = X_pos ** self.power_gamma
        self.memory_pos_rssi = X_pos

        buildings = train_df["BUILDINGID"].to_numpy(dtype=int)
        floors = train_df["FLOOR"].to_numpy(dtype=int)

        # Train RF Building
        self.clf_building = RandomForestClassifier(
            n_estimators=self.n_estimators, random_state=self.seed, n_jobs=-1
        )
        self.clf_building.fit(X_pos, buildings)

        # Train RF Floor (Cascaded with Building + Non-linear Power Scaling)
        X_floor_feat = np.column_stack((X_pow, buildings))
        self.clf_floor = RandomForestClassifier(
            n_estimators=self.n_estimators, random_state=self.seed, n_jobs=-1
        )
        self.clf_floor.fit(X_floor_feat, floors)

        # 2. AutoLoC 15-D PCA Latent Space
        X_std_raw = self.preprocess_autoloc_standard(X_raw)
        X_scaled = self.scaler.fit_transform(X_std_raw)
        n_comp = min(self.n_components, X_scaled.shape[0], X_scaled.shape[1])
        self.pca = PCA(n_components=n_comp, random_state=self.seed)
        self.memory_latent = self.pca.fit_transform(X_scaled)

        # 3. Store metadata and build Floor-Locked Subspace partitions
        meta_cols = ["BUILDINGID", "FLOOR", "SPACEID", "LATITUDE", "LONGITUDE"]
        self.memory_meta = train_df[meta_cols].copy().reset_index(drop=True)

        self.floor_groups = {}
        for idx, (b, f) in enumerate(zip(buildings, floors)):
            key = (int(b), int(f))
            if key not in self.floor_groups:
                self.floor_groups[key] = []
            self.floor_groups[key].append(idx)
        for key in self.floor_groups:
            self.floor_groups[key] = np.array(self.floor_groups[key], dtype=int)

        self.is_fitted = True

    def get_query_strongest_waps(self, query_raw_rssi: np.ndarray, top_k: int = 3) -> str:
        """Extract Top-K strongest detected WAPs for physical context."""
        valid_mask = (query_raw_rssi > -105.0) & (query_raw_rssi != 100.0)
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) == 0:
            return "No active WAPs"
        sorted_valid = valid_indices[np.argsort(-query_raw_rssi[valid_indices])]
        top_indices = sorted_valid[:top_k]
        return ", ".join(
            f"{self.wap_cols[i]}: {query_raw_rssi[i]:.0f}dBm" for i in top_indices
        )

    def retrieve_floor_locked_neighbors(
        self,
        query_raw_rssi: np.ndarray,
        locked_bld: int,
        locked_flr: int,
    ) -> Tuple[List[NeighborRecord], Tuple[int, float, float, float]]:
        """Retrieve Top-K neighbors strictly within (locked_bld, locked_flr) and compute spatial prior."""
        assert self.memory_pos_rssi is not None and self.memory_latent is not None and self.memory_meta is not None

        q_pos = self.preprocess_positive_shift(query_raw_rssi.reshape(1, -1))[0]
        q_std = self.preprocess_autoloc_standard(query_raw_rssi.reshape(1, -1))
        q_latent = self.pca.transform(self.scaler.transform(q_std))[0]

        # Candidate indices on the locked building & floor
        cand_indices = self.floor_groups.get((locked_bld, locked_flr))
        if cand_indices is None or len(cand_indices) == 0:
            cand_indices = np.where(self.memory_meta["BUILDINGID"].to_numpy(dtype=int) == locked_bld)[0]
        if len(cand_indices) == 0:
            cand_indices = np.arange(len(self.memory_meta))

        # Compute Manhattan (L1) distance in positive signal space + Latent Euclidean distance
        cand_pos = self.memory_pos_rssi[cand_indices]
        l1_dists = np.sum(np.abs(cand_pos - q_pos), axis=1)

        cand_latent = self.memory_latent[cand_indices]
        pca_dists = np.linalg.norm(cand_latent - q_latent, axis=1)

        # Standardized composite distance fusion (Dual-Space Metric)
        l1_z = l1_dists / (np.std(l1_dists) + 1e-5)
        pca_z = pca_dists / (np.std(pca_dists) + 1e-5)
        composite_dists = l1_z + 0.30 * pca_z

        order = np.argsort(composite_dists)
        k_prompt = min(self.n_neighbors, len(order))
        k_prior = min(self.knn_k, len(order))

        top_prior_idx = cand_indices[order[:k_prior]]
        top_prior_l1 = l1_dists[order[:k_prior]]

        # Non-linear inverse distance kernel + Spatial Outlier Pruning
        weights = 1.0 / ((top_prior_l1 + 1.0) ** self.idw_exponent)
        prior_lats = self.memory_meta.iloc[top_prior_idx]["LATITUDE"].to_numpy(dtype=float)
        prior_lons = self.memory_meta.iloc[top_prior_idx]["LONGITUDE"].to_numpy(dtype=float)

        med_lat = float(np.median(prior_lats[: min(3, len(prior_lats))]))
        med_lon = float(np.median(prior_lons[: min(3, len(prior_lons))]))
        spatial_dev = np.sqrt((prior_lats - med_lat) ** 2 + (prior_lons - med_lon) ** 2)
        weights = weights * np.exp(-spatial_dev / self.spatial_sigma)
        weights /= np.sum(weights)

        prior_lat = float(np.sum(prior_lats * weights))
        prior_lon = float(np.sum(prior_lons * weights))
        dispersion = float(np.mean(spatial_dev))

        # SpaceID from nearest dual-space neighbor
        prior_space = int(self.memory_meta.iloc[top_prior_idx[0]]["SPACEID"])

        neighbors: List[NeighborRecord] = []
        for rank_i in order[:k_prompt]:
            global_idx = int(cand_indices[rank_i])
            row = self.memory_meta.iloc[global_idx]
            neighbors.append(
                NeighborRecord(
                    index=global_idx,
                    distance=float(pca_dists[rank_i]),
                    building=int(row["BUILDINGID"]),
                    floor=int(row["FLOOR"]),
                    space_id=int(row["SPACEID"]),
                    latitude=float(row["LATITUDE"]),
                    longitude=float(row["LONGITUDE"]),
                )
            )

        return neighbors, (prior_space, prior_lat, prior_lon, dispersion)

    def format_hybrid_prompt(
        self,
        locked_bld: int,
        locked_flr: int,
        top_waps_str: str,
        neighbors: List[NeighborRecord],
        prior: Tuple[int, float, float, float],
    ) -> str:
        lines = []
        for i, n in enumerate(neighbors, 1):
            lines.append(
                f"N{i}: dist={n.distance:.2f}, Bld={n.building}, Flr={n.floor}, "
                f"Spc={n.space_id}, Lat={n.latitude:.2f}, Lon={n.longitude:.2f}"
            )
        prior_space, prior_lat, prior_lon, _ = prior
        return self.HYBRID_PROMPT_TEMPLATE.format(
            locked_bld=locked_bld,
            locked_flr=locked_flr,
            query_waps=top_waps_str,
            neighbors_block="\n".join(lines),
            prior_space=prior_space,
            prior_lat=prior_lat,
            prior_lon=prior_lon,
        )

    def call_ollama(self, prompt: str, timeout: float = 30.0) -> str:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "num_predict": 65,
                "temperature": 0.0,
            },
        }
        resp = requests.post(self.ollama_url, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json().get("response", "")

    def parse_constrained_response(
        self,
        raw: str,
        locked_bld: int,
        locked_flr: int,
        neighbors: List[NeighborRecord],
        prior: Tuple[int, float, float, float],
    ) -> AutoLoCPrediction:
        """Parse LLM JSON while enforcing Floor-Locked constraints and convex hull bounding."""
        prior_space, prior_lat, prior_lon, _ = prior
        clean = raw.strip()
        match = re.search(r"\{.*?\}", clean, re.DOTALL)
        if match:
            clean = match.group(0)

        try:
            data = json.loads(clean)
            spc = int(data.get("predicted_space", prior_space))
            lat = float(data["predicted_lat"])
            lon = float(data["predicted_lon"])

            if not np.isfinite(lat) or not np.isfinite(lon):
                raise ValueError("Non-finite coordinates")

            lats = [n.latitude for n in neighbors] + [prior_lat]
            lons = [n.longitude for n in neighbors] + [prior_lon]
            min_lat, max_lat = min(lats) - 10.0, max(lats) + 10.0
            min_lon, max_lon = min(lons) - 10.0, max(lons) + 10.0

            if not (min_lat <= lat <= max_lat and min_lon <= lon <= max_lon):
                lat, lon = prior_lat, prior_lon
            else:
                # Bayesian synthesis with Floor-Locked Dual-Space Prior
                lat = 0.25 * lat + 0.75 * prior_lat
                lon = 0.25 * lon + 0.75 * prior_lon

            valid_spaces = {n.space_id for n in neighbors}
            if spc not in valid_spaces:
                spc = prior_space

            return AutoLoCPrediction(
                predicted_building=locked_bld,
                predicted_floor=locked_flr,
                predicted_space=spc,
                predicted_lat=lat,
                predicted_lon=lon,
                raw_response=raw,
                fallback_used=False,
            )
        except Exception:
            return AutoLoCPrediction(
                predicted_building=locked_bld,
                predicted_floor=locked_flr,
                predicted_space=prior_space,
                predicted_lat=prior_lat,
                predicted_lon=prior_lon,
                raw_response=raw,
                fallback_used=True,
            )

    def predict_batch(
        self,
        X_queries_raw: np.ndarray,
        max_workers: int = 4,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> List[AutoLoCPrediction]:
        """Run Cascaded RF locking followed by Confidence-Gated Floor-Locked LLM reasoning."""
        if not self.is_fitted or self.clf_building is None or self.clf_floor is None:
            raise RuntimeError("HybridRFAutoLoCEngine must be fitted before predict_batch.")

        # Step 1: Vectorized Cascaded RF prediction for all queries
        X_pos = self.preprocess_positive_shift(X_queries_raw)
        X_pow = X_pos ** self.power_gamma
        pred_blds = self.clf_building.predict(X_pos).astype(int)
        X_flr_feat = np.column_stack((X_pow, pred_blds))
        pred_flrs = self.clf_floor.predict(X_flr_feat).astype(int)

        n = len(X_queries_raw)
        results: List[Optional[AutoLoCPrediction]] = [None] * n

        def _worker(i: int) -> Tuple[int, AutoLoCPrediction]:
            t0 = time.time()
            q_raw = X_queries_raw[i]
            b_hat = int(pred_blds[i])
            f_hat = int(pred_flrs[i])

            neighbors, prior = self.retrieve_floor_locked_neighbors(q_raw, b_hat, f_hat)
            prior_space, prior_lat, prior_lon, dispersion = prior

            # Confidence-Gated LLM Trigger: invoke Ollama LLM when spatial dispersion > threshold
            if dispersion > self.llm_gate_threshold:
                top_waps = self.get_query_strongest_waps(q_raw, top_k=3)
                prompt = self.format_hybrid_prompt(b_hat, f_hat, top_waps, neighbors, prior)
                try:
                    raw_resp = self.call_ollama(prompt)
                    pred = self.parse_constrained_response(raw_resp, b_hat, f_hat, neighbors, prior)
                except Exception as e:
                    pred = AutoLoCPrediction(
                        predicted_building=b_hat,
                        predicted_floor=f_hat,
                        predicted_space=prior_space,
                        predicted_lat=prior_lat,
                        predicted_lon=prior_lon,
                        raw_response=f"ERROR: {e}",
                        fallback_used=True,
                    )
            else:
                pred = AutoLoCPrediction(
                    predicted_building=b_hat,
                    predicted_floor=f_hat,
                    predicted_space=prior_space,
                    predicted_lat=prior_lat,
                    predicted_lon=prior_lon,
                    raw_response="FAST_FLOOR_LOCKED_CONVERGENCE",
                    fallback_used=False,
                )

            pred.latency_sec = time.time() - t0
            return i, pred

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
