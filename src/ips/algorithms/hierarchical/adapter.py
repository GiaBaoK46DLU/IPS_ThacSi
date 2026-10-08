from __future__ import annotations
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor

from ..base import Algorithm
from ...data_access.contracts import FingerprintInputs, FingerprintSplit
from ...experiments.contracts import AlgorithmExecution, ExperimentConfig, IPSPredictions

class HierarchicalAlgorithm(Algorithm):
    """
    Hierarchical IPS Algorithm based on UJIndoorLoc/hierarchical_ips.py
    Uses RandomForest for Building and Floor classification, and KNN for Coordinates regression.
    """
    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        self.clf_building = None
        self.clf_floor = None
        self.reg_coords = None
        
        # Hyperparameters can be passed via config.params
        self.n_estimators = self.config.params.get("n_estimators", 50)
        self.n_neighbors = self.config.params.get("n_neighbors", 3)
        self.penalty_weight = self.config.params.get("penalty_weight", 1000.0)
        self.seed = self.config.seed
        
        self._train = None
        self._classes = None

    def _preprocess_rssi(self, rssi: np.ndarray) -> np.ndarray:
        X = np.array(rssi, dtype=np.float32)
        X = np.where(X == 100, -110, X)
        X = X + 110
        X = np.clip(X, a_min=0, a_max=None)
        return X

    def prepare(self, train: FingerprintSplit) -> None:
        if train.name != "train":
            raise ValueError("Hierarchical preparation requires the train split.")
        self._train = train
        self._classes = train.info.classes

    def execute(self, validation: FingerprintSplit | None) -> AlgorithmExecution:
        if self._train is None:
            raise RuntimeError("Call prepare(train) before execute().")

        # 1. Prepare training data
        X_train = self._preprocess_rssi(self._train.inputs.rssi)
        
        # We need building and floor for training. We can extract them from the shared classes mapping.
        # self._train.class_ids gives the index into self._classes
        buildings = np.array([self._classes[cid]["building"] for cid in self._train.class_ids])
        floors = np.array([self._classes[cid]["floor"] for cid in self._train.class_ids])
        y_train_coords = self._train.xy

        # --- BƯỚC 1: Mô hình dự đoán Tòa nhà ---
        self.clf_building = RandomForestClassifier(
            n_estimators=self.n_estimators, random_state=self.seed, n_jobs=-1
        )
        self.clf_building.fit(X_train, buildings)

        # --- BƯỚC 2: Mô hình dự đoán Tầng ---
        X_train_floor = np.column_stack((X_train, buildings))
        self.clf_floor = RandomForestClassifier(
            n_estimators=self.n_estimators, random_state=self.seed, n_jobs=-1
        )
        self.clf_floor.fit(X_train_floor, floors)

        # --- BƯỚC 3: Mô hình dự đoán Tọa độ ---
        X_train_coords = np.column_stack((
            X_train, 
            buildings * self.penalty_weight, 
            floors * self.penalty_weight
        ))
        self.reg_coords = KNeighborsRegressor(
            n_neighbors=self.n_neighbors, weights='distance', metric='euclidean'
        )
        self.reg_coords.fit(X_train_coords, y_train_coords)

        import joblib
        output = self.config.output_dir / "algorithm"
        output.mkdir(parents=True, exist_ok=True)
        
        building_path = output / "model_building.pkl"
        floor_path = output / "model_floor.pkl"
        coords_path = output / "model_coordinates.pkl"
        
        joblib.dump(self.clf_building, building_path)
        joblib.dump(self.clf_floor, floor_path)
        joblib.dump(self.reg_coords, coords_path)
        
        artifacts = {
            "model_building": building_path,
            "model_floor": floor_path,
            "model_coordinates": coords_path
        }

        # Trả về cờ finalized=True vì scikit-learn training là synchronous và hoàn tất ngay lập tức
        return AlgorithmExecution(finalized=True, run_type="scikit-learn-training", history=None, artifacts=artifacts, resources=None)

    def predict(self, inputs: FingerprintInputs) -> IPSPredictions:
        if self.clf_building is None:
            raise RuntimeError("Call execute() before predict().")

        X_test = self._preprocess_rssi(inputs.rssi)

        # Bước 1
        pred_building = self.clf_building.predict(X_test)
        
        # Bước 2
        X_test_floor = np.column_stack((X_test, pred_building))
        pred_floor = self.clf_floor.predict(X_test_floor)
        
        # Bước 3
        X_test_coords = np.column_stack((
            X_test, 
            pred_building * self.penalty_weight, 
            pred_floor * self.penalty_weight
        ))
        pred_coords = self.reg_coords.predict(X_test_coords)

        # Convert pred_building and pred_floor to combined class_ids
        # Since classes is a list of dicts like {"building": 1, "floor": 2}
        mapping = {(c["building"], c["floor"]): i for i, c in enumerate(self._classes)}
        class_ids = []
        for b, f in zip(pred_building, pred_floor):
            # In rare cases the predicted (b, f) might not exist in mapping, fallback to nearest or 0
            # but RandomForest classifier outputs classes it has seen.
            cid = mapping.get((int(b), int(f)), 0)
            class_ids.append(cid)
        class_ids = np.array(class_ids, dtype=np.int64)

        return IPSPredictions(
            row_indices=inputs.row_indices.copy(),
            source_id=inputs.source_id,
            xy=pred_coords,
            class_ids=class_ids,
            buildings=pred_building.astype(np.int64),
            floors=pred_floor.astype(np.int64)
        )

