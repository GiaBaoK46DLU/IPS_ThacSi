"""KAN PyTorch representation of shared NumPy fingerprint splits."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from ...data import feature_transform
from ...data_access import FingerprintSplit


class FingerprintDataset(Dataset):
    def __init__(self, data, indices, preprocessing, classes):
        self.features = torch.from_numpy(feature_transform(data.rssi[indices]))
        self.indices = torch.as_tensor(indices.copy(), dtype=torch.int64)
        self.xy_original = data.metadata[indices, :2].copy()  # keep float64 for metre metrics
        coordinates = preprocessing["coordinates"]
        center = np.asarray(coordinates["center"], dtype=np.float64)
        scale = coordinates["shared_scale"]
        self.xy = torch.from_numpy(((self.xy_original - center) / scale).astype(np.float32))
        mapping = {(c["building"], c["floor"]): c["class_id"] for c in classes}
        pairs = data.pairs
        self.classes = torch.as_tensor([mapping[pairs[int(i)]] for i in indices], dtype=torch.int64)
        self.metadata = {name: data.column(name)[indices].copy() for name in ["USERID", "PHONEID"]}

    @classmethod
    def from_split(cls, split: FingerprintSplit):
        instance = cls.__new__(cls)
        instance.features = torch.from_numpy(feature_transform(split.inputs.rssi))
        instance.indices = torch.as_tensor(split.inputs.row_indices.copy(), dtype=torch.int64)
        instance.xy_original = split.xy.copy()
        coordinates = split.info.preprocessing["coordinates"]
        center = np.asarray(coordinates["center"], dtype=np.float64)
        instance.xy = torch.from_numpy(((instance.xy_original - center) /
                                       coordinates["shared_scale"]).astype(np.float32))
        instance.classes = torch.as_tensor(split.class_ids.copy(), dtype=torch.int64)
        instance.metadata = {name: split.metadata[name].copy() for name in ["USERID", "PHONEID"]}
        return instance

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        return self.features[item], self.classes[item], self.xy[item]
