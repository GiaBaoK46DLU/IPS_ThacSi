"""Quy ước mẫu NumPy; input dự đoán không chứa nhãn vị trí hay metadata."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..data import AP_COLUMNS


def readonly(array, dtype=None):
    result = np.array(array, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class FingerprintInputs:
    rssi: np.ndarray
    row_indices: np.ndarray
    source_id: str
    ap_order: tuple[str, ...] = tuple(AP_COLUMNS)

    def __post_init__(self):
        rssi, indices = np.asarray(self.rssi), np.asarray(self.row_indices)
        if self.ap_order != tuple(AP_COLUMNS):
            raise ValueError("Inputs must retain WAP001..WAP520 order.")
        if rssi.ndim != 2 or rssi.shape[1] != 520 or len(rssi) == 0:
            raise ValueError("Expected nonempty RSSI[N,520].")
        if (not np.isfinite(rssi).all() or not (rssi == np.floor(rssi)).all() or
                not ((rssi == 100) | ((rssi >= -104) & (rssi <= 0))).all()):
            raise ValueError("Invalid RSSI domain or missing-value code.")
        if (indices.shape != (len(rssi),) or not np.issubdtype(indices.dtype, np.integer) or
                (indices < 0).any() or len(np.unique(indices)) != len(indices)):
            raise ValueError("Row indices must be distinct nonnegative integers aligned with RSSI.")
        if not isinstance(self.source_id, str) or not self.source_id:
            raise ValueError("source_id is required: row indices alone do not identify a CSV.")
        object.__setattr__(self, "rssi", readonly(rssi, np.int16))
        object.__setattr__(self, "row_indices", readonly(indices, np.int64))

    def __len__(self):
        return len(self.row_indices)


@dataclass(frozen=True)
class DatasetInfo:
    directory: Path
    manifest: dict
    preprocessing: dict
    classes: list[dict]
    ap_order: list[str]


@dataclass(frozen=True)
class FingerprintSplit:
    name: str
    inputs: FingerprintInputs
    class_ids: np.ndarray
    xy: np.ndarray
    metadata: dict[str, np.ndarray]
    info: DatasetInfo

    def __post_init__(self):
        count = len(self.inputs)
        labels, xy = np.asarray(self.class_ids), np.asarray(self.xy)
        if (self.name not in ("train", "validation", "test") or labels.shape != (count,) or
                not np.issubdtype(labels.dtype, np.integer) or (labels < 0).any() or
                (labels >= len(self.info.classes)).any() or xy.shape != (count, 2) or
                not np.isfinite(xy).all()):
            raise ValueError("Invalid split, labels or original-coordinate targets.")
        if any(np.asarray(values).shape != (count,) for values in self.metadata.values()):
            raise ValueError("Metadata must retain sample alignment.")
        object.__setattr__(self, "class_ids", readonly(labels, np.int64))
        object.__setattr__(self, "xy", readonly(xy, np.float64))
        object.__setattr__(self, "metadata", {k: readonly(v) for k, v in self.metadata.items()})

    def __len__(self):
        return len(self.inputs)
