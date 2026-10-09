"""Áp gói split cố định, không fit lại bằng validation/test."""
from __future__ import annotations

import csv
import copy
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..data import AP_COLUMNS, COLUMNS, META_COLUMNS, ROOT, Data, encode_groups, read_data, sha256
from .contracts import DatasetInfo, FingerprintInputs, FingerprintSplit


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def index_file(path: Path) -> np.ndarray:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    indices = np.asarray([int(row["row_index"]) for row in rows], dtype=np.int64)
    if not np.array_equal(indices, np.unique(indices)):
        raise ValueError("Indices must be sorted and unique.")
    return indices


@dataclass
class Bundle:
    directory: Path
    manifest: dict
    preprocessing: dict
    classes: list[dict]
    ap_order: list[str]
    development: Data
    indices: dict[str, np.ndarray]

    @property
    def info(self) -> DatasetInfo:
        return DatasetInfo(self.directory, copy.deepcopy(self.manifest), copy.deepcopy(self.preprocessing),
                           copy.deepcopy(self.classes), self.ap_order.copy())

    def split(self, split: str, limit: int | None = None) -> FingerprintSplit:
        if split not in self.indices:
            raise ValueError(f"Unknown split: {split}")
        if split == "test":
            data = read_data(ROOT / self.manifest["sources"]["final_test"]["path"])
        else:
            data = self.development
        indices = self.indices[split]
        if limit is not None:
            if limit < 1:
                raise ValueError("Subset limit must be positive.")
            # Smoke/check: cùng subset xác định cho mọi backbone, có các lớp
            # thay vì chỉ lấy đầu CSV (thường là nhiều lần quét của một điểm).
            pairs = data.pairs
            base, remainder = divmod(min(limit, len(indices)), len(self.classes))
            selected = []
            for position, cls in enumerate(self.classes):
                pair = (cls["building"], cls["floor"])
                quota = base + int(position < remainder)
                members = [int(i) for i in indices if pairs[int(i)] == pair]
                selected.extend(members[:quota])
            indices = np.asarray(sorted(selected), dtype=np.int64)
        mapping = {(c["building"], c["floor"]): c["class_id"] for c in self.classes}
        pairs = data.pairs
        labels = np.asarray([mapping[pairs[int(i)]] for i in indices], dtype=np.int64)
        inputs = FingerprintInputs(data.rssi[indices], indices, str(data.path), tuple(self.ap_order))
        return FingerprintSplit(split, inputs, labels, data.metadata[indices, :2],
            {name: data.column(name)[indices] for name in META_COLUMNS}, self.info)

def load_bundle(directory: Path) -> Bundle:
    directory = Path(directory)
    manifest = read_json(directory / "manifest.json")
    for name, expected in manifest["artifact_sha256"].items():
        if sha256(directory / name) != expected:
            raise ValueError(f"Artifact hash mismatch: {name}")
    for source in manifest["sources"].values():
        if sha256(ROOT / source["path"]) != source["sha256"]:
            raise ValueError(f"Raw source hash mismatch: {source['path']}")
    development = read_data(ROOT / manifest["sources"]["development"]["path"])
    preprocessing = read_json(directory / "preprocessing.json")
    classes = read_json(directory / "class_mapping.json")
    ap_order = read_json(directory / "ap_order.json")
    indices = {split: index_file(directory / f"{split}_indices.csv") for split in ["train", "validation", "test"]}
    for split, values in indices.items():
        size = manifest["sources"]["final_test"]["rows"] if split == "test" else development.size
        if len(values) != manifest["counts"][split] or (values < 0).any() or (values >= size).any():
            raise ValueError(f"Invalid index coverage: {split}")
    if (set(indices["train"]) & set(indices["validation"]) or
        set(indices["train"]) | set(indices["validation"]) != set(range(development.size))):
        raise ValueError("Invalid development partition.")
    if not np.array_equal(indices["test"], np.arange(manifest["sources"]["final_test"]["rows"])):
        raise ValueError("Official test must be kept intact.")
    groups = encode_groups(development, manifest["split_settings"]["group_key_columns"])
    if set(groups[indices["train"]]) & set(groups[indices["validation"]]):
        raise ValueError("Train/validation groups overlap.")
    expected_pairs = sorted(set(development.pairs))
    if [(c["building"], c["floor"]) for c in classes] != expected_pairs:
        raise ValueError("Class mapping mismatch.")
    if [c["class_id"] for c in classes] != list(range(len(classes))) or ap_order != AP_COLUMNS:
        raise ValueError("AP order or class IDs mismatch.")
    coord = preprocessing["coordinates"]
    xy = development.metadata[indices["train"], :2]
    scale = max(float(np.sqrt(xy.var(axis=0).mean())), 1e-6)
    if (coord["fit_indices_sha256"] != sha256(directory / "train_indices.csv") or
        not np.allclose(coord["center"], xy.mean(axis=0), rtol=0, atol=1e-12) or
        abs(coord["shared_scale"] - scale) > 1e-12):
        raise ValueError("Coordinate transform was not fitted on internal train.")
    return Bundle(directory, manifest, preprocessing, classes, ap_order, development, indices)


def read_fingerprints(path: Path, ap_order: list[str]):
    if ap_order != AP_COLUMNS:
        raise ValueError("Checkpoint uses unsupported AP ordering.")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader, [])
        if header not in [AP_COLUMNS, COLUMNS]:
            raise ValueError("Input CSV must have ordered WAP001..WAP520, optionally followed by the nine original metadata columns.")
        rows = []
        for line, row in enumerate(reader, 2):
            if len(row) != len(header) or any(not value.strip() for value in row[:520]):
                raise ValueError(f"Invalid input row at CSV line {line}.")
            rows.append(row[:520])
    if not rows:
        raise ValueError("Input CSV contains no fingerprints.")
    rssi = np.asarray(rows, dtype=np.float64)
    valid = np.isfinite(rssi) & (rssi == np.floor(rssi)) & ((rssi == 100) | ((rssi >= -104) & (rssi <= 0)))
    if not valid.all():
        raise ValueError("Input contains invalid RSSI, NaN or infinity.")
    return rssi.astype(np.int16)
class DatasetRepository:
    """Load the saved protocol; never silently re-split input CSVs."""
    def load(self, directory: Path) -> Bundle:
        return load_bundle(directory)

    def read_inputs(self, path: Path) -> FingerprintInputs:
        path = Path(path).resolve()
        rssi = read_fingerprints(path, AP_COLUMNS)
        return FingerprintInputs(rssi, np.arange(len(rssi)), str(path))
