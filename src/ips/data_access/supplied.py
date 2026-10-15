"""Read supplied DLULIB, HCXY, TUJI1 and SYL without changing legacy UJI.

Source row numbers and AP order are retained. This module does not fit a
transform or expose query labels to an algorithm's forward pass.
"""
from __future__ import annotations

import csv
import posixpath
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from ..data import ROOT, sha256


@dataclass
class SuppliedFingerprints:
    name: str
    rssi: np.ndarray
    xy: np.ndarray
    floors: np.ndarray
    buildings: np.ndarray
    source_files: np.ndarray
    source_rows: np.ndarray
    rp: np.ndarray
    ap_order: list[str]
    missing_code: int
    sources: list[dict]
    assumptions: list[str]
    metadata: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self):
        n = len(self.rssi)
        if self.rssi.shape != (n, len(self.ap_order)) or self.xy.shape != (n, 2):
            raise ValueError("Misaligned RSSI/coordinate arrays.")
        if any(len(v) != n for v in (self.floors, self.buildings, self.source_files, self.source_rows, self.rp)):
            raise ValueError("Metadata lost source row alignment.")
        if any(len(v) != n for v in self.metadata.values()):
            raise ValueError("Extra metadata lost source row alignment.")
        if not np.isfinite(self.xy).all() or not np.isfinite(self.rssi).all():
            raise ValueError("Non-finite source observation.")
        detected = self.rssi != self.missing_code
        if not ((self.rssi[detected] >= -104) & (self.rssi[detected] <= 0)).all():
            raise ValueError("Detected RSSI is outside the frozen UJI transform range.")
        if len(set(zip(self.source_files.tolist(), self.source_rows.tolist()))) != n:
            raise ValueError("Source identities must be unique.")

    def __len__(self):
        return len(self.rssi)


def worksheet_rows(path):
    """Read actual cells, preserving spreadsheet gaps and original row numbers."""
    ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    with ZipFile(path) as archive:
        shared = []
        if "xl/sharedStrings.xml" in archive.namelist():
            doc = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            shared = ["".join(s.itertext()) for s in doc.findall(f"{{{ns}}}si")]
        targets = {r.attrib["Id"]: r.attrib["Target"] for r in ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))}
        sheets = ET.fromstring(archive.read("xl/workbook.xml")).findall(f"{{{ns}}}sheets/{{{ns}}}sheet")
        if len(sheets) != 1:
            raise ValueError("Expected one data worksheet.")
        target = targets[sheets[0].attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]]
        target = target.lstrip("/") if target.startswith("/") else posixpath.normpath("xl/" + target)
        with archive.open(target) as stream:
            for _, row in ET.iterparse(stream, events=("end",)):
                if row.tag != f"{{{ns}}}row":
                    continue
                cells = {}
                for cell in row:
                    column = 0
                    for char in cell.attrib["r"]:
                        if not char.isalpha():
                            break
                        column = column * 26 + ord(char.upper()) - 64
                    value = cell.findtext(f"{{{ns}}}v")
                    if cell.find(f"{{{ns}}}f") is not None:
                        raise ValueError("DLULIB reader does not accept formula observations.")
                    if cell.attrib.get("t") == "s" and value is not None:
                        value = shared[int(value)]
                    elif cell.attrib.get("t") == "inlineStr":
                        value = "".join(cell.find(f"{{{ns}}}is").itertext())
                    cells[column - 1] = value
                values = [cells.get(i) for i in range(max(cells, default=-1) + 1)]
                number = int(row.attrib["r"])
                row.clear()
                if any(v is not None and str(v).strip() for v in values):
                    yield number, values


def read_supplied(name, role, raw_root=None):
    if name not in ("dlulib", "hcxy", "tuji", "syl") or role not in ("train", "test"):
        raise ValueError("Implemented supplied datasets: dlulib, hcxy, tuji, syl; roles: train/test.")
    raw_root = Path(raw_root) if raw_root else ROOT / "dataset/raw"
    chunks, sources, metadata = [], [], {}
    if name == "dlulib":
        paths = [raw_root / f"dlulib/floor_{floor}" / (
            f"Training_DLULIB_Floor{floor}_All_20.xlsx" if role == "train" else
            f"Testing_DLULIB_Floor{floor}.xlsx") for floor in (1, 2, 3)]
        ap_order = [f"AP{i}" for i in range(1, 18)]
        for floor, path in zip((1, 2, 3), paths):
            rows = iter(worksheet_rows(path)); _, header = next(rows)
            if header[:17] != ap_order:
                raise ValueError("DLULIB AP order changed.")
            index = {v: i for i, v in enumerate(header)}
            for row_number, row in rows:
                if len(row) != len(header) or any(v is None for v in row):
                    raise ValueError(f"Incomplete DLULIB observation: {path}:{row_number}")
                observed_floor = int(float(row[index["FloorID"]]))
                if observed_floor != floor:
                    raise ValueError("Folder and FloorID disagree.")
                chunks.append((row[:17], [row[index["X"]], row[index["Y"]]], floor, 0,
                               path.relative_to(ROOT).as_posix(), row_number, row[index["ID_RP"]]))
            sources.append({"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path)})
        missing_code = -100
        assumptions = ["-100 is treated as undetected RSSI; not confirmed by the owner",
                       "X/Y are provisionally interpreted as metres; owner confirmation pending",
                       "AP1..AP17 have the same physical order across floors; owner confirmation pending",
                       "building=0 is a synthetic namespace for one site, not an observed label"]
    elif name == "hcxy":
        base = raw_root / "sodindoorloc/hcxy"
        path = base / f"{role}_HCXY.txt"; map_path = base / "mapxy_HCXY.txt"
        mapping = {}
        for line in map_path.read_text(encoding="utf-8-sig").splitlines():
            rp, x, y = map(float, line.split())
            if rp in mapping:
                raise ValueError("Repeated map RP ID.")
            mapping[rp] = (x, y)
        ap_order = [f"MAC{i}" for i in range(1, 348)]
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream, delimiter="\t")
            if next(reader) != ["RP", *ap_order]:
                raise ValueError("HCXY header/AP order changed.")
            for number, row in enumerate(reader, 2):
                if len(row) != 348 or float(row[0]) not in mapping:
                    raise ValueError(f"Invalid HCXY row/map: {number}")
                # The native files were numerically matched in the dataset audit.
                chunks.append((row[1:], mapping[float(row[0])], 4, 2,
                               path.relative_to(ROOT).as_posix(), number, row[0]))
        sources = [{"path": p.relative_to(ROOT).as_posix(), "sha256": sha256(p)} for p in (path, map_path)]
        missing_code = 100; assumptions = []
    else:
        # Headers in supplied CSVs were compared numerically to author files.
        # TUJI's six label names differ from their native meaning. In particular
        # RELATIVEPOSITION is device ID, never a room or a model input.
        if name == "tuji":
            path = raw_root / "tuji" / ("TUJI_training.csv" if role == "train" else "TUJI_testing.csv")
            ap_order = [f"WAP{i:03d}" for i in range(1, 311)]
            tail = ["LONGITUDE", "LATITUDE", "FLOOR", "BUILDINGID", "SPACEID", "RELATIVEPOSITION"]
            floor_column, building_column = "BUILDINGID", "SPACEID"
            extras = {"z": "FLOOR", "device_id": "RELATIVEPOSITION"}
        else:
            path = raw_root / "sodindoorloc/syl" / ("SYL_Train.csv" if role == "train" else "SYL_Test.csv")
            ap_order = [f"MAC{i}" for i in range(1, 364)]
            tail = ["LONGITUDE", "LATITUDE", "FloorID", "BuildingID", "SceneID", "UserID", "PhoneID", "SampleTimes"]
            floor_column, building_column = "FloorID", "BuildingID"
            extras = {"scene_id": "SceneID", "user_id": "UserID", "device_id": "PhoneID", "sample_times": "SampleTimes"}
        extra_rows = {key: [] for key in extras}
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream)
            if next(reader) != ap_order + tail:
                raise ValueError(f"{name} CSV header/AP order changed.")
            for number, row in enumerate(reader, 2):
                if len(row) != len(ap_order) + len(tail):
                    raise ValueError(f"Incomplete {name} observation: {number}")
                labels = dict(zip(tail, map(float, row[len(ap_order):])))
                floor, building = labels[floor_column], labels[building_column]
                if floor != int(floor) or building != int(building):
                    raise ValueError("Nonintegral building/floor label.")
                chunks.append((row[:len(ap_order)], [labels["LONGITUDE"], labels["LATITUDE"]],
                    int(floor), int(building), path.relative_to(ROOT).as_posix(), number, 0))
                for key, column in extras.items():
                    extra_rows[key].append(labels[column])
        # No native RP ID in these CSVs. Derived position IDs are scoped to each
        # role and are export metadata only; split keys use original coordinates.
        positions = sorted({(c[3], c[2], *c[1]) for c in chunks})
        position_ids = {key: i for i, key in enumerate(positions)}
        chunks = [(*c[:6], position_ids[(c[3], c[2], *c[1])]) for c in chunks]
        metadata = {key: np.asarray(values) for key, values in extra_rows.items()}
        sources = [{"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path)}]
        missing_code = 100
        assumptions = ["RP IDs are derived from exact building/floor/X/Y within each source role, not native RP labels"]
    return SuppliedFingerprints(name, np.asarray([c[0] for c in chunks], dtype=np.float32),
        np.asarray([c[1] for c in chunks], dtype=np.float64), np.asarray([c[2] for c in chunks]),
        np.asarray([c[3] for c in chunks]), np.asarray([c[4] for c in chunks]),
        np.asarray([c[5] for c in chunks]), np.asarray([float(c[6]) for c in chunks]),
        ap_order, missing_code, sources, assumptions, metadata)


def group_split(data, validation_fraction=0.2, seed=2026):
    """Hold out entire measured positions, with building/floor coverage."""
    if not 0 < validation_fraction < 1:
        raise ValueError("Validation fraction must be between 0 and 1.")
    keys = list(zip(data.buildings.tolist(), data.floors.tolist(),
                    data.xy[:, 0].tolist(), data.xy[:, 1].tolist()))
    groups = sorted(set(keys)); rng = np.random.default_rng(seed); heldout = set()
    for pair in sorted(set(k[:2] for k in groups)):
        candidates = [k for k in groups if k[:2] == pair]
        if len(candidates) < 2:
            raise ValueError("Need at least two positions in each class.")
        count = min(len(candidates) - 1, max(1, round(len(candidates) * validation_fraction)))
        heldout.update(candidates[i] for i in rng.permutation(len(candidates))[:count])
    mask = np.asarray([key in heldout for key in keys])
    return np.flatnonzero(~mask), np.flatnonzero(mask)
