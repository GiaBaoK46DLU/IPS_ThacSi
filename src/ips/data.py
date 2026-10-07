"""Schema, kiểm tra dữ liệu và hàm dùng chung cho chặng 5."""
from __future__ import annotations

import csv
import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "ujiindoorloc"
OUTPUT = ROOT / "data" / "processed" / "uji_group_split_v1"
AP_COLUMNS = [f"WAP{i:03d}" for i in range(1, 521)]
META_COLUMNS = ["LONGITUDE", "LATITUDE", "FLOOR", "BUILDINGID", "SPACEID",
                "RELATIVEPOSITION", "USERID", "PHONEID", "TIMESTAMP"]
COLUMNS = AP_COLUMNS + META_COLUMNS
GROUP_COLUMNS = ["USERID", "PHONEID", "BUILDINGID", "FLOOR", "LONGITUDE",
                 "LATITUDE", "SPACEID", "RELATIVEPOSITION"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def csv_bytes(header, rows) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def write_bundle(directory: Path, files: dict[str, bytes]) -> None:
    """Chạy lại được nếu nội dung giống nhau; không ghi đè nội dung khác."""
    for name, content in files.items():
        destination = directory / name
        if destination.exists() and destination.read_bytes() != content:
            raise ValueError(f"Tệp đã tồn tại với nội dung khác: {destination}. Dùng thư mục phiên bản mới.")
    directory.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        destination = directory / name
        if not destination.exists():
            destination.write_bytes(content)


@dataclass
class Data:
    path: Path
    rssi: np.ndarray
    metadata: np.ndarray
    digest: str

    def column(self, name: str) -> np.ndarray:
        return self.metadata[:, META_COLUMNS.index(name)]

    @property
    def size(self) -> int:
        return len(self.rssi)

    @property
    def pairs(self) -> list[tuple[int, int]]:
        return list(zip(self.column("BUILDINGID").astype(int).tolist(),
                        self.column("FLOOR").astype(int).tolist()))


def read_data(path: Path) -> Data:
    """Đọc CSV gốc; lỗi schema/NaN/RSSI lạ phải được xử lý trước khi split."""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader, [])
        if header != COLUMNS:
            raise ValueError(f"{path.name}: header phải đúng 520 WAP + 9 cột, đúng thứ tự/hoa thường.")
        rows = []
        for line, row in enumerate(reader, start=2):
            if len(row) != len(COLUMNS) or any(not cell.strip() for cell in row):
                raise ValueError(f"{path.name}, dòng CSV {line}: thiếu ô hoặc sai số cột.")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path.name}: CSV rỗng.")
    try:
        values = np.asarray(rows, dtype=np.float64)
    except ValueError as error:
        raise ValueError(f"{path.name}: ô không phải số.") from error
    del rows
    if not np.isfinite(values).all():
        raise ValueError(f"{path.name}: NaN hoặc vô cực; không phải mã 100.")
    raw_rssi = values[:, :520]
    valid_rssi = (raw_rssi == 100) | ((raw_rssi >= -104) & (raw_rssi <= 0))
    if not (valid_rssi & (raw_rssi == np.floor(raw_rssi))).all():
        bad = np.argwhere(~(valid_rssi & (raw_rssi == np.floor(raw_rssi))))[0]
        raise ValueError(f"{path.name}, dòng {int(bad[0])+2}, {AP_COLUMNS[int(bad[1])]}: RSSI sai miền/mã.")
    metadata = values[:, 520:].copy()
    integer_metadata = metadata[:, 2:]
    if not (integer_metadata == np.floor(integer_metadata)).all():
        raise ValueError(f"{path.name}: nhãn/ID/timestamp phải là số nguyên.")
    data = Data(path.resolve(), raw_rssi.astype(np.int16), metadata, sha256(path))
    for name, allowed in [("BUILDINGID", [0, 1, 2]), ("FLOOR", [0, 1, 2, 3, 4]),
                          ("RELATIVEPOSITION", [0, 1, 2])]:
        # UCI giải thích 1/2; CSV test chính thức thực tế dùng 0 cho cột này.
        # Chấp nhận 0 như metadata chưa xác định, không gán thành "inside".
        if not np.isin(data.column(name), allowed).all():
            raise ValueError(f"{path.name}: {name} ngoài miền tài liệu.")
    for name in ["SPACEID", "USERID", "PHONEID", "TIMESTAMP"]:
        if (data.column(name) < 0).any():
            raise ValueError(f"{path.name}: {name} âm.")
    return data


def encode_groups(data: Data, columns=GROUP_COLUMNS) -> np.ndarray:
    """Mã nhóm xác định: tuple số, sort tăng dần, đánh số từ 0."""
    keys = list(zip(*(data.column(name).tolist() for name in columns)))
    mapping = {key: index for index, key in enumerate(sorted(set(keys)))}
    return np.asarray([mapping[key] for key in keys], dtype=np.int64)


def row_keys(data: Data, mode: str, indices=None) -> set[bytes]:
    indices = np.arange(data.size) if indices is None else np.asarray(indices)
    if mode == "location":
        array = np.column_stack([data.column(name)[indices] for name in
                                 ["BUILDINGID", "FLOOR", "LONGITUDE", "LATITUDE"]])
    else:
        rssi = np.ascontiguousarray(data.rssi[indices]).view(np.uint8).reshape(len(indices), -1)
        if mode == "rssi":
            array = rssi
        else:
            names = (META_COLUMNS if mode == "full" else
                     ["BUILDINGID", "FLOOR"] if mode == "classification_rssi" else
                     ["BUILDINGID", "FLOOR", "LONGITUDE", "LATITUDE"])
            meta = np.ascontiguousarray(np.column_stack([data.column(n)[indices] for n in names]))
            meta_bytes = meta.view(np.uint8).reshape(len(indices), -1)
            array = np.column_stack([rssi, meta_bytes])
    return {row.tobytes() for row in np.ascontiguousarray(array)}


def overlaps(left: Data, right: Data, left_indices=None, right_indices=None) -> dict:
    return {mode + "_distinct_keys": len(row_keys(left, mode, left_indices) &
                                          row_keys(right, mode, right_indices))
            for mode in ["full", "rssi", "classification_rssi", "labeled_rssi", "location"]}


def feature_transform(rssi: np.ndarray) -> np.ndarray:
    """Giữ tất cả AP. Trục cuối theo thứ tự (z, mask)."""
    mask = (rssi != 100).astype(np.float32)
    z = np.where(mask == 1, (rssi.astype(np.float32) + 104) / 104, 0)
    return np.stack([z, mask], axis=-1)
