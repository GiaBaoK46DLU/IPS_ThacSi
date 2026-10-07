"""Kiểm chứng artifact đã lưu, raw hashes, nhóm, nhãn và train-only fit."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from data_utils import (AP_COLUMNS, OUTPUT, ROOT, encode_groups, feature_transform,
                        overlaps, read_data, sha256)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_indices(path):
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == ["row_index", "csv_line_number", "group_id"], "Sai header indices.")
        rows = list(reader)
    indices = np.asarray([int(row["row_index"]) for row in rows], dtype=np.int64)
    require(all(int(row["csv_line_number"]) == int(row["row_index"]) + 2 for row in rows), "Sai dòng CSV.")
    return indices, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    manifest = json.loads((args.split_dir / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["artifact_sha256"].items():
        require(sha256(args.split_dir / name) == expected, f"Artifact bị thay đổi: {name}")
    for source in manifest["sources"].values():
        require(sha256(ROOT / source["path"]) == source["sha256"], "Raw CSV đã thay đổi; không dùng indices cũ.")
    development = read_data(ROOT / manifest["sources"]["development"]["path"])
    final_test = read_data(ROOT / manifest["sources"]["final_test"]["path"])
    groups = encode_groups(development, manifest["split_settings"]["group_key_columns"])
    indices = {}
    for split in ["train", "validation", "test"]:
        values, rows = read_indices(args.split_dir / f"{split}_indices.csv")
        size = final_test.size if split == "test" else development.size
        require(len(values) == manifest["counts"][split], "Sai số dòng split.")
        require(len(set(values)) == len(values), "Chỉ số dòng lặp trong split.")
        require(((values >= 0) & (values < size)).all(), "Chỉ số dòng ngoài miền.")
        require(np.array_equal(values, np.sort(values)), "Thứ tự indices đã đổi.")
        if split != "test":
            require(all(int(row["group_id"]) == int(groups[i]) for i, row in zip(values, rows)), "Sai group ID.")
        else:
            require(all(row["group_id"] == "" for row in rows), "Test không có internal group ID.")
        indices[split] = values
    require(not set(indices["train"]) & set(indices["validation"]), "Trùng chỉ số train/val.")
    require(set(indices["train"]) | set(indices["validation"]) == set(range(development.size)), "Thiếu dòng development.")
    require(np.array_equal(indices["test"], np.arange(final_test.size)), "Test không còn nguyên file chính thức.")
    require(not set(groups[indices["train"]]) & set(groups[indices["validation"]]), "Trùng nhóm train/val.")
    classes = json.loads((args.split_dir / "class_mapping.json").read_text(encoding="utf-8"))
    pairs = [(item["building"], item["floor"]) for item in classes]
    require([item["class_id"] for item in classes] == list(range(len(classes))), "Class IDs sai thứ tự.")
    development_pairs = development.pairs
    require(pairs == sorted(set(development_pairs)), "Sai class mapping development.")
    for split in ["train", "validation"]:
        require({development_pairs[int(i)] for i in indices[split]} == set(pairs), "Thiếu lớp trong split.")
    require(set(final_test.pairs).issubset(pairs), "Test chứa lớp lạ.")
    overlap = overlaps(development, development, indices["train"], indices["validation"])
    require(overlap == manifest["internal_train_validation_overlap"], "Thống kê giao nhau sai.")
    require(overlap["full_distinct_keys"] == overlap["labeled_rssi_distinct_keys"] == 0, "Trùng bản ghi có nhãn.")
    preprocessing = json.loads((args.split_dir / "preprocessing.json").read_text(encoding="utf-8"))
    coord = preprocessing["coordinates"]
    xy_train = development.metadata[indices["train"], :2]
    expected_mean = xy_train.mean(axis=0)
    expected_scale = max(float(np.sqrt(xy_train.var(axis=0).mean())), 1e-6)
    require(np.allclose(coord["center"], expected_mean, rtol=0, atol=1e-12), "Center không fit đúng train.")
    require(abs(coord["shared_scale"] - expected_scale) < 1e-12, "Scale không fit đúng train.")
    require(coord["fit_indices_sha256"] == sha256(args.split_dir / "train_indices.csv"), "Sai tập dùng fit.")
    normalized = (xy_train - expected_mean) / expected_scale
    require(np.allclose(normalized * expected_scale + expected_mean, xy_train), "Giải chuẩn hóa sai.")
    require(json.loads((args.split_dir / "ap_order.json").read_text()) == AP_COLUMNS, "Sai thứ tự AP.")
    demo = feature_transform(np.asarray([[-52, 100, -104, 0]], dtype=np.int16))
    require(np.array_equal(demo, np.asarray([[[0.5, 1], [0, 0], [0, 1], [1, 1]]])), "Sai sentinel/mask.")
    example = feature_transform(development.rssi[indices["train"][:2]])
    require(example.shape == (2, 520, 2), "Sai tensor AP.")
    require(example.reshape(2, 26, 40).shape == (2, 26, 40), "Sai tensor Mixer.")
    print("PASS: raw/artifact hashes, coverage, groups, 13 classes, train-only coordinate fit, RSSI mask.")
    print(f"Internal overlap (distinct keys): {overlap}")
    print("Dữ liệu sẵn sàng cho chặng 6; chưa huấn luyện mô hình.")


if __name__ == "__main__":
    main()
