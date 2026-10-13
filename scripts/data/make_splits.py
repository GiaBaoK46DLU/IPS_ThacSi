"""Hiện thực giao thức chặng 4; lưu chỉ số dòng, schema và tiền xử lý."""
import argparse
import hashlib
import json
import platform
from collections import Counter
from pathlib import Path

import numpy as np
import sklearn
from sklearn.model_selection import StratifiedGroupKFold

from data_utils import (AP_COLUMNS, GROUP_COLUMNS, OUTPUT, RAW, ROOT, csv_bytes,
                        encode_groups, json_bytes, overlaps, read_data, sha256, write_bundle)

DESIGN = Path(__file__).with_name("split_policy.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    # Mã hóa groups + phiên bản thư viện ảnh hưởng split: khóa bản đã kiểm tra.
    if sklearn.__version__ != "1.6.1" or np.__version__ != "2.2.6":
        raise RuntimeError("Để tạo lại split v1: numpy==2.2.6, scikit-learn==1.6.1.")
    design = json.loads(DESIGN.read_text(encoding="utf-8"))
    settings = design["split"]
    if settings["group_key_columns"] != GROUP_COLUMNS:
        raise ValueError("Group schema chặng 4 đã đổi; tạo phiên bản split mới.")
    development = read_data(ROOT / design["development_file"])
    final_test = read_data(ROOT / design["final_test_file"])
    class_pairs = sorted(set(development.pairs))
    if [list(pair) for pair in class_pairs] != design["class_pairs"]:
        raise ValueError("Class mapping không khớp thiết kế chặng 4.")
    if not set(final_test.pairs).issubset(class_pairs):
        raise ValueError("Test chứa lớp chưa có trong tập phát triển; cần ghi nhận, không fit mapping trên test.")
    mapping = {pair: index for index, pair in enumerate(class_pairs)}
    targets = np.asarray([mapping[pair] for pair in development.pairs], dtype=np.int64)
    groups = encode_groups(development)
    splitter = StratifiedGroupKFold(n_splits=settings["n_splits"], shuffle=settings["shuffle"],
                                    random_state=settings["random_state"])
    folds = splitter.split(np.zeros((development.size, 1)), targets, groups)
    train_indices, validation_indices = list(folds)[settings["validation_fold_index"]]
    splits = {"train": train_indices, "validation": validation_indices,
              "test": np.arange(final_test.size, dtype=np.int64)}
    if set(groups[train_indices]) & set(groups[validation_indices]):
        raise ValueError("Rò rỉ nhóm train/validation.")
    for indices in [train_indices, validation_indices]:
        if set(targets[indices]) != set(range(len(class_pairs))):
            raise ValueError("Thiếu lớp ở train hoặc validation; không tự đổi seed để chọn kết quả đẹp.")
    internal_overlap = overlaps(development, development, train_indices, validation_indices)
    if internal_overlap["full_distinct_keys"] or internal_overlap["labeled_rssi_distinct_keys"]:
        raise ValueError("Trùng bản ghi hoặc trùng fingerprint kèm nhãn vị trí giữa train/validation.")
    files = {}
    for split, indices in splits.items():
        source_groups = groups if split != "test" else None
        rows = [[int(i), int(i) + 2, int(source_groups[i]) if source_groups is not None else ""] for i in indices]
        files[f"{split}_indices.csv"] = csv_bytes(["row_index", "csv_line_number", "group_id"], rows)
    files["ap_order.json"] = json_bytes(AP_COLUMNS)
    files["class_mapping.json"] = json_bytes([
        {"class_id": i, "building": b, "floor": f} for i, (b, f) in enumerate(class_pairs)])
    # Chỉ lấy nhãn tọa độ của train nội bộ để fit, không lấy toàn development.
    xy = development.metadata[train_indices, :2]
    center = xy.mean(axis=0)
    scale = max(float(np.sqrt(xy.var(axis=0, ddof=0).mean())),
                design["preprocessing"]["coordinate_scale_minimum"])
    files["preprocessing.json"] = json_bytes({
        "rssi": {"sentinel": 100, "detected_range": [-104, 0], "z_detected": "(rssi+104)/104",
                 "z_undetected": 0, "mask_detected": 1, "mask_undetected": 0,
                 "ap_feature_order": ["z", "mask"], "removed_aps": [], "fit_required": False,
                 "all_undetected_policy": "retain"},
        "coordinates": {"order": ["LONGITUDE", "LATITUDE"], "center": center.tolist(),
                        "shared_scale": scale, "variance_ddof": 0, "transform": "(xy-center)/shared_scale",
                        "inverse": "normalized_xy*shared_scale+center", "fit_split": "train",
                        "fit_source": design["development_file"], "fit_rows": len(train_indices),
                        "fit_indices_sha256": hashlib.sha256(files["train_indices.csv"]).hexdigest(),
                        "crs_epsg": None, "units": "dataset_metre_convention_not_verified_full_CRS"},
        "shapes_without_batch": {"AP_features": [520, 2], "MLP_input": [1040], "Mixer_input": [26, 40]}})
    class_counts = {}
    for split, indices in splits.items():
        source_pairs = final_test.pairs if split == "test" else development.pairs
        counts = Counter(source_pairs[int(i)] for i in indices)
        class_counts[split] = {f"{b}_{f}": counts.get((b, f), 0) for b, f in class_pairs}
    files["split_label_distribution.csv"] = csv_bytes(["class_id", "building", "floor", "train", "validation", "test"],
        [[i, b, f] + [class_counts[split][f"{b}_{f}"] for split in splits]
         for i, (b, f) in enumerate(class_pairs)])
    manifest = {
        "protocol": "uji_group_split_v1", "status": "materialized_and_checked_not_trained",
        "sources": {"development": {"path": design["development_file"], "sha256": development.digest,
                                     "rows": development.size},
                    "final_test": {"path": design["final_test_file"], "sha256": final_test.digest,
                                   "rows": final_test.size}},
        "design_path": DESIGN.relative_to(ROOT).as_posix(), "design_sha256": sha256(DESIGN),
        "software": {"python": platform.python_version(), "numpy": np.__version__, "scikit_learn": sklearn.__version__},
        "group_encoding": "numeric tuple in group_key_columns order; sorted unique tuples; zero-based IDs",
        "split_settings": {key: value for key, value in settings.items() if key != "state"},
        "counts": {key: len(value) for key, value in splits.items()},
        "group_counts": {"development": len(set(groups)), "train": len(set(groups[train_indices])),
                         "validation": len(set(groups[validation_indices])), "train_validation_overlap": 0},
        "class_counts": class_counts,
        "internal_train_validation_overlap": internal_overlap,
        "original_development_final_test_overlap": overlaps(development, final_test),
        "all_undetected_rows": {key: int((source.rssi[indices] == 100).all(axis=1).sum())
            for key, indices, source in [("train", train_indices, development),
                                         ("validation", validation_indices, development),
                                         ("test", splits["test"], final_test)]},
        "limitations": ["group is a proxy, not an official acquisition session",
                        "users/devices/spatial locations may overlap internal train and validation",
                        "equal RSSI with the same building-floor but different xy can cross internal split",
                        "test USERID=0 is not evidence of one physical person"],
        "artifact_sha256": {name: hashlib.sha256(content).hexdigest() for name, content in files.items()},
        "generator_sha256": {name: sha256(Path(__file__).parent / name)
                              for name in ["data_utils.py", "make_splits.py"]},
        "shared_reader": {"path": "src/ips/data.py", "sha256": sha256(ROOT / "src/ips/data.py")},
        "row_index_convention": "zero-based data rows excluding header; csv_line_number=row_index+2",
        "test_group_id": "empty: official test not generated by internal group splitter",
        "model_results": None,
    }
    files["manifest.json"] = json_bytes(manifest)
    write_bundle(args.output_dir, files)
    print(f"PASS: train={len(train_indices)}, validation={len(validation_indices)}, test={final_test.size}; group_overlap=0")
    print(f"Chỉ số và manifest: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
