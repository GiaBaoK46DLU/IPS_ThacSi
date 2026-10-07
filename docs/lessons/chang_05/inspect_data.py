"""Kiểm kê CSV gốc; đây là thống kê dữ liệu, không phải kết quả mô hình."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from data_utils import AP_COLUMNS, RAW, ROOT, csv_bytes, json_bytes, overlaps, read_data, row_keys, write_bundle


def stats(data):
    detected = data.rssi != 100
    signals = data.rssi[detected]
    counts = detected.sum(axis=1)
    result = {
        "path": data.path.relative_to(ROOT).as_posix(), "sha256": data.digest,
        "rows": data.size, "columns": 529, "ap_columns": 520,
        "blank_nan_inf_invalid_cells": 0,
        "detected_rssi_min_max": [int(signals.min()), int(signals.max())] if len(signals) else None,
        "undetected_100_cells": int((~detected).sum()),
        "undetected_fraction": float((~detected).mean()),
        "all_undetected_rows": int((counts == 0).sum()),
        "never_detected_aps": [name for name, count in zip(AP_COLUMNS, detected.sum(axis=0)) if count == 0],
        "detected_aps_per_row": {"min": int(counts.min()), "median": float(np.median(counts)),
                                 "mean": float(counts.mean()), "max": int(counts.max())},
        "building_floor_counts": {f"{b}_{f}": count for (b, f), count in sorted(Counter(data.pairs).items())},
        "duplicate_excess_full": data.size - len(row_keys(data, "full")),
        "duplicate_excess_rssi": data.size - len(row_keys(data, "rssi")),
        "distinct_locations_building_floor_xy": len(row_keys(data, "location")),
    }
    for name in ["USERID", "PHONEID", "SPACEID", "RELATIVEPOSITION"]:
        result[name.lower() + "_counts"] = {str(int(k)): v for k, v in
                                              sorted(Counter(data.column(name)).items())}
    for name in ["LONGITUDE", "LATITUDE"]:
        result[name.lower() + "_range"] = [float(data.column(name).min()), float(data.column(name).max())]
    timestamps = data.column("TIMESTAMP")
    result["timestamp_range_unix"] = [int(timestamps.min()), int(timestamps.max())]
    result["timestamp_range_utc"] = [datetime.fromtimestamp(int(t), timezone.utc).isoformat()
                                     for t in result["timestamp_range_unix"]]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "artifacts")
    args = parser.parse_args()
    train = read_data(RAW / "trainingData.csv")
    test = read_data(RAW / "validationData.csv")
    report = {"kind": "data_audit_not_model_results", "files": {
        "trainingData.csv": stats(train), "validationData.csv": stats(test)},
        "cross_original_files": overlaps(train, test),
        "duplicate_excess_definition": "number_of_rows_minus_number_of_distinct_numeric_rows"}
    distribution = [[b, f, Counter(train.pairs).get((b, f), 0), Counter(test.pairs).get((b, f), 0)]
                    for b, f in sorted(set(train.pairs) | set(test.pairs))]
    write_bundle(args.output_dir, {"data_audit.json": json_bytes(report),
        "original_label_distribution.csv": csv_bytes(["building", "floor", "development", "final_test"], distribution)})
    for name, values in report["files"].items():
        print(f"{name}: {values['rows']} rows, 520 AP, missing100={values['undetected_100_cells']}, "
              f"all_missing={values['all_undetected_rows']}, never_detected_AP={len(values['never_detected_aps'])}")
    print(f"PASS: audit lưu tại {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
