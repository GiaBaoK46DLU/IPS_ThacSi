"""Dự đoán từ RSSI mới và đánh giá checkpoint trên split đã lưu."""
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .data import AP_COLUMNS, COLUMNS, csv_bytes, feature_transform, sha256
from .dataset import load_bundle
from .metrics import evaluate_predictions, inverse_xy
from .runtime import configure_runtime, load_checkpoint, save_json
from .train import infer


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


def prediction_rows(logits, normalized_xy, checkpoint, indices):
    probabilities = logits.softmax(dim=1)
    confidence, predicted = probabilities.max(dim=1)
    predicted = predicted.numpy()
    coordinates = inverse_xy(normalized_xy.numpy(), checkpoint["preprocessing"])
    rows = []
    for position, index in enumerate(indices):
        cls = checkpoint["classes"][int(predicted[position])]
        rows.append([int(index), int(index) + 2, int(predicted[position]), cls["building"], cls["floor"],
                     float(confidence[position]), *coordinates[position].tolist()])
    return rows, predicted, coordinates


PREDICTION_HEADER = ["row_index", "csv_line_number", "class_id", "pred_building", "pred_floor",
                     "max_softmax_probability", "pred_x", "pred_y"]


def predict(checkpoint_path: Path, input_path: Path, output_path: Path, device_name: str):
    if output_path.exists():
        raise FileExistsError(f"Output already exists: {output_path}")
    device = configure_runtime(0, device_name)
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    rssi = read_fingerprints(input_path, checkpoint["ap_order"])
    features = torch.from_numpy(feature_transform(rssi))
    loader = DataLoader(TensorDataset(features, torch.zeros(len(rssi), dtype=torch.long),
                                    torch.zeros(len(rssi), 2)), batch_size=64, shuffle=False)
    logits, xy, _ = infer(model, loader, device)
    rows, _, _ = prediction_rows(logits, xy, checkpoint, np.arange(len(rssi)))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(csv_bytes(PREDICTION_HEADER, rows))
    print(f"Saved {len(rows)} predictions: {output_path}")


def evaluate(checkpoint_path: Path, split_dir: Path, split: str, output_dir: Path, device_name: str):
    if output_dir.exists():
        raise FileExistsError(f"Evaluation directory already exists: {output_dir}")
    device = configure_runtime(0, device_name)
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    bundle = load_bundle(split_dir)
    if sha256(bundle.directory / "manifest.json") != checkpoint["split_manifest_sha256"]:
        raise ValueError("Checkpoint and split manifest differ.")
    if (bundle.preprocessing != checkpoint["preprocessing"] or bundle.classes != checkpoint["classes"] or
        bundle.ap_order != checkpoint["ap_order"]):
        raise ValueError("Checkpoint and dataset preprocessing/mapping differ.")
    if split == "test" and checkpoint["run_type"] == "smoke_check":
        raise ValueError("Smoke checkpoints are for pipeline checks; use a finalized training checkpoint for test.")
    dataset = bundle.dataset(split)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    logits, xy, true_classes = infer(model, loader, device)
    rows, predicted, coordinates = prediction_rows(logits, xy, checkpoint, dataset.indices.numpy())
    metrics = evaluate_predictions(true_classes.numpy(), predicted, dataset.xy_original, coordinates, bundle.classes)
    errors = np.linalg.norm(coordinates - dataset.xy_original, axis=1)
    detected_aps = dataset.features[:, :, 1].sum(dim=1).numpy().astype(np.int64)
    metrics["per_metadata_group"] = {}
    for name, values in dataset.metadata.items():
        metrics["per_metadata_group"][name] = {str(int(group)): {
            "count": int((values == group).sum()), "mean_2d": float(errors[values == group].mean()),
            "joint_accuracy": float((predicted[values == group] == true_classes.numpy()[values == group]).mean())}
            for group in sorted(set(values))}
    metrics.update({"split": split, "checkpoint": str(checkpoint_path),
                    "checkpoint_sha256": sha256(checkpoint_path), "run_type": checkpoint["run_type"],
                    "seed": checkpoint["seed"], "model": checkpoint["config"]["model"],
                    "split_manifest_sha256": checkpoint["split_manifest_sha256"],
                    "checkpoint_epoch": checkpoint["epoch"], "test_userid_note":
                    "USERID 0 in official test means unrecorded user; do not interpret as one person."})
    pairs = [(item["building"], item["floor"]) for item in bundle.classes]
    for i, row in enumerate(rows):
        b, f = pairs[int(true_classes[i])]
        row.extend([b, f, *dataset.xy_original[i].tolist(), float(errors[i]),
                    int(dataset.metadata["USERID"][i]), int(dataset.metadata["PHONEID"][i]), int(detected_aps[i])])
    metrics["per_detected_ap_bin"] = {}
    for name, lower, upper in [("0", 0, 0), ("1-9", 1, 9), ("10-19", 10, 19),
                                ("20-29", 20, 29), ("30+", 30, 520)]:
        mask = (detected_aps >= lower) & (detected_aps <= upper)
        metrics["per_detected_ap_bin"][name] = {"count": int(mask.sum()),
            "mean_2d": float(errors[mask].mean()) if mask.any() else None,
            "joint_accuracy": float((predicted[mask] == true_classes.numpy()[mask]).mean()) if mask.any() else None}
    output_dir.mkdir(parents=True)
    save_json(output_dir / "metrics.json", metrics)
    (output_dir / "predictions.csv").write_bytes(csv_bytes(PREDICTION_HEADER +
        ["true_building", "true_floor", "true_x", "true_y", "error_2d", "USERID", "PHONEID", "detected_aps"], rows))
    print(f"{split}: n={len(dataset)} mean2d={metrics['mean_2d']:.3f} joint_acc={metrics['joint_accuracy']:.4f}")
    return metrics
