"""Validation-only calibration of an already selected reference KAN-Mixer.

python -m ips.algorithms.kan.accuracy --run-dir ... --output-dir ...
The official test is deliberately not accessible through this entry point.
"""
from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import torch

from ...data import ROOT, sha256
from ...data_access import load_bundle
from ...metrics import evaluate_predictions, inverse_xy
from .dataset import FingerprintDataset
from .models.refinement import db_features
from .runtime import configure_runtime, load_checkpoint, save_json


def decode_reference(scores, model, k, temperature):
    padded = scores[:, model.catalogue_indices].masked_fill(~model.catalogue_valid[None], float("-inf"))
    logits = torch.logsumexp(padded, -1)
    top_scores, indices = padded.topk(min(k, padded.shape[-1]), -1)
    positions = model.reference_xy[model.catalogue_indices][None].expand(len(scores), -1, -1, -1)
    xy = positions.gather(2, indices[..., None].expand(-1, -1, -1, 2))
    conditional = ((top_scores / temperature).softmax(-1)[..., None] * xy).sum(2)
    predicted = logits.argmax(1)
    return {"logits": logits, "conditional_xy": conditional,
            "xy": conditional[torch.arange(len(scores), device=scores.device), predicted]}


@torch.inference_mode()
def calibrate(run_dir: Path, output_dir: Path, device_name="cuda", hybrid=True):
    if not (run_dir / "summary.json").is_file():
        raise ValueError("Calibration requires a completed training run with a selected checkpoint.")
    output_dir.mkdir(parents=True, exist_ok=False)
    device = configure_runtime(42, device_name, 2)
    started = time.perf_counter()
    model, checkpoint = load_checkpoint(run_dir / "best.pt", device)
    if checkpoint.get("metadata", {}).get("full_development_refit"):
        raise ValueError("Final refit incorporated validation; inference settings must already be frozen.")
    if checkpoint["config"].get("positioning", {}).get("variant") != "reference":
        raise ValueError("This calibration tool requires a reference KAN-Mixer checkpoint.")
    if checkpoint["run_type"] != "training":
        raise ValueError("Smoke runs cannot be used in accuracy research.")
    bundle = load_bundle(ROOT / checkpoint["config"]["split_dir"])
    train = FingerprintDataset.from_split(bundle.split("train"))
    validation = FingerprintDataset.from_split(bundle.split("validation"))
    if checkpoint["split_manifest_sha256"] != sha256(bundle.directory / "manifest.json"):
        raise ValueError("Checkpoint split differs from the audited bundle.")
    scores = torch.cat([model(x.to(device))["reference_logits"] for x in validation.features.split(128)])
    rows = []

    def measure(output):
        prediction = output["logits"].argmax(1).cpu().numpy()
        predicted_xy = inverse_xy(output["xy"].cpu().numpy(), bundle.preprocessing)
        return evaluate_predictions(validation.classes.numpy(), prediction,
                                    validation.xy_original, predicted_xy, bundle.classes)

    for k in (1, 3, 5, 10, 20, 1000):
        for temperature in (0.5, 1.0, 2.0, 4.0):
            output = decode_reference(scores, model, k, temperature)
            metrics = measure(output)
            rows.append({"kind": "pure_reference_kan_mixer", "top_k": k,
                         "temperature": temperature, "mean_2d": metrics["mean_2d"],
                         "joint_accuracy": metrics["joint_accuracy"]})
    pure = min(rows, key=lambda row: row["mean_2d"]).copy()
    print(f"Pure KAN validation: {pure}", flush=True)
    neural = decode_reference(scores, model, pure["top_k"], pure["temperature"])
    best = pure.copy()
    catalogue = None
    if hybrid:
        train_signal = db_features(train.features).to(device)
        query_signal = db_features(validation.features).to(device)
        train_xy, labels = train.xy.to(device), train.classes.to(device)
        # One raw-RSSI distance matrix, shared by every candidate setting.
        squared = torch.cat([(q.square().sum(1, keepdim=True) + train_signal.square().sum(1)[None]
                    - 2 * q @ train_signal.T).clamp_min(0) for q in query_signal.split(128)])
        neural_probability = neural["logits"].softmax(1)
        for k in (3, 5, 10, 20):
            for power in (1.0, 2.0):
                d, index = squared.topk(k, largest=False)
                weights = (d.sqrt() + 1e-4).pow(-power)
                neighbor_labels = labels[index]
                votes = torch.stack([(weights * (neighbor_labels == c)).sum(1) for c in range(len(bundle.classes))], 1)
                neighbor_probability = votes / votes.sum(1, keepdim=True)
                for strength in (0.0, 0.5, 1.0):
                    probabilities = (1 - strength) * neural_probability + strength * neighbor_probability
                    predicted = probabilities.argmax(1)
                    floor_distance = squared.masked_fill(labels[None] != predicted[:, None], float("inf"))
                    d, index = floor_distance.topk(k, largest=False)
                    weights = (d.sqrt() + 1e-4).pow(-power)
                    weights = weights / weights.sum(1, keepdim=True)
                    neighbor_xy = (weights[..., None] * train_xy[index]).sum(1)
                    neural_xy = neural["conditional_xy"][torch.arange(len(scores), device=device), predicted]
                    for alpha in (0.25, 0.5, 0.75):
                        output = {"logits": probabilities.clamp_min(1e-12).log(),
                                  "xy": alpha * neural_xy + (1 - alpha) * neighbor_xy}
                        metrics = measure(output)
                        row = {"kind": "kan_mixer_fingerprint_hybrid", "top_k": pure["top_k"],
                               "temperature": pure["temperature"], "neighbors": k,
                               "distance_power": power, "neighbor_class_weight": strength,
                               "neural_coordinate_weight": alpha, "mean_2d": metrics["mean_2d"],
                               "joint_accuracy": metrics["joint_accuracy"]}
                        rows.append(row)
                        if row["mean_2d"] < best["mean_2d"]:
                            best = row.copy()
        catalogue = {"signal": train_signal.cpu(), "xy": train_xy.cpu(), "class_ids": labels.cpu(),
                     "row_indices": train.indices.clone(), "fit_scope": "internal_train_only",
                     "train_indices_sha256": sha256(bundle.directory / "train_indices.csv")}
    payload = copy.deepcopy(checkpoint)
    payload["config"]["positioning"].update(top_k=best["top_k"], temperature=best["temperature"])
    if best["kind"] == "kan_mixer_fingerprint_hybrid":
        catalogue["settings"] = {k: best[k] for k in ["neighbors", "distance_power",
            "neighbor_class_weight", "neural_coordinate_weight"]}
        payload["fingerprint_refinement"] = catalogue
        payload["format_version"] = 2
    payload["validation_mean_2d"] = best["mean_2d"]
    calibration = {"selection_split": "internal_validation_only", "test_evaluated": False,
        "candidate_count": len(rows), "candidates": sorted(rows, key=lambda r: r["mean_2d"]),
        "best_pure": pure, "selected": best, "source_checkpoint_sha256": sha256(run_dir / "best.pt"),
        "train_indices_sha256": sha256(bundle.directory / "train_indices.csv"),
        "validation_indices_sha256": sha256(bundle.directory / "validation_indices.csv"),
        "elapsed_seconds": time.perf_counter() - started}
    payload["metadata"]["validation_inference_calibration"] = calibration
    torch.save(payload, output_dir / "selected.pt")
    pure_payload = copy.deepcopy(checkpoint)
    pure_payload["config"]["positioning"].update(top_k=pure["top_k"], temperature=pure["temperature"])
    pure_payload["validation_mean_2d"] = pure["mean_2d"]
    pure_payload["metadata"]["validation_inference_calibration"] = {**calibration, "selected": pure}
    torch.save(pure_payload, output_dir / "pure_selected.pt")
    save_json(output_dir / "calibration.json", calibration)
    print(f"Selected validation: {best}; candidates={len(rows)}; official test unopened", flush=True)
    return calibration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pure-only", action="store_true")
    args = parser.parse_args()
    calibrate(args.run_dir, args.output_dir, args.device, not args.pure_only)


if __name__ == "__main__":
    main()
