"""Reproduce the historical V2 reference configuration, then refit ten seeds.

All ten validation trials and all ten refits start from fresh seeded weights.
No search over architectures, no hybrid refinement, and no ensemble.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ips.data import sha256
from ips.data_access import load_bundle
from ips.metrics import evaluate_predictions
from ips.algorithms.kan.runtime import code_hashes, load_checkpoint, save_json

SEEDS = [42, 43, 44, 100, 2024, 777, 999, 45, 46, 47]
CONFIG = ROOT / "src/ips/algorithms/kan/configs/v2_reference_reproduction.json"
SPLIT = ROOT / "data/processed/uji_group_split_v1"
HISTORICAL = {
    42: ("runs/kan_accuracy_20261009/spatial_seed42", "runs/kan_accuracy_20261009/spatial_seed42_calibration"),
    43: ("runs/kan_accuracy_20261009/selected_seed43", "runs/kan_accuracy_20261009/selected_seed43_calibration"),
    44: ("runs/kan_seed44_complete_20261009_v1/training", "runs/kan_seed44_complete_20261009_v1/calibration"),
}
METRICS = ["mean_2d", "median_2d", "p90_2d", "rmse_2d", "building_accuracy", "floor_accuracy", "joint_accuracy"]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def immutable(path, value):
    if path.exists():
        if read(path) != value:
            raise ValueError(f"Frozen artifact changed: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        save_json(path, value)


def training_design(config):
    result = copy.deepcopy(config)
    result["positioning"].pop("references", None)
    result["positioning"].pop("fit_scope", None)
    return result


def refit_epochs(results):
    if [r["seed"] for r in results] != SEEDS:
        raise ValueError("Exactly the ten predeclared validation seeds are required, in order.")
    epochs = [r["best_epoch"] for r in results]
    if any(type(e) is not int or not 1 <= e <= 100 for e in epochs):
        raise ValueError("Invalid validation best epoch.")
    return round(statistics.median(epochs))


def prepare(output):
    if output.exists():
        raise FileExistsError(output)
    bundle = load_bundle(SPLIT)
    config = read(CONFIG)
    if config != read(ROOT / "runs/kan_seed44_complete_20261009_v1/training_config.json"):
        raise ValueError("V2 configuration differs from historical training input.")
    output.mkdir(parents=True)
    guards = {str(CONFIG.relative_to(ROOT)): sha256(CONFIG),
              str(Path(__file__).relative_to(ROOT)): sha256(Path(__file__)),
              "data/processed/uji_group_split_v1/manifest.json": sha256(SPLIT / "manifest.json")}
    guards.update(code_hashes())
    for seed, (training, calibration) in HISTORICAL.items():
        paths = [ROOT / training / name for name in ("summary.json", "resolved_config.json")]
        if training_design(read(paths[1])) != config:
            raise ValueError(f"Historical seed {seed} is not the same V2 training design.")
        for path in paths:
            guards[path.relative_to(ROOT).as_posix()] = sha256(path)
    for baseline in ("docs/Bao/KAN_V4/official_test_summary.json", "runs/kan_joint_under6_20261020_v1/official_test_summary.json"):
        guards[baseline] = sha256(ROOT / baseline)
    immutable(output / "protocol.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(), "seeds": SEEDS,
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        "config": config, "guard_sha256": guards,
        "split_manifest_sha256": sha256(SPLIT / "manifest.json"),
        "counts": {"internal_train": len(bundle.split("train").class_ids),
                   "validation": len(bundle.split("validation").class_ids), "development": bundle.development.size},
        "historical_validation_seeds_reused": [], "all_ten_validation_trials_fresh": True,
        "refit_epoch_rule": "rounded median of ten validation-best epochs; cosine horizon remains100",
        "decoder_rule": "V2 per-seed Validation-only calibration: k in1,3,5,10,20,1000; T in0.5,1,2,4",
        "test_policy": "all ten refit checkpoints frozen before any new Test evaluation; full1111 rows each; mean and sample SD(ddof1)",
        "historical_test_exposure": True, "test_evaluated_in_this_experiment": False,
        "refit_initialization": "fresh weights for every seed, not fine tuning", "ensemble": False, "knn": False,
    })


def verify(output):
    protocol = read(output / "protocol.json")
    for rel, digest in protocol["guard_sha256"].items():
        if sha256(ROOT / rel) != digest:
            raise ValueError(f"Protected source/data/baseline changed: {rel}")
    return protocol


def validation_trial(output, seed, config):
    from ips.algorithms.kan.train import train
    from ips.algorithms.kan.accuracy import calibrate
    from ips.algorithms.kan.predict import evaluate
    destination = output / "validation_trials" / f"seed{seed}"
    destination.mkdir(parents=True, exist_ok=True)
    if not (destination / "training/summary.json").exists():
        train(copy.deepcopy(config), destination / "training", seed, "cuda")
    if not (destination / "calibration/calibration.json").exists():
        calibrate(destination / "training", destination / "calibration", "cuda", hybrid=False)
    checkpoint = destination / "calibration/pure_selected.pt"
    if not (destination / "validation/metrics.json").exists():
        evaluate(checkpoint, SPLIT, "validation", destination / "validation", "cuda")
    summary = read(destination / "training/summary.json")
    calibration = read(destination / "calibration/calibration.json")
    metrics = read(destination / "validation/metrics.json")
    _, payload = load_checkpoint(checkpoint, torch.device("cpu"))
    if (summary["seed"] != seed or payload["seed"] != seed or metrics["checkpoint_sha256"] != sha256(checkpoint)
            or summary["test_evaluated"] is not False or calibration["test_evaluated"] is not False
            or calibration["source_checkpoint_sha256"] != sha256(destination / "training/best.pt")):
        raise ValueError("Invalid Validation provenance.")
    if training_design(read(destination / "training/resolved_config.json")) != config:
        raise ValueError("Training settings changed from V2.")
    result = {"seed": seed, "best_epoch": summary["best_epoch"], "mean_2d": metrics["mean_2d"],
              "decoder": calibration["best_pure"], "checkpoint_sha256": sha256(checkpoint),
              "reused_historical_validation": False, "test_evaluated": False}
    immutable(destination / "result.json", result)
    return result


def audit_test(checkpoint, destination):
    metrics = read(destination / "metrics.json")
    with (destination / "predictions.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    truth = load_bundle(SPLIT).split("test")
    indices = np.array([int(r["row_index"]) for r in rows])
    xy = np.array([[float(r["pred_x"]), float(r["pred_y"])] for r in rows])
    saved_truth = np.array([[float(r["true_x"]), float(r["true_y"])] for r in rows])
    pairs = {(c["building"], c["floor"]): c["class_id"] for c in truth.info.classes}
    predicted = np.array([pairs[(int(r["pred_building"]), int(r["pred_floor"]))] for r in rows])
    actual = evaluate_predictions(truth.class_ids, predicted, truth.xy, xy, truth.info.classes)
    if (len(rows) != 1111 or not np.array_equal(indices, truth.inputs.row_indices)
            or not np.array_equal(saved_truth, truth.xy) or metrics["checkpoint_sha256"] != sha256(checkpoint)
            or any(abs(actual[k] - metrics[k]) > 1e-10 for k in METRICS)):
        raise ValueError("Independent Test row/truth/metric audit failed.")
    immutable(destination / "independent_audit.json", {"status": "passed", "rows": 1111,
        "checkpoint_sha256": sha256(checkpoint), "predictions_sha256": sha256(destination / "predictions.csv"),
        "metrics_recomputed": {k: actual[k] for k in METRICS}})
    return {"seed": metrics["seed"], **{k: metrics[k] for k in METRICS}}


def run(output):
    from ips.algorithms.kan.final_refit import refit
    from ips.algorithms.kan.predict import evaluate
    protocol = verify(output)
    results = []
    for seed in SEEDS:
        verify(output)
        save_json(output / "state.json", {"stage": "validation", "current_seed": seed, "completed": len(results)})
        results.append(validation_trial(output, seed, protocol["config"]))
    sources = [output / "validation_trials" / f"seed{s}" / "calibration/pure_selected.pt" for s in SEEDS]
    lock = {"scope": "one_joint_kan_mixer", "selection_split": "internal_validation_only", "test_evaluated": False,
        "seeds": SEEDS, "final_refit_epochs": refit_epochs(results), "all_ten_validation_results": results,
        "source_sha256": protocol["guard_sha256"], "split_manifest_sha256": protocol["split_manifest_sha256"],
        "allowed_source_checkpoint_sha256": [sha256(p) for p in sources],
        "training_configs_by_source_sha256": {sha256(p): {"path": str(p.parent.parent / "training/resolved_config.json"),
            "sha256": sha256(p.parent.parent / "training/resolved_config.json")} for p in sources}}
    immutable(output / "selection_lock.json", lock)
    for seed, source in zip(SEEDS, sources):
        verify(output)
        save_json(output / "state.json", {"stage": "refit", "current_seed": seed, "epochs": lock["final_refit_epochs"]})
        destination = output / "final_refit" / f"seed{seed}"
        if not (destination / "summary.json").exists():
            refit(source, output / "selection_lock.json", destination, seed, "cuda")
    checkpoints = [output / "final_refit" / f"seed{s}" / "best.pt" for s in SEEDS]
    hashes = [sha256(p) for p in checkpoints]
    if len(set(hashes)) != 10:
        raise ValueError("Ten different refit checkpoints are required.")
    immutable(output / "test_checkpoint_lock.json", {"test_evaluated_at_creation": False,
        "selection_lock_sha256": sha256(output / "selection_lock.json"),
        "checkpoints": [{"seed": s, "path": str(p), "sha256": h} for s, p, h in zip(SEEDS, checkpoints, hashes)]})
    per_seed = []
    for seed, checkpoint, digest in zip(SEEDS, checkpoints, hashes):
        verify(output)
        if sha256(checkpoint) != digest:
            raise ValueError("Refit checkpoint changed after freezing.")
        save_json(output / "state.json", {"stage": "official_test", "current_seed": seed})
        destination = output / "official_test" / f"seed{seed}"
        if not (destination / "metrics.json").exists():
            evaluate(checkpoint, SPLIT, "test", destination, "cuda")
        per_seed.append(audit_test(checkpoint, destination))
    aggregate = {k: {"mean": statistics.mean(r[k] for r in per_seed),
                     "sample_std": statistics.stdev(r[k] for r in per_seed)} for k in METRICS}
    immutable(output / "official_test_summary.json", {"config_name": "V2_reference", "regime": "full_development_refit",
        "n_models": 10, "test_rows_per_model": 1111, "seeds": SEEDS, "refit_epochs": lock["final_refit_epochs"],
        "per_seed": per_seed, "aggregate": aggregate, "historical_test_exposure": True, "no_reselection_after_test": True})
    save_json(output / "state.json", {"stage": "completed", "mean_2d": aggregate["mean_2d"], "audited_rows": 11110})
    print("V2_REFIT_COMPLETE " + json.dumps(aggregate), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "run"])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if args.action == "prepare":
        prepare(output)
    else:
        try:
            run(output)
        except Exception as error:
            save_json(output / "failure.json", {"type": type(error).__name__, "message": str(error),
                "utc": datetime.now(timezone.utc).isoformat()})
            raise


if __name__ == "__main__":
    main()
