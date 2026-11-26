"""Validation-only screening of KAN-Mixer candidates (FastKAN, GR-KAN, ...) against V4.

Each candidate = V4 config (v4_balanced_points.json) + a declared dict of
changes. For every seed: fresh training on internal train, Validation-only
decoder calibration, Validation evaluation. Test is never touched here.

Usage:
  python scripts/screen_kan_fastkan.py --name fk_base --changes '{"kan.backend":"fastkan"}'
"""
from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ips.algorithms.kan.runtime import save_json

V4_CONFIG = ROOT / "src/ips/algorithms/kan/configs/v4_balanced_points.json"
SPLIT = ROOT / "data/processed/uji_group_split_v1"
OUTPUT = ROOT / "runs/kan_fastkan_screen_20261123"
V4_VALIDATION = {42: 5.54265224712112, 43: 5.6043077619993955, 44: 5.576649069123984,
                 100: 5.567222060325186, 2024: 5.612166796772303, 777: 5.604045508109041,
                 999: 5.542068127889344, 45: 5.577112049972976, 46: 5.669575816026873,
                 47: 5.557740772667518}


def apply_changes(config, changes):
    result = copy.deepcopy(config)
    for dotted, value in changes.items():
        node = result
        *parents, leaf = dotted.split(".")
        for key in parents:
            node = node.setdefault(key, {})
        if value is None:
            node.pop(leaf, None)
        else:
            node[leaf] = value
    return result


def trial(destination, config, seed):
    from ips.algorithms.kan.train import train
    from ips.algorithms.kan.accuracy import calibrate
    from ips.algorithms.kan.predict import evaluate
    destination.mkdir(parents=True, exist_ok=True)
    if not (destination / "training/summary.json").exists():
        if (destination / "training").exists():
            raise FileExistsError(f"Incomplete training directory: {destination / 'training'}")
        train(copy.deepcopy(config), destination / "training", seed, "cuda")
    if not (destination / "calibration/calibration.json").exists():
        calibrate(destination / "training", destination / "calibration", "cuda", hybrid=False)
    if not (destination / "validation/metrics.json").exists():
        evaluate(destination / "calibration/pure_selected.pt", SPLIT, "validation", destination / "validation", "cuda")
    summary = json.loads((destination / "training/summary.json").read_text(encoding="utf-8"))
    metrics = json.loads((destination / "validation/metrics.json").read_text(encoding="utf-8"))
    calibration = json.loads((destination / "calibration/calibration.json").read_text(encoding="utf-8"))
    return {"seed": seed, "best_epoch": summary["best_epoch"], "epochs_completed": summary["epochs_completed"],
            "parameters": summary["trainable_parameters"], "training_seconds": summary["training_seconds"],
            "mean_2d": metrics["mean_2d"], "median_2d": metrics["median_2d"], "p90_2d": metrics["p90_2d"],
            "floor_accuracy": metrics["floor_accuracy"], "decoder": calibration["best_pure"],
            "v4_mean_2d": V4_VALIDATION[seed], "delta_vs_v4": metrics["mean_2d"] - V4_VALIDATION[seed]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--changes", required=True, help="JSON dict of dotted-key changes to the V4 config")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--output-root", type=Path, default=OUTPUT, help="folder that holds one subfolder per candidate")
    args = parser.parse_args()
    changes = json.loads(args.changes)
    seeds = [int(s) for s in args.seeds.split(",")]
    config = apply_changes(json.loads(V4_CONFIG.read_text(encoding="utf-8")), changes)
    folder = args.output_root / args.name
    folder.mkdir(parents=True, exist_ok=True)
    registered = folder / "candidate.json"
    if registered.exists():
        previous = json.loads(registered.read_text(encoding="utf-8"))
        if previous["config"] != config:
            raise ValueError(f"Candidate {args.name} already registered with a different config.")
    else:
        save_json(registered, {"name": args.name, "changes": changes, "config": config,
                               "registered_utc": datetime.now(timezone.utc).isoformat(), "test_evaluated": False})
    results = []
    for seed in seeds:
        result = trial(folder / f"seed{seed}", config, seed)
        results.append(result)
        print(f"SCREEN {args.name} seed={seed} val={result['mean_2d']:.4f} v4={result['v4_mean_2d']:.4f} "
              f"delta={result['delta_vs_v4']:+.4f} best_epoch={result['best_epoch']}", flush=True)
    summary = {"name": args.name, "changes": changes, "seeds": seeds, "per_seed": results,
               "mean_2d": statistics.mean(r["mean_2d"] for r in results),
               "v4_mean_2d": statistics.mean(r["v4_mean_2d"] for r in results),
               "mean_delta_vs_v4": statistics.mean(r["delta_vs_v4"] for r in results),
               "better_than_v4_seeds": sum(r["delta_vs_v4"] < 0 for r in results),
               "test_evaluated": False}
    save_json(folder / f"summary_{'_'.join(map(str, seeds))}.json", summary)
    print("SCREEN_SUMMARY " + json.dumps({k: summary[k] for k in
          ("name", "mean_2d", "v4_mean_2d", "mean_delta_vs_v4", "better_than_v4_seeds")}), flush=True)


if __name__ == "__main__":
    main()
