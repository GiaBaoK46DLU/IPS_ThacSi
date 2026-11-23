"""Cross-device proxy on development data only (Test is never read).

Official Test uses 11 phones, 9 of which never appear in training, while the
internal Validation shares all 16 training phones. This proxy holds out a
phone group: the model trains on internal-train rows of the remaining phones
(early stopping on their internal-Validation rows, as usual) and is scored on
internal-Validation rows of the held-out phones with the config's decoder.

Reported: mean2D on all held-out-phone rows and on rows whose (class, x, y)
coordinate also exists in the reduced train (separates device shift from
coverage gaps). Proxy only; not a substitute for the official protocol.
"""
from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from screen_kan_fastkan import V4_CONFIG, apply_changes
from ips.data_access import load_bundle
from ips.metrics import evaluate_predictions, inverse_xy
from ips.algorithms.kan.dataset import FingerprintDataset
from ips.algorithms.kan.runtime import load_checkpoint, save_json

SPLIT = ROOT / "data/processed/uji_group_split_v1"
OUTPUT = ROOT / "runs/kan_fastkan_crossdevice_20261123"
FOLDS = {"A": [1, 6, 8, 10, 17, 18, 23], "B": [3, 7, 11, 16, 19, 22, 24]}


def subset(dataset, keep):
    keep = np.asarray(keep, dtype=bool)
    result = FingerprintDataset.__new__(FingerprintDataset)
    result.split_name = dataset.split_name
    index = torch.from_numpy(np.flatnonzero(keep))
    result.features = dataset.features[index].clone()
    result.indices = dataset.indices[index].clone()
    result.xy_original = dataset.xy_original[keep].copy()
    result.xy = dataset.xy[index].clone()
    result.classes = dataset.classes[index].clone()
    result.metadata = {k: v[keep].copy() for k, v in dataset.metadata.items()}
    return result


def score(checkpoint, data, bundle):
    model, payload = load_checkpoint(checkpoint, torch.device("cuda"))
    loader = DataLoader(data, batch_size=512, shuffle=False)
    logits, xy = [], []
    with torch.no_grad():
        for features, _, _ in loader:
            output = model(features.cuda())
            logits.append(output["logits"].cpu())
            xy.append(output["xy"].cpu())
    predicted = torch.cat(logits).argmax(1).numpy()
    predicted_xy = inverse_xy(torch.cat(xy).numpy(), payload["preprocessing"])
    return evaluate_predictions(data.classes.numpy(), predicted, data.xy_original, predicted_xy, bundle.classes)


def run_fold(name, config, fold, seed, bundle, train_all, validation_all):
    from ips.algorithms.kan.train import train
    held = FOLDS[fold]
    train_phone = train_all.metadata["PHONEID"].astype(int)
    validation_phone = validation_all.metadata["PHONEID"].astype(int)
    train_in = subset(train_all, ~np.isin(train_phone, held))
    validation_in = subset(validation_all, ~np.isin(validation_phone, held))
    validation_out = subset(validation_all, np.isin(validation_phone, held))
    if len(set(train_in.classes.tolist())) != len(bundle.classes):
        raise ValueError(f"Fold {fold} leaves a building/floor class without training rows.")
    destination = OUTPUT / name / f"fold{fold}_seed{seed}"
    if not (destination / "training/summary.json").exists():
        train(copy.deepcopy(config), destination / "training", seed, "cuda",
              bundle=bundle, train_data=train_in, validation_data=validation_in)
    known = {(int(c), float(x), float(y)) for c, (x, y) in zip(train_in.classes.tolist(), train_in.xy_original)}
    seen = np.array([(int(c), float(x), float(y)) in known
                     for c, (x, y) in zip(validation_out.classes.tolist(), validation_out.xy_original)])
    checkpoint = destination / "training/best.pt"
    all_rows = score(checkpoint, validation_out, bundle)
    seen_rows = score(checkpoint, subset(validation_out, seen), bundle) if seen.any() else None
    in_rows = score(checkpoint, validation_in, bundle)
    result = {"fold": fold, "seed": seed, "held_out_phones": held, "rows_out": len(validation_out),
              "rows_out_seen_coordinate": int(seen.sum()), "out_mean_2d": all_rows["mean_2d"],
              "out_seen_mean_2d": seen_rows["mean_2d"] if seen_rows else None,
              "out_floor_accuracy": all_rows["floor_accuracy"], "in_mean_2d": in_rows["mean_2d"],
              "best_epoch": json.loads((destination / "training/summary.json").read_text())["best_epoch"],
              "test_evaluated": False}
    save_json(destination / "proxy_result.json", result)
    print(f"PROXY {name} fold={fold} seed={seed} out={result['out_mean_2d']:.4f} "
          f"out_seen={result['out_seen_mean_2d']:.4f} in={result['in_mean_2d']:.4f}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--changes", default="{}")
    parser.add_argument("--folds", default="A,B")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    changes = json.loads(args.changes)
    config = apply_changes(json.loads(V4_CONFIG.read_text(encoding="utf-8")), changes)
    bundle = load_bundle(SPLIT)
    train_all = FingerprintDataset.from_split(bundle.split("train"))
    validation_all = FingerprintDataset.from_split(bundle.split("validation"))
    folder = OUTPUT / args.name
    folder.mkdir(parents=True, exist_ok=True)
    save_json(folder / "candidate.json", {"name": args.name, "changes": changes, "config": config})
    results = [run_fold(args.name, config, fold, args.seed, bundle, train_all, validation_all)
               for fold in args.folds.split(",")]
    summary = {"name": args.name, "changes": changes, "per_fold": results,
               "out_mean_2d": statistics.mean(r["out_mean_2d"] for r in results),
               "out_seen_mean_2d": statistics.mean(r["out_seen_mean_2d"] for r in results),
               "in_mean_2d": statistics.mean(r["in_mean_2d"] for r in results), "test_evaluated": False}
    save_json(folder / f"summary_seed{args.seed}.json", summary)
    print("PROXY_SUMMARY " + json.dumps({k: summary[k] for k in
          ("name", "out_mean_2d", "out_seen_mean_2d", "in_mean_2d")}), flush=True)


if __name__ == "__main__":
    main()
