"""Independent UJI building experts using the original frozen sample split.

Each expert needs a KNOWN building at prediction time. Reference KAN only.
prepare -> train (three buildings x three seeds) -> lock -> evaluate.
Never select an epoch, a decoder, or a seed from test metrics.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from ...data import ROOT, csv_bytes, sha256
from ...data_access import load_bundle
from ...data_access.contracts import DatasetInfo, FingerprintInputs, FingerprintSplit
from ...metrics import evaluate_predictions, inverse_xy
from .dataset import FingerprintDataset
from .resources import MemorySampler, cuda_memory, synchronize
from .runtime import configure_runtime, load_checkpoint, save_json
from .supplied_experiment import SOURCE_RUNS, SOURCE_TRAINING_CONFIG
from .train import train

PARENT = ROOT / "data/processed/uji_group_split_v1"
BUILDINGS = (0, 1, 2)
SEEDS = (42, 43, 44)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def building_mask(split, building):
    labels = np.asarray([c["building"] for c in split.info.classes])
    return labels[split.class_ids] == building


def partition(split, building, info):
    """Filter a frozen split without changing source identities or sample order."""
    mask = building_mask(split, building)
    if not mask.any():
        raise ValueError(f"No observations for building {building}, split {split.name}.")
    pairs = {(c["building"], c["floor"]): c["class_id"] for c in info.classes}
    original = split.info.classes
    labels = np.asarray([pairs[(original[c]["building"], original[c]["floor"])]
                         for c in split.class_ids[mask]], dtype=np.int64)
    return FingerprintSplit(split.name,
        FingerprintInputs(split.inputs.rssi[mask], split.inputs.row_indices[mask],
                          split.inputs.source_id, split.inputs.ap_order),
        labels, split.xy[mask], {k: v[mask] for k, v in split.metadata.items()}, info)


def fit_coordinates(train_xy):
    """Exactly the original shared-scalar formula, now fit within one building."""
    return {"center": train_xy.mean(0).tolist(),
            "shared_scale": max(float(np.sqrt(train_xy.var(0, ddof=0).mean())), 1e-6)}


def prepare(output):
    if output.exists():
        raise FileExistsError("Choose a new immutable experiment directory.")
    parent = load_bundle(PARENT)
    base = read(SOURCE_TRAINING_CONFIG)
    for key in ("mixer", "kan", "loss", "training", "positioning"):
        assert base[key] == read(ROOT / "src/ips/algorithms/kan/configs/accuracy_spatial.json")[key]
    assert base["positioning"]["variant"] == "reference"
    settings = {}
    for seed in SEEDS:
        calibration = ROOT / SOURCE_RUNS[seed] / "calibration.json"
        source = read(calibration)
        checkpoint = calibration.parent / "pure_selected.pt"
        settings[str(seed)] = {
            "decoder": {k: source["best_pure"][k] for k in ("top_k", "temperature")},
            "calibration": calibration.relative_to(ROOT).as_posix(),
            "calibration_sha256": sha256(calibration),
            "joint_checkpoint": checkpoint.relative_to(ROOT).as_posix(),
            "joint_checkpoint_sha256": sha256(checkpoint)}
    output.mkdir(parents=True)
    (output / "code_snapshot").mkdir()
    (output / "code_snapshot/uji_buildings.py").write_bytes(Path(__file__).read_bytes())
    save_json(output / "training_config.json", base)
    frozen = {name: parent.split(name) for name in ("train", "validation")}
    for building in BUILDINGS:
        directory = output / f"building{building}/split"
        directory.mkdir(parents=True)
        mask = building_mask(frozen["train"], building)
        classes = [{**c, "class_id": i} for i, c in enumerate(
            c for c in parent.classes if c["building"] == building)]
        preprocessing = copy.deepcopy(parent.preprocessing)
        preprocessing["coordinates"].update(fit_coordinates(frozen["train"].xy[mask]))
        preprocessing["coordinates"].update(fit_split="train", fit_scope="building_train_only",
            fit_rows=int(mask.sum()), fit_indices_sha256=None)
        counts = {}
        for split, data in frozen.items():
            selected = data.inputs.row_indices[building_mask(data, building)]
            (directory / f"{split}_indices.csv").write_bytes(csv_bytes(["row_index"], [[int(i)] for i in selected]))
            counts[split] = len(selected)
        preprocessing["coordinates"]["fit_indices_sha256"] = sha256(directory / "train_indices.csv")
        for name, value in [("preprocessing.json", preprocessing), ("classes.json", classes),
                            ("ap_order.json", parent.ap_order)]:
            save_json(directory / name, value)
        manifest = {"protocol": "uji_known_building_experts_v1", "building": building,
            "parent_split_dir": PARENT.relative_to(ROOT).as_posix(),
            "parent_manifest_sha256": sha256(PARENT / "manifest.json"),
            "sources": parent.manifest["sources"], "counts": counts,
            "classes": classes, "known_building_required": True,
            "split_policy": "filter original frozen train/validation by building; no re-split",
            "coordinate_fit": "separate building train mean and shared scale; same normalized sigma 0.05",
            "test_used_for_training_or_selection": False,
            "controller_sha256": sha256(Path(__file__)),
            "artifact_sha256": {name: sha256(directory / name) for name in
                ["train_indices.csv", "validation_indices.csv", "preprocessing.json", "classes.json", "ap_order.json"]}}
        save_json(directory / "manifest.json", manifest)
        print(f"PREPARED building={building} {counts} floors={len(classes)} scale={preprocessing['coordinates']['shared_scale']:.6f}", flush=True)
    save_json(output / "protocol.json", {"status": "frozen_before_new_training", "buildings": list(BUILDINGS),
        "seeds": list(SEEDS), "max_epochs": base["training"]["max_epochs"],
        "variant": "pure_reference_KAN_Mixer", "known_building_required": True,
        "parent_manifest_sha256": sha256(PARENT / "manifest.json"),
        "training_config_source": SOURCE_TRAINING_CONFIG.relative_to(ROOT).as_posix(),
        "training_config_source_sha256": sha256(SOURCE_TRAINING_CONFIG),
        "training_config_sha256": sha256(output / "training_config.json"),
        "inference_by_seed": settings,
        "selection": "same validation early-stopping; training decoder k5/T1; final decoder inherited per seed from UJI validation",
        "comparators": ["separate_known_building", "joint_original", "joint_known_building"],
        "primary_aggregation": "per seed arithmetic mean of three building mean2D, then mean/sampleSD over 3 seeds",
        "secondary_aggregation": "pooled sample-weighted mean2D on the same official test rows",
        "test_history": "official UJI test already inspected in earlier experiments; this is an exploratory follow-up, not a new blind benchmark"})


def load_prepared(output):
    protocol = read(output / "protocol.json")
    parent = load_bundle(PARENT)
    assert sha256(PARENT / "manifest.json") == protocol["parent_manifest_sha256"]
    assert sha256(output / "training_config.json") == protocol["training_config_sha256"]
    train_split, val_split = parent.split("train"), parent.split("validation")
    result = {}
    for building in BUILDINGS:
        directory = output / f"building{building}/split"
        manifest = read(directory / "manifest.json")
        for name, value in manifest["artifact_sha256"].items():
            if sha256(directory / name) != value:
                raise ValueError(f"Frozen artifact changed: {directory / name}")
        info = DatasetInfo(directory, manifest, read(directory / "preprocessing.json"),
                           read(directory / "classes.json"), read(directory / "ap_order.json"))
        parts = {s.name: partition(s, building, info) for s in (train_split, val_split)}
        for name, split in parts.items():
            with (directory / f"{name}_indices.csv").open(newline="") as stream:
                indices = [int(r["row_index"]) for r in csv.DictReader(stream)]
            np.testing.assert_array_equal(indices, split.inputs.row_indices)
        fitted = fit_coordinates(parts["train"].xy)
        for key in fitted:
            np.testing.assert_allclose(info.preprocessing["coordinates"][key], fitted[key], rtol=0, atol=1e-12)
        result[building] = (info, {name: FingerprintDataset.from_split(split) for name, split in parts.items()})
    for name in ("train", "validation"):
        original = parent.indices[name]
        combined = np.concatenate([result[b][1][name].indices.numpy() for b in BUILDINGS])
        np.testing.assert_array_equal(np.sort(combined), original)
        assert len(np.unique(combined)) == len(original)
    return parent, result


def small_dataset(data):
    selected = np.sort(np.concatenate([np.flatnonzero(data.classes.numpy() == c)[:16]
                                      for c in np.unique(data.classes.numpy())]))
    result = copy.copy(data)
    for name in ("features", "classes", "xy", "xy_original", "indices"):
        setattr(result, name, getattr(data, name)[selected])
    result.metadata = {name: values[selected] for name, values in data.metadata.items()}
    return result


def train_all(output, device_name, smoke=False):
    _, data = load_prepared(output)
    protocol = read(output / "protocol.json")
    for seed in SEEDS:
        for building in BUILDINGS:
            info, datasets = data[building]
            destination = output / f"building{building}/seed{seed}" / ("smoke" if smoke else "training")
            if (destination / "summary.json").exists():
                summary = read(destination / "summary.json")
                if summary["status"] == "completed":
                    print(f"SKIP completed building={building} seed={seed} smoke={smoke}", flush=True)
                    continue
            config = read(output / "training_config.json")
            config["split_dir"] = info.directory.relative_to(ROOT).as_posix()
            active = {name: small_dataset(ds) for name, ds in datasets.items()} if smoke else datasets
            print(f"BEGIN building={building} seed={seed} mode={'smoke' if smoke else 'full'} device={device_name}", flush=True)
            summary = train(config, destination, seed, device_name, smoke=smoke, bundle=info,
                            train_data=active["train"], validation_data=active["validation"])
            if not smoke:
                payload = torch.load(destination / "best.pt", map_location="cpu", weights_only=True)
                settings = protocol["inference_by_seed"][str(seed)]
                payload["config"]["positioning"].update(settings["decoder"])
                payload["metadata"]["building_expert"] = {"building": building, "known_building_required": True,
                    "parent_manifest_sha256": protocol["parent_manifest_sha256"], "decoder_provenance": settings,
                    "validation_score_note": "checkpoint selection score uses k5/T1; report final frozen decoder separately"}
                frozen = destination.parent / "frozen"
                frozen.mkdir(exist_ok=False)
                torch.save(payload, frozen / "pure.pt")
                save_json(frozen / "freeze.json", {"source_best_sha256": sha256(destination / "best.pt"),
                    "pure_sha256": sha256(frozen / "pure.pt"), "seed": seed, "building": building,
                    "decoder": settings["decoder"], "test_used_for_selection": False})
            print(f"DONE building={building} seed={seed} epochs={summary['epochs_completed']} best={summary['best_epoch']} val_selection={summary['best_validation_mean_2d']:.4f} seconds={summary['training_seconds']:.1f}", flush=True)
            if device_name.startswith("cuda"):
                torch.cuda.empty_cache()
    if not smoke:
        entries = []
        for seed in SEEDS:
            for building in BUILDINGS:
                path = output / f"building{building}/seed{seed}/frozen/pure.pt"
                frozen = read(path.parent / "freeze.json")
                assert sha256(path) == frozen["pure_sha256"]
                entries.append({"seed": seed, "building": building,
                                "path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path)})
        save_json(output / "training_completed.json", {"status": "completed", "models": entries, "test_evaluated": False})


@torch.inference_mode()
def predict_batches(checkpoint_path, datasets, device, *, known_building=False):
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    assert "fingerprint_refinement" not in checkpoint and "ensemble_members" not in checkpoint
    memory = MemorySampler().start()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    results = {}
    for name, data in datasets.items():
        xy, labels, oracle_xy, oracle_labels = [], [], [], []
        synchronize(device)
        started = time.perf_counter()
        class_buildings = torch.tensor([c["building"] for c in checkpoint["classes"]], device=device)
        for start in range(0, len(data), 128):
            features = data.features[start:start+128].to(device)
            output = model(features)
            selected = output["logits"].argmax(1)
            labels.append(selected.cpu())
            xy.append(output["xy"].cpu())
            if known_building:
                # This branch uses only the building to route; no query XY/floor.
                truth = data.classes[start:start+128].to(device)
                buildings = class_buildings[truth]
                scores = output["logits"].masked_fill(class_buildings[None] != buildings[:, None], -torch.inf)
                chosen = scores.argmax(1)
                oracle_labels.append(chosen.cpu())
                oracle_xy.append(output["conditional_xy"][torch.arange(len(features), device=device), chosen].cpu())
        synchronize(device)
        seconds = time.perf_counter() - started
        results[name] = {"indices": data.indices.numpy(), "classes": torch.cat(labels).numpy(),
                        "xy": inverse_xy(torch.cat(xy).numpy(), checkpoint["preprocessing"]),
                        "inference_seconds": seconds}
        if known_building:
            results[name]["known_classes"] = torch.cat(oracle_labels).numpy()
            results[name]["known_xy"] = inverse_xy(torch.cat(oracle_xy).numpy(), checkpoint["preprocessing"])
    resources = {"process_ram": memory.stop(), "cuda": cuda_memory(device), "batch_size": 128,
                 "timing_scope": "forward and transfers; joint known-building branch shares forward with original"}
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return results, resources


def export_prediction(destination, truth, predicted_class, predicted_xy, global_classes, metadata):
    destination.mkdir(parents=True, exist_ok=False)
    metrics = evaluate_predictions(truth.class_ids, predicted_class, truth.xy, predicted_xy, global_classes)
    pair = np.asarray([[c["building"], c["floor"]] for c in global_classes])
    true_pair, predicted_pair = pair[truth.class_ids], pair[predicted_class]
    errors = np.linalg.norm(predicted_xy - truth.xy, axis=1)
    per_building = {}
    for building in BUILDINGS:
        mask = true_pair[:, 0] == building
        per_building[str(building)] = evaluate_predictions(truth.class_ids[mask], predicted_class[mask],
            truth.xy[mask], predicted_xy[mask], global_classes)
    metrics["per_building"] = per_building
    metrics["macro_building_mean_2d"] = float(np.mean([m["mean_2d"] for m in per_building.values()]))
    metrics.update(metadata)
    save_json(destination / "metrics.json", metrics)
    rows = [[int(index), int(index)+2, int(t), int(p), int(tb), int(pb), int(tf), int(pf),
             *target.tolist(), *prediction.tolist(), float(error),
             int(truth.metadata["USERID"][i]), int(truth.metadata["PHONEID"][i])]
            for i, (index, t, p, (tb, tf), (pb, pf), target, prediction, error) in enumerate(zip(
                truth.inputs.row_indices, truth.class_ids, predicted_class, true_pair, predicted_pair,
                truth.xy, predicted_xy, errors))]
    (destination / "predictions.csv").write_bytes(csv_bytes([
        "row_index", "csv_line", "true_class", "predicted_class", "true_building", "predicted_building",
        "true_floor", "predicted_floor", "true_x", "true_y", "predicted_x", "predicted_y", "error_2d", "USERID", "PHONEID"], rows))
    for task in ("joint", "building", "floor"):
        np.savetxt(destination / f"confusion_{task}.csv", metrics[f"{task}_confusion_matrix"], fmt="%d", delimiter=",")
    print(f"EVAL {metadata['method']} seed={metadata['seed']} split={truth.name} macro_building={metrics['macro_building_mean_2d']:.4f} pooled={metrics['mean_2d']:.4f}", flush=True)
    return metrics


def evaluate_all(output, device_name):
    parent, parts = load_prepared(output)
    protocol, completed = read(output / "protocol.json"), read(output / "training_completed.json")
    if (output / "pre_test_lock.json").exists():
        raise FileExistsError("Evaluation already locked. Do not overwrite results.")
    locks = completed["models"]
    for entry in locks:
        assert sha256(ROOT / entry["path"]) == entry["sha256"]
    for setting in protocol["inference_by_seed"].values():
        assert sha256(ROOT / setting["joint_checkpoint"]) == setting["joint_checkpoint_sha256"]
    save_json(output / "pre_test_lock.json", {"experts": locks, "joint_checkpoints": protocol["inference_by_seed"],
        "all_9_models_frozen": True, "test_used_for_selection": False,
        "history_note": protocol["test_history"]})
    raw = {name: parent.split(name) for name in ("train", "validation", "test")}
    whole = {name: FingerprintDataset.from_split(split) for name, split in raw.items()}
    for building, (info, datasets) in parts.items():
        datasets["test"] = FingerprintDataset.from_split(partition(raw["test"], building, info))
    global_map = {(c["building"], c["floor"]): c["class_id"] for c in parent.classes}
    device = configure_runtime(42, device_name, 2)
    resources = []
    for seed in SEEDS:
        combined = {name: [] for name in raw}
        for building in BUILDINGS:
            info, datasets = parts[building]
            checkpoint = output / f"building{building}/seed{seed}/frozen/pure.pt"
            result, memory = predict_batches(checkpoint, datasets, device)
            resources.append({"seed": seed, "building": building, "method": "separate_known_building", **memory})
            mapping = np.asarray([global_map[(c["building"], c["floor"])] for c in info.classes])
            for split, prediction in result.items():
                combined[split].append({**prediction, "classes": mapping[prediction["classes"]]})
        for split, predictions in combined.items():
            indices = np.concatenate([p["indices"] for p in predictions])
            order = np.argsort(indices)
            np.testing.assert_array_equal(indices[order], raw[split].inputs.row_indices)
            classes = np.concatenate([p["classes"] for p in predictions])[order]
            xy = np.concatenate([p["xy"] for p in predictions])[order]
            assert all(parent.classes[int(p)]["building"] == parent.classes[int(t)]["building"]
                       for p, t in zip(classes, raw[split].class_ids))
            export_prediction(output / f"evaluation/seed{seed}/separate_known_building/{split}", raw[split], classes, xy,
                parent.classes, {"seed": seed, "method": "separate_known_building", "known_building_required": True,
                    "inference_seconds": sum(p["inference_seconds"] for p in predictions)})
        source = ROOT / protocol["inference_by_seed"][str(seed)]["joint_checkpoint"]
        result, memory = predict_batches(source, whole, device, known_building=True)
        resources.append({"seed": seed, "method": "joint_original_and_known_building", **memory})
        for split, prediction in result.items():
            for method, label_key, xy_key in [("joint_original", "classes", "xy"),
                                               ("joint_known_building", "known_classes", "known_xy")]:
                export_prediction(output / f"evaluation/seed{seed}/{method}/{split}", raw[split],
                    prediction[label_key], prediction[xy_key], parent.classes,
                    {"seed": seed, "method": method, "known_building_required": method != "joint_original",
                     "checkpoint_sha256": sha256(source), "inference_seconds_shared_forward": prediction["inference_seconds"]})
    for entry in locks:
        assert sha256(ROOT / entry["path"]) == entry["sha256"]
    save_json(output / "evaluation_resources.json", resources)
    save_json(output / "completed.json", {"status": "completed", "seeds": list(SEEDS), "buildings": list(BUILDINGS),
        "methods": protocol["comparators"], "test_predictions_only_after_all_9_models_frozen": True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "train", "evaluate"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if args.stage == "prepare":
        prepare(output)
    elif args.stage == "train":
        train_all(output, args.device, args.smoke)
    elif args.smoke:
        raise ValueError("Smoke runs are not test experiments.")
    else:
        evaluate_all(output, args.device)


if __name__ == "__main__":
    main()
