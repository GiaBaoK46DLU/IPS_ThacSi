"""Frozen UJI settings transferred to supplied DLULIB, HCXY, TUJI1 and SYL.

Prepare -> train (train/validation only) -> lock -> evaluate. No calibration on
the new test set. Real APs retain order and are padded to 520 with absent masks.
The original UJI model, loss, optimizer, decoder and hybrid code are reused.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ...data import ROOT, csv_bytes, feature_transform, json_bytes, sha256
from ...data_access.supplied import group_split, read_supplied
from ...metrics import evaluate_predictions, inverse_xy
from .models.refinement import db_features
from .resources import MemorySampler, cuda_memory, synchronize
from .runtime import configure_runtime, load_checkpoint, save_json
from .train import train

SOURCE_RUNS = {
    42: "runs/kan_accuracy_20261009/spatial_seed42_calibration",
    43: "runs/kan_accuracy_20261009/selected_seed43_calibration",
    44: "runs/kan_seed44_complete_20261009_v1/calibration",
}
SOURCE_TRAINING_CONFIG = ROOT / "runs/kan_seed44_complete_20261009_v1/training_config.json"


class PaddedDataset(Dataset):
    def __init__(self, raw, indices, preprocessing, classes, split_name):
        self.split_name = split_name
        self.indices = torch.as_tensor(np.asarray(indices).copy(), dtype=torch.long)
        values = raw.rssi[indices]
        if values.shape[1] > 520:
            raise ValueError("Frozen 520 AP architecture cannot accept more than 520 real APs.")
        padded = np.full((len(indices), 520), 100, dtype=np.float32)
        padded[:, :values.shape[1]] = np.where(values == raw.missing_code, 100, values)
        self.features = torch.from_numpy(feature_transform(padded))
        self.xy_original = raw.xy[indices].copy()
        coord = preprocessing["coordinates"]
        self.xy = torch.from_numpy(((self.xy_original - coord["center"]) / coord["shared_scale"]).astype(np.float32))
        mapping = {(c["building"], c["floor"]): c["class_id"] for c in classes}
        self.classes = torch.as_tensor([mapping[(int(raw.buildings[i]), int(raw.floors[i]))] for i in indices])
        self.source_files, self.source_rows = raw.source_files[indices], raw.source_rows[indices]
        self.rp = raw.rp[indices]
        self.metadata = {key: values[indices] for key, values in raw.metadata.items()}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.features[index], self.classes[index], self.xy[index]


@dataclass
class ExperimentBundle:
    directory: Path
    manifest: dict
    preprocessing: dict
    classes: list
    ap_order: list


def source_settings(seed):
    if seed not in SOURCE_RUNS:
        raise ValueError("Only seeds 42/43/44 have verified prior UJI settings.")
    calibration_path = ROOT / SOURCE_RUNS[seed] / "calibration.json"
    source = json.loads(calibration_path.read_text(encoding="utf-8"))
    config = json.loads(SOURCE_TRAINING_CONFIG.read_text(encoding="utf-8"))
    pure, selected = source["best_pure"], source["selected"]
    if selected["kind"] != "kan_mixer_fingerprint_hybrid":
        raise ValueError("Source did not select a hybrid.")
    decoder = {k: pure[k] for k in ("top_k", "temperature")}
    hybrid = {k: selected[k] for k in ("neighbors", "distance_power", "neighbor_class_weight", "neural_coordinate_weight")}
    if any(selected[k] != decoder[k] for k in decoder):
        raise ValueError("Pure and hybrid source decoders differ.")
    evidence = {"training_config_path": SOURCE_TRAINING_CONFIG.relative_to(ROOT).as_posix(),
                "training_config_sha256": sha256(SOURCE_TRAINING_CONFIG),
                "calibration_path": calibration_path.relative_to(ROOT).as_posix(),
                "calibration_sha256": sha256(calibration_path), "pure_decoder": decoder, "hybrid": hybrid,
                "new_dataset_calibration": False}
    return config, decoder, hybrid, evidence


def write_indices(path, raw, indices):
    path.write_bytes(csv_bytes(["development_index", "source_file", "source_row", "rp", "building", "floor", "x", "y"],
        [(int(i), raw.source_files[i], int(raw.source_rows[i]), raw.rp[i], int(raw.buildings[i]),
          int(raw.floors[i]), *raw.xy[i]) for i in indices]))


def prepare_dataset(name, output, seeds):
    directory = output / name / "split"
    if directory.exists():
        raise FileExistsError("Choose a new experiment directory; split files are immutable.")
    raw = read_supplied(name, "train")
    train_indices, validation_indices = group_split(raw)
    pairs = sorted(set(zip(raw.buildings.tolist(), raw.floors.tolist())))
    classes = [{"class_id": i, "building": int(b), "floor": int(f)} for i, (b, f) in enumerate(pairs)]
    xy = raw.xy[train_indices]
    preprocessing = {"coordinates": {"center": xy.mean(0).tolist(),
        "shared_scale": max(float(np.sqrt(xy.var(0, ddof=0).mean())), 1e-6),
        "fit_split": "train", "fit_rows": len(train_indices), "order": ["x", "y"],
        "units": ("metres_assumed_pending_owner_confirmation" if name == "dlulib" else
                  "metres_local_TUJI_coordinates" if name == "tuji" else "metres_local_SOD_coordinates")},
        "rssi": {"source_missing_code": raw.missing_code, "internal_missing_code": 100,
                 "detected_transform": "(rssi+104)/104", "mask": "1 iff detected",
                 "real_ap_count": len(raw.ap_order), "padded_ap_count": 520-len(raw.ap_order),
                 "padding": "trailing absent APs with (z,mask)=(0,0); no invented observations"}}
    directory.mkdir(parents=True)
    write_indices(directory / "train_indices.csv", raw, train_indices)
    write_indices(directory / "validation_indices.csv", raw, validation_indices)
    ap_order = raw.ap_order + [f"__PAD_{i}" for i in range(len(raw.ap_order)+1, 521)]
    manifest = {"dataset": name, "split_seed": 2026, "validation_group_fraction": 0.2,
        "group_columns": ["building", "floor", "x", "y"], "counts": {"development": len(raw),
        "train": len(train_indices), "validation": len(validation_indices)},
        "train_validation_position_overlap": 0, "development_sources": raw.sources,
        "assumptions": raw.assumptions, "real_ap_order": raw.ap_order, "padded_ap_order": ap_order,
        "classes": classes, "preprocessing": preprocessing,
        "train_indices_sha256": sha256(directory / "train_indices.csv"),
        "validation_indices_sha256": sha256(directory / "validation_indices.csv"),
        "test_opened_for_training": False, "mean_files_used": False,
        "protocol": "official test retained; development train/validation held out by position",
        "source_code_sha256": {str(p.relative_to(ROOT).as_posix()): sha256(p) for p in
            (Path(__file__), ROOT / "src/ips/data_access/supplied.py")}}
    save_json(directory / "manifest.json", manifest)
    configs = output / name / "configs"; configs.mkdir()
    for seed in seeds:
        config, decoder, hybrid, evidence = source_settings(seed)
        # This path is not a UJI DatasetRepository input. train() receives the
        # explicitly supplied data/bundle, preserving the legacy train loop.
        config["split_dir"] = directory.relative_to(ROOT).as_posix()
        save_json(configs / f"training_seed{seed}.json", config)
        save_json(configs / f"inference_seed{seed}.json", evidence)
    print(f"PREPARED {name}: {manifest['counts']} real_AP={len(raw.ap_order)} padding={520-len(raw.ap_order)}", flush=True)


def load_prepared(name, output):
    directory = output / name / "split"
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    raw = read_supplied(name, "train")
    if raw.sources != manifest["development_sources"]:
        raise ValueError("Development source hashes changed.")
    indices = {}
    for split in ("train", "validation"):
        path = directory / f"{split}_indices.csv"
        if sha256(path) != manifest[f"{split}_indices_sha256"]:
            raise ValueError("Frozen split changed.")
        with path.open(encoding="utf-8", newline="") as stream:
            indices[split] = np.asarray([int(r["development_index"]) for r in csv.DictReader(stream)])
    bundle = ExperimentBundle(directory, manifest, manifest["preprocessing"], manifest["classes"], manifest["padded_ap_order"])
    datasets = {split: PaddedDataset(raw, idx, bundle.preprocessing, bundle.classes, split) for split, idx in indices.items()}
    return bundle, datasets


def train_dataset(name, output, seeds, device_name, smoke=False):
    # sklearn 1.6 emits this even with an explicit one-label list. The
    # descriptive 1x1 matrix is expected here; other warnings remain visible.
    warnings.filterwarnings("ignore", message="A single label was found", module="sklearn.metrics._classification")
    bundle, datasets = load_prepared(name, output)
    for seed in seeds:
        base = output / name / f"seed{seed}"
        config = json.loads((output / name / "configs" / f"training_seed{seed}.json").read_text())
        config_original, _, _, _ = source_settings(seed)
        assert all(config[k] == config_original[k] for k in ("mixer", "kan", "loss", "training", "positioning"))
        if smoke:
            # Representative per-class first samples, without modifying the
            # full split or training config. Smoke results never enter reports.
            small = {}
            for split, data in datasets.items():
                quota = 256 if split == "train" else 128
                rows = np.concatenate([np.flatnonzero(data.classes.numpy() == c)[:max(1,quota//len(bundle.classes))]
                                       for c in range(len(bundle.classes))])
                small[split] = torch.utils.data.Subset(data, sorted(rows.tolist()))
                for attr in ("features", "classes", "xy", "xy_original", "indices"):
                    setattr(small[split], attr, getattr(data, attr)[rows])
                small[split].split_name = split
            active = small
        else:
            active = datasets
        print(f"TRAIN {name} seed={seed} max_epochs={config['training']['max_epochs']} device={device_name}", flush=True)
        train(config, base / ("smoke" if smoke else "training"), seed, device_name, smoke=smoke,
              bundle=bundle, train_data=active["train"], validation_data=active["validation"])
        if not smoke:
            # Legacy metric code has a UJI-specific default units label. Only
            # reporting metadata is replaced; all computed values stay intact.
            metric_path = base / "training/best_validation_metrics.json"
            best_metrics = json.loads(metric_path.read_text(encoding="utf-8"))
            best_metrics["coordinate_units"] = bundle.preprocessing["coordinates"]["units"]
            save_json(metric_path, best_metrics)
            freeze_checkpoints(name, output, seed, bundle, datasets["train"])


def freeze_checkpoints(name, output, seed, bundle, training):
    base = output / name / f"seed{seed}"; destination = base / "frozen"
    destination.mkdir(exist_ok=False)
    payload = torch.load(base / "training/best.pt", map_location="cpu", weights_only=True)
    _, decoder, hybrid, evidence = source_settings(seed)
    payload["config"]["positioning"].update(decoder)
    payload["metadata"]["frozen_UJI_transfer"] = evidence
    payload["metadata"]["supplied_dataset"] = name
    payload["metadata"]["evaluation_entrypoint"] = "ips.algorithms.kan.supplied_experiment"
    pure = destination / "pure.pt"; torch.save(payload, pure)
    payload = copy.deepcopy(payload)
    payload["format_version"] = 2
    payload["fingerprint_refinement"] = {"signal": db_features(training.features), "xy": training.xy,
        "class_ids": training.classes, "row_indices": training.indices, "fit_scope": "internal_train_only",
        "train_indices_sha256": sha256(bundle.directory / "train_indices.csv"), "settings": hybrid}
    torch.save(payload, destination / "hybrid.pt")
    save_json(destination / "freeze.json", {"source_best_sha256": sha256(base / "training/best.pt"),
        "pure_sha256": sha256(pure), "hybrid_sha256": sha256(destination / "hybrid.pt"),
        "source_inference_settings": evidence, "test_evaluated": False,
        "train_catalogue_size": len(training), "neural_weights_identical": True})


def measure(data, predicted_classes, predicted_xy, bundle):
    result = evaluate_predictions(data.classes.numpy(), predicted_classes, data.xy_original, predicted_xy, bundle.classes)
    result["coordinate_units"] = bundle.preprocessing["coordinates"]["units"]
    result["classification_applicability"] = {"building": False, "floor": len(bundle.classes) > 1,
        "joint": len(bundle.classes) > 1}
    result["note_constant_labels"] = "Constant/site namespace classification is descriptive only, not a learned localization task."
    return result


@torch.inference_mode()
def evaluate_checkpoint(path, data, bundle, destination, device):
    destination.mkdir(parents=True, exist_ok=False)
    memory = MemorySampler().start()
    model, checkpoint = load_checkpoint(path, device)
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
    synchronize(device); started = time.perf_counter(); outputs = []; labels = []
    for batch in data.features.split(128):
        output = model(batch.to(device))
        outputs.append(output["xy"].cpu()); labels.append(output["logits"].argmax(1).cpu())
    synchronize(device); seconds = time.perf_counter() - started
    predicted_class = torch.cat(labels).numpy()
    predicted_xy = inverse_xy(torch.cat(outputs).numpy(), bundle.preprocessing)
    metrics = measure(data, predicted_class, predicted_xy, bundle)
    errors = np.linalg.norm(predicted_xy-data.xy_original, axis=1)
    true_class = data.classes.numpy()
    pair = np.asarray([[c["building"],c["floor"]] for c in bundle.classes])
    with (destination / "predictions.csv").open("w",encoding="utf-8",newline="") as stream:
        writer = csv.writer(stream); writer.writerow(["source_file","source_row","split_row_index","rp",
            "true_class","predicted_class","true_floor","predicted_floor","true_x","true_y","predicted_x","predicted_y","error_2d", *data.metadata])
        writer.writerows([(str(data.source_files[i]),int(data.source_rows[i]),int(data.indices[i]),float(data.rp[i]),
            int(true_class[i]),int(predicted_class[i]),int(pair[true_class[i],1]),int(pair[predicted_class[i],1]),
            *data.xy_original[i],*predicted_xy[i],float(errors[i]), *(values[i] for values in data.metadata.values())) for i in range(len(data))])
    save_json(destination / "metrics.json",metrics)
    for task in ("joint","floor","building"):
        np.savetxt(destination / f"confusion_{task}.csv",metrics[f"{task}_confusion_matrix"],fmt="%d",delimiter=",")
    resources={"inference_seconds":seconds,"batch_size":128,"samples":len(data),
        "milliseconds_per_sample_amortized":seconds*1000/len(data),"checkpoint_sha256":sha256(path),
        "cuda":cuda_memory(device),"process_ram":memory.stop(),
        "timing_scope":"batched forward including H2D/D2H; excludes loading, metrics and file exports",
        "hybrid_train_query_self_included":bool(data.split_name=="train" and "fingerprint_refinement" in checkpoint)}
    save_json(destination / "resources.json",resources)
    print(f"EVAL {bundle.manifest['dataset']} {path.name} {data.split_name}: n={len(data)} mean2d={metrics['mean_2d']:.4f} median={metrics['median_2d']:.4f} floor_acc={metrics['floor_accuracy']:.4f}",flush=True)
    del model
    if device.type=="cuda":torch.cuda.empty_cache()


def evaluate_dataset(name,output,seeds,device_name):
    warnings.filterwarnings("ignore", message="A single label was found", module="sklearn.metrics._classification")
    bundle,datasets=load_prepared(name,output)
    locks=[]
    for seed in seeds:
        base=output/name/f"seed{seed}"
        summary=json.loads((base/"training/summary.json").read_text())
        if summary["run_type"]!="training" or summary["test_evaluated"]:
            raise ValueError("Requires completed training with no test selection.")
        freeze=json.loads((base/"frozen/freeze.json").read_text())
        for variant in ("pure","hybrid"):
            path=base/f"frozen/{variant}.pt"
            if sha256(path)!=freeze[f"{variant}_sha256"]:raise ValueError("Frozen checkpoint changed.")
            locks.append({"seed":seed,"variant":variant,"path":path.relative_to(ROOT).as_posix(),"sha256":sha256(path)})
    lock=output/name/"pre_test_lock.json"
    if lock.exists():raise FileExistsError("Test evaluation already locked; choose new explicit protocol if repeating.")
    save_json(lock,{"checkpoints":locks,"new_test_used_for_selection":False,"selection_scope":"internal_validation_for_epoch_only; decoder/hybrid copied from prior UJI validation"})
    # Produce test predictions only after every checkpoint is frozen. Source
    # quality audits and the shared HCXY map may already expose test metadata;
    # none of it participates in fitting/selection.
    raw_test=read_supplied(name,"test")
    if raw_test.ap_order!=bundle.manifest["real_ap_order"]:raise ValueError("Train/test AP order mismatch.")
    datasets["test"]=PaddedDataset(raw_test,np.arange(len(raw_test)),bundle.preprocessing,bundle.classes,"test")
    save_json(output/name/"test_source_manifest.json",{"sources":raw_test.sources,"samples":len(raw_test),"pre_test_lock_sha256":sha256(lock)})
    device=configure_runtime(42,device_name,2)
    for entry in locks:
        for split,data in datasets.items():
            evaluate_checkpoint(ROOT/entry["path"],data,bundle,output/name/f"seed{entry['seed']}/evaluation/{entry['variant']}/{split}",device)
    save_json(output/name/"completed.json",{"dataset":name,"seeds":seeds,"variants":["pure","hybrid"],"splits":list(datasets),"test_used_for_selection":False})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=["prepare","train","evaluate"])
    parser.add_argument("--datasets",nargs="+",choices=["dlulib","hcxy","tuji","syl"],default=["dlulib","hcxy"])
    parser.add_argument("--seeds",nargs="+",type=int,default=[42,43,44])
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--device",default="cuda")
    parser.add_argument("--smoke",action="store_true")
    args=parser.parse_args();output=args.output_dir.resolve()
    for name in args.datasets:
        if args.stage=="prepare":prepare_dataset(name,output,args.seeds)
        elif args.stage=="train":train_dataset(name,output,args.seeds,args.device,args.smoke)
        else:
            if args.smoke:raise ValueError("Smoke results cannot be evaluated as research.")
            evaluate_dataset(name,output,args.seeds,args.device)


if __name__=="__main__":main()
