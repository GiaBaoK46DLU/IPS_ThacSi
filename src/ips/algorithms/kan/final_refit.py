"""Refit on the full development CSV AFTER validation-based selection.

No validation/test metrics are computed in this training routine. Epoch count
and inference settings must come from a frozen selection JSON, not the test.
The historical split files and coordinate transform are never overwritten.
"""
import argparse
import copy
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from ...data import ROOT, sha256
from ...data_access import load_bundle
from .dataset import FingerprintDataset
from .models import build_model, parameter_count
from .models.reference import fit_reference_points
from .runtime import configure_runtime, load_checkpoint, save_json, code_hashes, software_info
from .resources import MemorySampler, cuda_memory, synchronize
from .train import combined_loss, reference_consistency_loss, update_ema


def recover_training_config(source, selection, source_hash):
    """Require original training evidence; calibrated inference T/k are not it."""
    metadata = source.get("metadata", {})
    declared = selection.get("training_configs_by_source_sha256", {}).get(source_hash)
    stored = metadata.get("training_config")
    if declared is not None:
        path = Path(declared["path"])
        if not path.is_absolute():
            path = ROOT/path
        if sha256(path) != declared["sha256"]:
            raise ValueError("Original training configuration hash differs from the frozen selection.")
        original = json.loads(path.read_text(encoding="utf-8"))
        if stored is not None and original != stored:
            raise ValueError("Stored and declared original training configurations disagree.")
        evidence = {"kind":"hashed_original_json", "path":str(path), "sha256":declared["sha256"]}
    elif stored is not None:
        original = copy.deepcopy(stored)
        evidence = {"kind":"checkpoint_metadata.training_config", "checkpoint_sha256":source_hash}
    elif "validation_inference_calibration" in metadata:
        raise ValueError("Calibrated legacy checkpoint requires a hashed original training configuration; "
                         "metadata.config may alias calibrated inference settings.")
    else:
        original = copy.deepcopy(metadata.get("config", source["config"]))
        evidence = {"kind":"uncalibrated_checkpoint_config", "checkpoint_sha256":source_hash}
    # Only the decoder's inference choices may differ after calibration.
    selected_config, original_config = copy.deepcopy(source["config"]), copy.deepcopy(original)
    for config in (selected_config, original_config):
        for key in ("top_k", "temperature"):
            config["positioning"].pop(key, None)
    if selected_config != original_config:
        raise ValueError("Selected checkpoint and original training configuration differ beyond inference T/k.")
    return original, evidence


def validate_refit_bundle(source, bundle, selection):
    """Reject changed data/mappings before any post-selection catalogue fit."""
    expected = {'split_manifest_sha256':sha256(bundle.directory/'manifest.json'),
                'classes':bundle.classes,'preprocessing':bundle.preprocessing,'ap_order':bundle.ap_order}
    strict = selection.get('scope') == 'one_joint_kan_mixer'
    if ('split_manifest_sha256' in selection
            and selection['split_manifest_sha256'] != expected['split_manifest_sha256']):
        raise ValueError('Frozen selection and refit split manifest differ.')
    for key,value in expected.items():
        if key not in source:
            if strict:
                raise ValueError(f'Selected checkpoint is missing required refit provenance: {key}')
        elif source[key] != value:
            raise ValueError(f'Selected checkpoint and refit dataset differ: {key}')


def refit(source_checkpoint, selection_path, run_dir, seed, device_name="cuda"):
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("selection_split") != "internal_validation_only" or selection.get("test_evaluated") is not False:
        raise ValueError("Refit requires an explicit validation-only selection lock made before test evaluation.")
    epochs = selection["final_refit_epochs"]
    if type(epochs) is not int or not 1 <= epochs <= 100:
        raise ValueError("Frozen refit epoch count must be in 1..100.")
    source_hash = sha256(source_checkpoint)
    if source_hash not in selection["allowed_source_checkpoint_sha256"]:
        raise ValueError("Source checkpoint was not included in the frozen selection.")
    if run_dir.exists():
        raise FileExistsError(f"Run already exists: {run_dir}")
    for name,digest in selection.get('source_sha256',{}).items():
        if sha256(ROOT/name) != digest:
            raise ValueError(f'Refit code changed after selection was frozen: {name}')
    device = configure_runtime(seed, device_name, 2)
    # CPU load: use selected config only; refit network weights from a fresh seed.
    _, source = load_checkpoint(source_checkpoint, torch.device("cpu"))
    from .improvement import validate_kan_only
    validate_kan_only(source["config"], source)
    if source.get("metadata", {}).get("full_development_refit"):
        raise ValueError("Refit source must be the original validation-selected model, not another refit.")
    original_config, config_evidence = recover_training_config(source, selection, source_hash)
    config = copy.deepcopy(source["config"])
    original_positioning = original_config["positioning"]
    config["positioning"]["training_temperature"] = original_positioning.get(
        "training_temperature", original_positioning.get("temperature", 1.0))
    config["positioning"]["training_top_k"] = original_positioning.get(
        "training_top_k", original_positioning.get("top_k", 5))
    bundle = load_bundle(ROOT / config["split_dir"])
    validate_refit_bundle(source,bundle,selection)
    indices = np.arange(bundle.development.size, dtype=np.int64)
    dataset = FingerprintDataset(bundle.development, indices, bundle.preprocessing, bundle.classes)
    dataset.split_name = "development"
    fit_reference_points(config, dataset, final_refit=True)
    # Loading the source creates a temporary model and consumes RNG. Refit must
    # start from the declared seed, independently of that reconstruction.
    configure_runtime(seed, str(device), config["training"]["threads"])
    model = build_model(config, len(bundle.classes)).to(device)
    settings = config["training"]
    consistency_weight = settings.get("same_point_consistency_weight",0.)
    from .augmentation import training_representation
    loader_data = training_representation(dataset, settings, final_refit=True)
    loader = DataLoader(loader_data, batch_size=settings["batch_size"], shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(seed))
    ema_decay = settings.get("ema_decay", 0.0)
    if ema_decay and not 0 < ema_decay < 1:
        raise ValueError("EMA decay must be strictly between zero and one.")
    ema_model = copy.deepcopy(model).eval().requires_grad_(False) if ema_decay else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    # Keep the original validation trial's LR trajectory, rather than compress
    # a 100-epoch cosine schedule into the selected number of refit epochs.
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, settings["max_epochs"],
                    eta_min=settings.get("minimum_learning_rate", settings["learning_rate"] * 0.05))
                    if settings.get("scheduler") == "cosine" else None)
    run_dir.mkdir(parents=True)
    metadata = {"full_development_refit": True, "selection_split": "previous_internal_validation_only",
        "selection_lock_sha256": sha256(selection_path), "selection_lock": selection,
        "source_checkpoint_sha256": sha256(source_checkpoint), "actual_counts": {"train": len(dataset), "validation": 0},
        "fit_scope": "development_after_validation_selection", "test_evaluated_during_training": False,
        "split_manifest_sha256": sha256(bundle.directory / "manifest.json"), "code_sha256": code_hashes(),
        "software": software_info(device), "trainable_parameters": parameter_count(model),
        "original_preprocessing_retained": True,
        "source_training_config_evidence": config_evidence,
        "source_training_config": original_config,
        "selection_weights": "train_step_ema" if ema_model is not None else "live_model",
        "same_point_mixing_probability": settings.get("same_point_mixing", 0),
        "training_views_per_dataset_row": 2 if consistency_weight else 1,
        "rng_reset_after_source_loading": True,
        "source_validation_metric_is_not_a_score_for_this_refitted_model": True}
    save_json(run_dir / "run_metadata.json", metadata)
    save_json(run_dir / "resolved_config.json", config)
    memory = MemorySampler().start()
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    steps = 0
    try:
        with (run_dir / "history.csv").open("w", encoding="utf-8", newline="") as stream:
            fields = ["epoch", "train_loss", "train_class_loss", "train_xy_loss", "learning_rate", "epoch_seconds", "optimizer_steps"]
            if consistency_weight:
                fields.append('train_reference_consistency_loss')
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for epoch in range(1, epochs + 1):
                epoch_started = time.perf_counter()
                model.train()
                totals = np.zeros(3)
                samples,consistency_total = 0,0.
                for batch in loader:
                    x,y,xy = batch[:3]
                    pair_count = len(x)
                    if consistency_weight:
                        x = torch.cat([x,batch[3]])
                        y,xy = torch.cat([y,y]),torch.cat([xy,xy])
                    x, y, xy = x.to(device), y.to(device), xy.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    output = model(x)
                    loss, classification, coordinates = combined_loss(output, y, xy, config["loss"])
                    if consistency_weight:
                        consistency = reference_consistency_loss(output['reference_logits'],pair_count,
                            settings.get('consistency_temperature',1.))
                        loss = loss+consistency_weight*consistency
                        consistency_total += consistency.item()*pair_count
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite refit loss")
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip_l2"], error_if_nonfinite=True)
                    optimizer.step()
                    if ema_model is not None:
                        update_ema(ema_model, model, ema_decay, steps)
                    totals += np.array([loss.item(), classification.item(), coordinates.item()]) * len(x)
                    samples += len(x)
                    steps += 1
                synchronize(device)
                values = [epoch, *map(float, totals / samples),
                    optimizer.param_groups[0]["lr"], time.perf_counter() - epoch_started, steps]
                if consistency_weight:
                    values.append(consistency_total/len(dataset))
                row = dict(zip(fields,values))
                writer.writerow(row)
                stream.flush()
                print(f"kan_mixer FULL-DEVELOPMENT seed={seed} epoch={epoch}/{epochs} loss={row['train_loss']:.4f}; no validation/test evaluation", flush=True)
                if scheduler is not None:
                    scheduler.step()
        selected_model = ema_model if ema_model is not None else model
        payload = {"format_version": 2, "model_state_dict": selected_model.state_dict(), "config": config,
            "optimizer_state_dict": optimizer.state_dict(),
            "preprocessing": bundle.preprocessing, "classes": bundle.classes, "ap_order": bundle.ap_order,
            "seed": seed, "epoch": epochs, "validation_mean_2d": None, "run_type": "training",
            "split_manifest_sha256": metadata["split_manifest_sha256"], "metadata": metadata}
        if ema_model is not None:
            payload["training_model_state_dict"] = model.state_dict()
            payload["ema_updates"] = steps
        torch.save(payload, run_dir / "best.pt")
        resources = {"training_phase": memory.stop(), "cuda_training_phase": cuda_memory(device),
                     "training_phase_seconds": time.perf_counter() - started, "software": metadata["software"]}
        save_json(run_dir / "resources.json", resources)
        summary = {"status": "completed", "run_type": "fixed_epoch_final_refit", "seed": seed,
                   "epochs_completed": epochs, "train_samples": len(dataset), "test_evaluated": False,
                   "training_seconds": resources["training_phase_seconds"], "checkpoint": str(run_dir / "best.pt")}
        save_json(run_dir / "summary.json", summary)
        return summary
    finally:
        if not memory.stop_event.is_set():
            memory.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    refit(args.source_checkpoint, args.selection_lock, args.run_dir, args.seed, args.device)


if __name__ == "__main__":
    main()
