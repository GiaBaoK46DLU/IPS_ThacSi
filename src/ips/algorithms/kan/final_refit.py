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
from .train import combined_loss


def refit(source_checkpoint, selection_path, run_dir, seed, device_name="cuda"):
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("selection_split") != "internal_validation_only" or selection.get("test_evaluated") is not False:
        raise ValueError("Refit requires an explicit validation-only selection lock made before test evaluation.")
    epochs = selection["final_refit_epochs"]
    if type(epochs) is not int or not 1 <= epochs <= 100:
        raise ValueError("Frozen refit epoch count must be in 1..100.")
    if sha256(source_checkpoint) not in selection["allowed_source_checkpoint_sha256"]:
        raise ValueError("Source checkpoint was not included in the frozen selection.")
    if run_dir.exists():
        raise FileExistsError(f"Run already exists: {run_dir}")
    device = configure_runtime(seed, device_name, 2)
    # CPU load: use selected config only; refit network weights from a fresh seed.
    _, source = load_checkpoint(source_checkpoint, torch.device("cpu"))
    config = copy.deepcopy(source["config"])
    config["positioning"]["training_temperature"] = source["metadata"]["config"]["positioning"]["temperature"]
    bundle = load_bundle(ROOT / config["split_dir"])
    indices = np.arange(bundle.development.size, dtype=np.int64)
    dataset = FingerprintDataset(bundle.development, indices, bundle.preprocessing, bundle.classes)
    dataset.split_name = "development"
    fit_reference_points(config, dataset, final_refit=True)
    model = build_model(config, len(bundle.classes)).to(device)
    settings = config["training"]
    loader = DataLoader(dataset, batch_size=settings["batch_size"], shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    # Keep the original validation trial's LR trajectory, rather than compress
    # a 100-epoch cosine schedule into the selected number of refit epochs.
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, settings["max_epochs"],
                    eta_min=settings["learning_rate"] * 0.05) if settings.get("scheduler") == "cosine" else None)
    run_dir.mkdir(parents=True)
    metadata = {"full_development_refit": True, "selection_split": "previous_internal_validation_only",
        "selection_lock_sha256": sha256(selection_path), "selection_lock": selection,
        "source_checkpoint_sha256": sha256(source_checkpoint), "actual_counts": {"train": len(dataset), "validation": 0},
        "fit_scope": "development_after_validation_selection", "test_evaluated_during_training": False,
        "split_manifest_sha256": sha256(bundle.directory / "manifest.json"), "code_sha256": code_hashes(),
        "software": software_info(device), "trainable_parameters": parameter_count(model),
        "original_preprocessing_retained": True,
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
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for epoch in range(1, epochs + 1):
                epoch_started = time.perf_counter()
                model.train()
                totals = np.zeros(3)
                for x, y, xy in loader:
                    x, y, xy = x.to(device), y.to(device), xy.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    loss, classification, coordinates = combined_loss(model(x), y, xy, config["loss"])
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite refit loss")
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip_l2"], error_if_nonfinite=True)
                    optimizer.step()
                    totals += np.array([loss.item(), classification.item(), coordinates.item()]) * len(x)
                    steps += 1
                synchronize(device)
                row = dict(zip(fields, [epoch, *map(float, totals / len(dataset)),
                    optimizer.param_groups[0]["lr"], time.perf_counter() - epoch_started, steps]))
                writer.writerow(row)
                stream.flush()
                print(f"kan_mixer FULL-DEVELOPMENT seed={seed} epoch={epoch}/{epochs} loss={row['train_loss']:.4f}; no validation/test evaluation", flush=True)
                if scheduler is not None:
                    scheduler.step()
        payload = {"format_version": 2, "model_state_dict": model.state_dict(), "config": config,
            "preprocessing": bundle.preprocessing, "classes": bundle.classes, "ap_order": bundle.ap_order,
            "seed": seed, "epoch": epochs, "validation_mean_2d": None, "run_type": "training",
            "split_manifest_sha256": metadata["split_manifest_sha256"], "metadata": metadata}
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
