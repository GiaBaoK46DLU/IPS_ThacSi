"""Train dùng train/val nội bộ; không đánh giá test trong vòng lặp."""
from __future__ import annotations

import csv
import copy
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
from .diagnostics import KANDiagnostics
from ...metrics import evaluate_predictions, inverse_xy
from .models import build_model, parameter_count
from .runtime import code_hashes, configure_runtime, save_json, software_info
from .resources import MemorySampler, cuda_memory, synchronize


def combined_loss(output, classes, xy, settings):
    classification = nn.functional.cross_entropy(output["logits"], classes)
    if "reference_logits" in output:
        distances = torch.cdist(xy, output["reference_xy"]).square()
        distances = distances.masked_fill(
            output["reference_classes"][None] != classes[:, None], float("inf"))
        targets = distances.argmin(1)
        sigma = settings.get("reference_sigma_normalized", 0.0)
        if sigma > 0:
            # Spatially nearby points on the same floor get compatible targets.
            # These targets use training labels only, never query labels.
            soft_targets = (-distances / (2 * sigma ** 2)).softmax(1)
            reference_loss = -(soft_targets * output["reference_logits"].log_softmax(1)).sum(1).mean()
        else:
            reference_loss = nn.functional.cross_entropy(output["reference_logits"], targets,
                                          label_smoothing=settings.get("label_smoothing", 0.0))
        predicted_xy = output["dense_conditional_xy"][
            torch.arange(len(classes), device=classes.device), classes]
        coordinate = torch.sqrt((predicted_xy - xy).square().sum(1) + 1e-8).mean()
        loss = (settings.get("lambda_class", 0.5) * classification
                + settings.get("lambda_reference", 1.0) * reference_loss
                + settings["lambda_xy"] * coordinate)
        return loss, classification + reference_loss, coordinate
    coordinate = nn.functional.smooth_l1_loss(output["xy"], xy, beta=settings["smooth_l1_beta"])
    return classification + settings["lambda_xy"] * coordinate, classification, coordinate


@torch.inference_mode()
def infer(model, loader, device):
    model.eval()
    logits, coordinates, targets = [], [], []
    for features, classes, _ in loader:
        output = model(features.to(device))
        logits.append(output["logits"].cpu())
        coordinates.append(output["xy"].cpu())
        targets.append(classes)
    return torch.cat(logits), torch.cat(coordinates), torch.cat(targets)


def train(config: dict, run_dir: Path, seed: int, device_name: str,
          epochs: int | None = None, smoke: bool = False, *,
          bundle=None, train_data=None, validation_data=None):
    if epochs is not None and epochs < 1:
        raise ValueError("epochs must be positive.")
    supplied = [bundle is not None, train_data is not None, validation_data is not None]
    if any(supplied) and not all(supplied):
        raise ValueError("Supply bundle metadata, train_data and validation_data together.")
    # Mỗi lượt có thư mục mới, tránh vô tình trộn log/checkpoint các mô hình.
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}. Choose a new --run-dir.")
    device = configure_runtime(seed, device_name, config["training"]["threads"])
    pipeline_started = time.perf_counter()
    pipeline_memory = MemorySampler().start()
    if not any(supplied):
        bundle = load_bundle(ROOT / config["split_dir"])
        train_data = FingerprintDataset.from_split(bundle.split("train", limit=256 if smoke else None))
        validation_data = FingerprintDataset.from_split(bundle.split("validation", limit=128 if smoke else None))
    config = copy.deepcopy(config)
    from .models.reference import fit_reference_points
    fit_reference_points(config, train_data)
    settings = config["training"]
    max_epochs = 1 if smoke else (epochs if epochs is not None else settings["max_epochs"])
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_data, batch_size=settings["batch_size"], shuffle=True,
                              num_workers=0, generator=generator)
    validation_loader = DataLoader(validation_data, batch_size=settings["batch_size"], shuffle=False, num_workers=0)
    # KAN initialization (CPU lstsq) occurs before moving to GPU.
    model = build_model(config, len(bundle.classes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_epochs,
                 eta_min=settings.get("minimum_learning_rate", settings["learning_rate"] * 0.05))
                 if settings.get("scheduler") == "cosine" else None)
    diagnostics = KANDiagnostics(model)
    run_dir.mkdir(parents=True)
    hashes = code_hashes()
    training_files = ["data.py", "metrics.py", "data_access/contracts.py", "data_access/repository.py"]
    training_files += ["algorithms/kan/" + name for name in
        ["algorithm.py", "dataset.py", "train.py", "predict.py", "runtime.py", "resources.py", "diagnostics.py",
         "models/networks.py", "models/vendor/efficient_kan.py"]]
    if config.get("positioning", {}).get("variant") == "reference":
        training_files.append("algorithms/kan/models/reference.py")
    metadata = {"run_type": "smoke_check" if smoke else "training",
        "seed": seed, "config": config, "resolved_max_epochs": max_epochs,
        "actual_counts": {"train": len(train_data), "validation": len(validation_data)},
        "trainable_parameters": parameter_count(model), "software": software_info(device),
        "split_manifest_sha256": sha256(bundle.directory / "manifest.json"),
        "split_manifest": bundle.manifest, "code_sha256": hashes,
        "training_implementation_sha256": {"src/ips/" + name: hashes["src/ips/" + name] for name in training_files},
        "kan_source": json.loads((Path(__file__).parent / "models/vendor/provenance.json").read_text()),
        "selection": "internal_validation_mean_2d", "test_evaluated_during_training": False}
    save_json(run_dir / "run_metadata.json", metadata)
    save_json(run_dir / "resolved_config.json", config)
    if smoke:
        save_json(run_dir / "smoke_indices.json", {"train": train_data.indices.tolist(),
                  "validation": validation_data.indices.tolist(), "selection": "fixed per-class quota in original row order"})
    fields = ["epoch", "train_loss", "train_class_loss", "train_xy_loss", "val_mean_2d",
              "val_joint_accuracy", "val_joint_macro_f1", "learning_rate", "epoch_seconds", "optimizer_steps"]
    best = float("inf")
    stale = 0
    steps = 0
    diagnostic_log = []
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    training_memory = MemorySampler().start()
    started = time.perf_counter()
    try:
        with (run_dir / "history.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for epoch in range(1, max_epochs + 1):
                synchronize(device)
                epoch_started = time.perf_counter()
                diagnostics.reset()
                model.train()
                totals = np.zeros(3)
                samples = 0
                for batch_index, (features, classes, xy) in enumerate(train_loader):
                    features, classes, xy = features.to(device), classes.to(device), xy.to(device)
                    diagnostics.enabled = batch_index < settings["kan_diagnostic_batches"]
                    optimizer.zero_grad(set_to_none=True)
                    output = model(features)
                    loss, class_loss, xy_loss = combined_loss(output, classes, xy, config["loss"])
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Non-finite loss, epoch {epoch}, batch {batch_index}.")
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip_l2"], error_if_nonfinite=True)
                    optimizer.step()
                    batch_size = len(features)
                    totals += np.asarray([loss.item(), class_loss.item(), xy_loss.item()]) * batch_size
                    samples += batch_size
                    steps += 1
                diagnostics.enabled = False
                logits, normalized_xy, true_classes = infer(model, validation_loader, device)
                predicted = logits.argmax(dim=1).numpy()
                predicted_xy = inverse_xy(normalized_xy.numpy(), bundle.preprocessing)
                metrics = evaluate_predictions(true_classes.numpy(), predicted, validation_data.xy_original,
                                               predicted_xy, bundle.classes)
                current = metrics["mean_2d"]
                if not np.isfinite(current):
                    raise FloatingPointError("Non-finite validation coordinate metric.")
                improved = current < best - settings["min_delta"]
                payload = {"format_version": 1, "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(), "config": config,
                    "preprocessing": bundle.preprocessing, "classes": bundle.classes,
                    "ap_order": bundle.ap_order, "seed": seed, "epoch": epoch,
                    "validation_mean_2d": current, "run_type": metadata["run_type"],
                    "split_manifest_sha256": metadata["split_manifest_sha256"], "metadata": metadata}
                torch.save(payload, run_dir / "last.pt")
                if improved:
                    best, stale, best_epoch = current, 0, epoch
                    torch.save(payload, run_dir / "best.pt")
                    save_json(run_dir / "best_validation_metrics.json", metrics)
                else:
                    stale += 1
                diagnostic_log.append({"epoch": epoch, "sampled_train_batches": min(len(train_loader),
                    settings["kan_diagnostic_batches"]), "layers": diagnostics.summary()})
                save_json(run_dir / "kan_diagnostics.json", diagnostic_log)
                synchronize(device)
                row = dict(zip(fields, [epoch, *map(float, totals / samples), current,
                    metrics["joint_accuracy"], metrics["joint_macro_f1"], optimizer.param_groups[0]["lr"],
                    time.perf_counter() - epoch_started, steps]))
                writer.writerow(row)
                stream.flush()
                print(f"{config['model']} seed={seed} epoch={epoch}/{max_epochs} "
                      f"loss={row['train_loss']:.4f} val_mean2d={current:.3f} "
                      f"val_joint_acc={metrics['joint_accuracy']:.4f}", flush=True)
                if scheduler is not None:
                    scheduler.step()
                if stale >= settings["patience"]:
                    break
        synchronize(device)
        elapsed_training = time.perf_counter() - started
        resources = {"training_phase": training_memory.stop(), "pipeline": pipeline_memory.stop(),
                     "cuda_training_phase": cuda_memory(device),
                     "training_phase_seconds": elapsed_training,
                     "pipeline_seconds_including_data_and_initialization": time.perf_counter() - pipeline_started,
                     "checkpoint_best_bytes": (run_dir / "best.pt").stat().st_size,
                     "parameter_tensor_bytes": sum(p.numel()*p.element_size() for p in model.parameters()),
                     "buffer_tensor_bytes": sum(b.numel()*b.element_size() for b in model.buffers()),
                     "software": metadata["software"]}
        save_json(run_dir / "resources.json", resources)
        summary = {"status": "completed", "run_type": metadata["run_type"], "model": config["model"],
                   "seed": seed, "epochs_completed": epoch, "best_epoch": best_epoch,
                   "best_validation_mean_2d": best, "optimizer_steps": steps,
                   "trainable_parameters": metadata["trainable_parameters"],
                   "training_seconds": elapsed_training,
                   "test_evaluated": False}
        save_json(run_dir / "summary.json", summary)
        return summary
    except Exception as error:
        save_json(run_dir / "failure.json", {"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        diagnostics.close()
        if not training_memory.stop_event.is_set():
            training_memory.stop()
        if not pipeline_memory.stop_event.is_set():
            pipeline_memory.stop()
