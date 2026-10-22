"""Train dùng train/val nội bộ; không đánh giá test trong vòng lặp."""
from __future__ import annotations

import csv
import copy
import json
import math
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


@torch.no_grad()
def update_ema(averaged, model, decay, steps):
    """Train-step EMA with a warm start; reference buffers stay identical."""
    if not 0 < decay < 1:
        raise ValueError("EMA decay must be strictly between zero and one.")
    effective_decay = min(decay, (1 + steps) / (10 + steps))
    for shadow, current in zip(averaged.parameters(), model.parameters()):
        shadow.mul_(effective_decay).add_(current.detach(), alpha=1-effective_decay)
    for shadow, current in zip(averaged.buffers(), model.buffers()):
        shadow.copy_(current)


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
        coordinate_source = settings.get("coordinate_source", "dense")
        if coordinate_source not in ("dense", "topk"):
            raise ValueError("coordinate_source must be dense or topk.")
        predicted_xy = output["dense_conditional_xy" if coordinate_source == "dense" else "conditional_xy"][
            torch.arange(len(classes), device=classes.device), classes]
        coordinate = torch.sqrt((predicted_xy - xy).square().sum(1) + 1e-8).mean()
        loss = (settings.get("lambda_class", 0.5) * classification
                + settings.get("lambda_reference", 1.0) * reference_loss
                + settings["lambda_xy"] * coordinate)
        if "auxiliary_logits" in output:
            loss = loss + settings.get("lambda_auxiliary_class", 0.5) * nn.functional.cross_entropy(
                output["auxiliary_logits"], classes)
        return loss, classification + reference_loss, coordinate
    coordinate = nn.functional.smooth_l1_loss(output["xy"], xy, beta=settings["smooth_l1_beta"])
    return classification + settings["lambda_xy"] * coordinate, classification, coordinate


def reference_consistency_loss(reference_logits, pair_count, temperature=1.):
    """Jensen-Shannon consistency for same-location train pairs, no teacher."""
    if temperature <= 0 or len(reference_logits) != 2*pair_count:
        raise ValueError('Consistency requires positive temperature and exactly two ordered train views.')
    left,right = (reference_logits/temperature).log_softmax(1).split(pair_count)
    mean = torch.logaddexp(left,right)-math.log(2)
    return .5*((left.exp()*(left-mean)).sum(1)+(right.exp()*(right-mean)).sum(1)).mean()


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
    consistency_weight = settings.get("same_point_consistency_weight", 0.)
    from .augmentation import training_representation
    loader_data = training_representation(train_data, settings)
    train_loader = DataLoader(loader_data, batch_size=settings["batch_size"], shuffle=True,
                              num_workers=0, generator=generator)
    validation_loader = DataLoader(validation_data, batch_size=settings["batch_size"], shuffle=False, num_workers=0)
    # KAN initialization (CPU lstsq) occurs before moving to GPU.
    model = build_model(config, len(bundle.classes)).to(device)
    ema_decay = settings.get("ema_decay", 0.0)
    if ema_decay and not 0 < ema_decay < 1:
        raise ValueError("EMA decay must be strictly between zero and one.")
    ema_model = copy.deepcopy(model).eval().requires_grad_(False) if ema_decay else None
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
    if settings.get("same_point_mixing", 0) or consistency_weight or settings.get('nearby_point_mixing', 0):
        training_files.append("algorithms/kan/augmentation.py")
    metadata = {"run_type": "smoke_check" if smoke else "training",
        "seed": seed, "config": config, "resolved_max_epochs": max_epochs,
        "actual_counts": {"train": len(train_data), "validation": len(validation_data)},
        "trainable_parameters": parameter_count(model), "software": software_info(device),
        "split_manifest_sha256": sha256(bundle.directory / "manifest.json"),
        "split_manifest": bundle.manifest, "code_sha256": hashes,
        "training_implementation_sha256": {"src/ips/" + name: hashes["src/ips/" + name] for name in training_files},
        "kan_source": json.loads((Path(__file__).parent / "models/vendor/provenance.json").read_text()),
        "selection": "internal_validation_mean_2d", "test_evaluated_during_training": False}
    metadata["selection_weights"] = "train_step_ema" if ema_model is not None else "live_model"
    metadata["training_views_per_dataset_row"] = 2 if consistency_weight else 1
    save_json(run_dir / "run_metadata.json", metadata)
    save_json(run_dir / "resolved_config.json", config)
    if smoke:
        save_json(run_dir / "smoke_indices.json", {"train": train_data.indices.tolist(),
                  "validation": validation_data.indices.tolist(), "selection": "fixed per-class quota in original row order"})
    fields = ["epoch", "train_loss", "train_class_loss", "train_xy_loss", "val_mean_2d",
              "val_joint_accuracy", "val_joint_macro_f1", "learning_rate", "epoch_seconds", "optimizer_steps"]
    if consistency_weight:
        fields.append("train_reference_consistency_loss")
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
                consistency_total = 0.
                for batch_index, batch in enumerate(train_loader):
                    features, classes, xy = batch[:3]
                    pair_count = len(features)
                    if consistency_weight:
                        features = torch.cat([features,batch[3]])
                        classes,xy = torch.cat([classes,classes]),torch.cat([xy,xy])
                    features, classes, xy = features.to(device), classes.to(device), xy.to(device)
                    diagnostics.enabled = batch_index < settings["kan_diagnostic_batches"]
                    optimizer.zero_grad(set_to_none=True)
                    output = model(features)
                    loss, class_loss, xy_loss = combined_loss(output, classes, xy, config["loss"])
                    if consistency_weight:
                        consistency = reference_consistency_loss(output['reference_logits'],pair_count,
                            settings.get('consistency_temperature',1.))
                        loss = loss+consistency_weight*consistency
                        consistency_total += consistency.item()*pair_count
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Non-finite loss, epoch {epoch}, batch {batch_index}.")
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip_l2"], error_if_nonfinite=True)
                    optimizer.step()
                    if ema_model is not None:
                        update_ema(ema_model, model, ema_decay, steps)
                    batch_size = len(features)
                    totals += np.asarray([loss.item(), class_loss.item(), xy_loss.item()]) * batch_size
                    samples += batch_size
                    steps += 1
                diagnostics.enabled = False
                selected_model = ema_model if ema_model is not None else model
                logits, normalized_xy, true_classes = infer(selected_model, validation_loader, device)
                predicted = logits.argmax(dim=1).numpy()
                predicted_xy = inverse_xy(normalized_xy.numpy(), bundle.preprocessing)
                metrics = evaluate_predictions(true_classes.numpy(), predicted, validation_data.xy_original,
                                               predicted_xy, bundle.classes)
                current = metrics["mean_2d"]
                if not np.isfinite(current):
                    raise FloatingPointError("Non-finite validation coordinate metric.")
                improved = current < best - settings["min_delta"]
                payload = {"format_version": 1, "model_state_dict": selected_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(), "config": config,
                    "preprocessing": bundle.preprocessing, "classes": bundle.classes,
                    "ap_order": bundle.ap_order, "seed": seed, "epoch": epoch,
                    "validation_mean_2d": current, "run_type": metadata["run_type"],
                    "split_manifest_sha256": metadata["split_manifest_sha256"], "metadata": metadata}
                if ema_model is not None:
                    payload["training_model_state_dict"] = model.state_dict()
                    payload["ema_updates"] = steps
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
                values = [epoch, *map(float, totals / samples), current,
                    metrics["joint_accuracy"], metrics["joint_macro_f1"], optimizer.param_groups[0]["lr"],
                    time.perf_counter() - epoch_started, steps]
                if consistency_weight:
                    values.append(consistency_total/len(train_data))
                row = dict(zip(fields,values))
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
