"""Train dùng train/val nội bộ; không đánh giá test trong vòng lặp."""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import ROOT, sha256
from .dataset import load_bundle
from .diagnostics import KANDiagnostics
from .metrics import evaluate_predictions, inverse_xy
from .models import build_model, parameter_count
from .runtime import code_hashes, configure_runtime, save_json, software_info


def combined_loss(output, classes, xy, settings):
    classification = nn.functional.cross_entropy(output["logits"], classes)
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
          epochs: int | None = None, smoke: bool = False):
    if epochs is not None and epochs < 1:
        raise ValueError("epochs must be positive.")
    # Mỗi lượt có thư mục mới, tránh vô tình trộn log/checkpoint các mô hình.
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}. Choose a new --run-dir.")
    device = configure_runtime(seed, device_name, config["training"]["threads"])
    bundle = load_bundle(ROOT / config["split_dir"])
    train_data = bundle.dataset("train", limit=256 if smoke else None)
    validation_data = bundle.dataset("validation", limit=128 if smoke else None)
    settings = config["training"]
    max_epochs = 1 if smoke else (epochs if epochs is not None else settings["max_epochs"])
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_data, batch_size=settings["batch_size"], shuffle=True,
                              num_workers=0, generator=generator)
    validation_loader = DataLoader(validation_data, batch_size=settings["batch_size"], shuffle=False, num_workers=0)
    # KAN initialization (CPU lstsq) occurs before moving to GPU.
    model = build_model(config, len(bundle.classes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    diagnostics = KANDiagnostics(model)
    run_dir.mkdir(parents=True)
    metadata = {"run_type": "smoke_check" if smoke else "training",
        "seed": seed, "config": config, "resolved_max_epochs": max_epochs,
        "actual_counts": {"train": len(train_data), "validation": len(validation_data)},
        "trainable_parameters": parameter_count(model), "software": software_info(device),
        "split_manifest_sha256": sha256(bundle.directory / "manifest.json"),
        "split_manifest": bundle.manifest, "code_sha256": code_hashes(),
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
    started = time.perf_counter()
    try:
        with (run_dir / "history.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for epoch in range(1, max_epochs + 1):
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
                row = dict(zip(fields, [epoch, *map(float, totals / samples), current,
                    metrics["joint_accuracy"], metrics["joint_macro_f1"], optimizer.param_groups[0]["lr"],
                    time.perf_counter() - epoch_started, steps]))
                writer.writerow(row)
                stream.flush()
                print(f"{config['model']} seed={seed} epoch={epoch}/{max_epochs} "
                      f"loss={row['train_loss']:.4f} val_mean2d={current:.3f} "
                      f"val_joint_acc={metrics['joint_accuracy']:.4f}", flush=True)
                if stale >= settings["patience"]:
                    break
        summary = {"status": "completed", "run_type": metadata["run_type"], "model": config["model"],
                   "seed": seed, "epochs_completed": epoch, "best_epoch": best_epoch,
                   "best_validation_mean_2d": best, "optimizer_steps": steps,
                   "trainable_parameters": metadata["trainable_parameters"],
                   "training_seconds": time.perf_counter() - started,
                   "test_evaluated": False}
        save_json(run_dir / "summary.json", summary)
        return summary
    except Exception as error:
        save_json(run_dir / "failure.json", {"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        diagnostics.close()
