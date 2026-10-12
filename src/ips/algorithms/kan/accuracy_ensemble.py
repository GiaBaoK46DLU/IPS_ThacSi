"""Select ensemble averaging on validation, then serialize a self-contained model."""
import argparse
import copy
from pathlib import Path

import torch

from ...data import ROOT, sha256
from ...data_access import load_bundle
from ...metrics import evaluate_predictions, inverse_xy
from .dataset import FingerprintDataset
from .models.ensemble import EnsembleReferenceKANMixer
from .models.refinement import db_features, RefinedKANMixer
from .runtime import configure_runtime, load_checkpoint, save_json


@torch.inference_mode()
def assemble(paths, output_dir, device_name="cuda"):
    output_dir.mkdir(parents=True, exist_ok=False)
    device = configure_runtime(42, device_name, 2)
    loaded = [load_checkpoint(path, device) for path in paths]
    models, payloads = zip(*loaded)
    if any("fingerprint_refinement" in p or "ensemble_members" in p for p in payloads):
        raise ValueError("Ensemble members must be independently selected pure reference KAN-Mixers.")
    first = payloads[0]
    if any(p.get("metadata", {}).get("full_development_refit") for p in payloads):
        raise ValueError("Final refit incorporated validation; ensemble settings must be frozen beforehand.")
    for payload in payloads:
        if (payload["preprocessing"] != first["preprocessing"] or payload["classes"] != first["classes"]
                or payload["ap_order"] != first["ap_order"]
                or payload["split_manifest_sha256"] != first["split_manifest_sha256"]
                or payload["run_type"] != "training"):
            raise ValueError("Member preprocessing, labels, split or finalization differ.")
    bundle = load_bundle(ROOT / first["config"]["split_dir"])
    validation = FingerprintDataset.from_split(bundle.split("validation"))
    train = FingerprintDataset.from_split(bundle.split("train"))

    def measure(model):
        outputs = [model(x.to(device)) for x in validation.features.split(64)]
        probabilities = torch.cat([o["logits"] for o in outputs]).argmax(1).cpu().numpy()
        xy = inverse_xy(torch.cat([o["xy"] for o in outputs]).cpu().numpy(), bundle.preprocessing)
        return evaluate_predictions(validation.classes.numpy(), probabilities, validation.xy_original, xy, bundle.classes)

    candidates = []
    for weighted in (False, True):
        model = EnsembleReferenceKANMixer(list(models), weighted).to(device).eval()
        metrics = measure(model)
        candidates.append({"kind": "pure_kan_mixer_ensemble", "confidence_weighted": weighted,
                           "mean_2d": metrics["mean_2d"], "joint_accuracy": metrics["joint_accuracy"]})
    pure = min(candidates, key=lambda r: r["mean_2d"]).copy()
    selected = pure.copy()
    ensemble = EnsembleReferenceKANMixer(list(models), pure["confidence_weighted"]).to(device).eval()
    refinement = {"fit_scope": "internal_train_only", "signal": db_features(train.features).to(device),
                  "xy": train.xy.to(device), "class_ids": train.classes.to(device),
                  "row_indices": train.indices, "train_indices_sha256": sha256(bundle.directory / "train_indices.csv")}
    # k=10, inverse-square weights were chosen on the first trial's validation.
    # Sweep only blend strength here, not another full nearest-neighbor grid.
    for class_weight in (0.0, 0.5, 1.0):
        for alpha in (0.25, 0.5, 0.75):
            settings = {"neighbors": 10, "distance_power": 2.0,
                        "neighbor_class_weight": class_weight, "neural_coordinate_weight": alpha}
            refinement["settings"] = settings
            model = RefinedKANMixer(ensemble, refinement).to(device).eval()
            metrics = measure(model)
            row = {"kind": "hybrid_kan_mixer_ensemble", "confidence_weighted": pure["confidence_weighted"],
                   **settings, "mean_2d": metrics["mean_2d"], "joint_accuracy": metrics["joint_accuracy"]}
            candidates.append(row)
            if row["mean_2d"] < selected["mean_2d"]:
                selected = row.copy()
    payload = copy.deepcopy(first)
    payload.pop("model_state_dict", None)
    payload.pop("optimizer_state_dict", None)
    payload["format_version"] = 2
    payload["ensemble_members"] = [{"config": p["config"], "model_state_dict": p["model_state_dict"]} for p in payloads]
    payload["ensemble_confidence_weighted"] = pure["confidence_weighted"]
    payload["epoch"] = -1
    payload["seed"] = -1
    payload["validation_mean_2d"] = pure["mean_2d"]
    metadata = {"selection_split": "internal_validation_only", "test_evaluated": False,
        "members": [{"checkpoint": str(path), "sha256": sha256(path), "seed": p["seed"], "epoch": p["epoch"]}
                    for path, p in zip(paths, payloads)], "candidates": candidates, "best_pure": pure, "selected": selected}
    payload["metadata"]["ensemble_selection"] = metadata
    torch.save(payload, output_dir / "pure_selected.pt")
    if selected["kind"] == "hybrid_kan_mixer_ensemble":
        refinement["settings"] = {k: selected[k] for k in ["neighbors", "distance_power", "neighbor_class_weight", "neural_coordinate_weight"]}
        payload["fingerprint_refinement"] = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in refinement.items()}
        payload["validation_mean_2d"] = selected["mean_2d"]
    torch.save(payload, output_dir / "selected.pt")
    save_json(output_dir / "selection.json", metadata)
    print(f"Ensemble: pure={pure}; selected={selected}; official test unopened", flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--members", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    assemble(args.members, args.output_dir, args.device)


if __name__ == "__main__":
    main()
