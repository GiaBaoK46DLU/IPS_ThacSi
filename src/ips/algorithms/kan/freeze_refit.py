"""Serialize final-refit models using a PREVIOUS validation selection lock.

This step does not compute any validation or test predictions/metrics.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from ...data import ROOT, sha256
from ...data_access import load_bundle
from .dataset import FingerprintDataset
from .models.refinement import db_features
from .runtime import load_checkpoint, save_json


def freeze(paths, lock_path, output_dir):
    policy = json.loads(lock_path.read_text(encoding="utf-8"))
    if policy.get("selection_split") != "internal_validation_only" or policy.get("test_evaluated") is not False:
        raise ValueError("Final artifact freezing requires a pre-test validation selection lock.")
    output_dir.mkdir(parents=True, exist_ok=False)
    payloads = []
    for path in paths:
        _, payload = load_checkpoint(path, torch.device("cpu"))
        if (not payload["metadata"].get("full_development_refit")
                or payload["metadata"].get("selection_lock_sha256") != sha256(lock_path)):
            raise ValueError("Refit members do not share the frozen selection lock.")
        payloads.append(payload)
    first = payloads[0]
    for payload in payloads:
        for key in ("classes", "preprocessing", "ap_order", "split_manifest_sha256"):
            if payload[key] != first[key]:
                raise ValueError(f"Refit member {key} differs.")
    bundle = load_bundle(ROOT / first["config"]["split_dir"])
    dataset = FingerprintDataset(bundle.development, np.arange(bundle.development.size),
                                 bundle.preprocessing, bundle.classes)
    refinement = {"signal": db_features(dataset.features), "xy": dataset.xy, "class_ids": dataset.classes,
                  "row_indices": dataset.indices, "fit_scope": "development_after_validation_selection",
                  "development_source_sha256": bundle.development.digest,
                  "settings": policy["frozen_hybrid_settings"]}
    artifacts = []
    for path, source in zip(paths, payloads):
        payload = copy.deepcopy(source)
        payload["fingerprint_refinement"] = refinement
        destination = output_dir / f"hybrid_seed{source['seed']}.pt"
        torch.save(payload, destination)
        artifacts.append({"kind": "full_development_hybrid", "path": str(destination), "sha256": sha256(destination)})
    if len(payloads) >= 2:
        payload = copy.deepcopy(first)
        payload.pop("model_state_dict", None)
        payload["ensemble_members"] = [{"config": p["config"], "model_state_dict": p["model_state_dict"]} for p in payloads]
        payload["ensemble_confidence_weighted"] = policy["frozen_ensemble_confidence_weighted"]
        payload["seed"] = -1
        payload["metadata"]["ensemble_members"] = [{"path": str(p), "sha256": sha256(p), "seed": c["seed"]} for p, c in zip(paths, payloads)]
        destination = output_dir / "pure_ensemble.pt"
        torch.save(payload, destination)
        artifacts.append({"kind": "full_development_pure_ensemble", "path": str(destination), "sha256": sha256(destination)})
        payload["fingerprint_refinement"] = refinement
        destination = output_dir / "hybrid_ensemble.pt"
        torch.save(payload, destination)
        artifacts.append({"kind": "full_development_hybrid_ensemble", "path": str(destination), "sha256": sha256(destination)})
    save_json(output_dir / "frozen_artifacts.json", {"selection_lock_sha256": sha256(lock_path), "test_evaluated": False, "artifacts": artifacts})
    print(f"Frozen {len(artifacts)} full-development artifacts; no validation/test evaluated.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--members", nargs="+", type=Path, required=True)
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    freeze(args.members, args.selection_lock, args.output_dir)


if __name__ == "__main__":
    main()
