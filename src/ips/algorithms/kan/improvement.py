"""Validation-only, joint-building KAN research controller.

This entry point cannot evaluate the official test. Each trial gets immutable
configuration, source snapshots, train logs and validation-only calibration.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
import subprocess
import sys

from ...data import ROOT, sha256
from .runtime import save_json


KAN_ONLY_SCOPE = {
    "scope": "one_joint_kan_mixer",
    "reason": "User explicitly requires KAN-Mixer only, with no hybrid prediction method.",
    "allowed": ["KAN token/channel mixing", "internal pooling and prediction heads",
                "train-only RSSI augmentation", "loss/optimizer/schedule tuning",
                "single-set inference weights, including optional training-step EMA"],
    "excluded": ["fingerprint retrieval or embedding memory", "raw-RSSI kNN correction",
                 "other prediction models", "model ensembles", "multi-view inference averaging",
                 "separate building/floor experts"],
    "inference": "one KAN-Mixer forward pass followed by its reference-point coordinate decoder",
    "selection_split": "internal_validation_only",
    "official_test_evaluated": False,
}


def validate_kan_only(config, checkpoint=None):
    """Reject broader experimental paths before any training process starts."""
    settings = config.get("positioning", {})
    if (config.get("model") != "kan_mixer" or settings.get("variant") != "reference"
            or settings.get("head_type", "point_classifier") != "point_classifier"):
        raise ValueError("KAN-Mixer only: use the reference point_classifier head; no retrieval/memory model.")
    if settings.get("inference_shifts_db"):
        raise ValueError("KAN-Mixer only: use one forward pass, without multi-view averaging.")
    for payload in (config, settings, checkpoint or {}):
        if any(key in payload for key in ("fingerprint_refinement", "ensemble_members")):
            raise ValueError("KAN-Mixer only: hybrid refinement and ensembles are excluded.")


def record_scope(output):
    """Add the user's narrowed scope without rewriting the original protocol."""
    target = output / "kan_only_scope.json"
    if target.exists():
        if json.loads(target.read_text(encoding="utf-8")) != KAN_ONLY_SCOPE:
            raise FileExistsError(f"Recorded research scope differs: {target}")
        return
    save_json(target, KAN_ONLY_SCOPE)


def prepare(output):
    if output.exists():
        raise FileExistsError(f"Choose a new research directory: {output}")
    output.mkdir(parents=True)
    config_path = ROOT / "src/ips/algorithms/kan/configs/accuracy_spatial.json"
    base = json.loads(config_path.read_text(encoding="utf-8"))
    trials = {
        "mask": {"positioning": {"mask_scale": 1.0}},
        "robust_aug": {"positioning": {"rssi_noise_db": 4.0, "device_shift_db": 6.0,
                                         "ap_dropout": 0.1}},
    }
    directory = output / "configs"
    directory.mkdir()
    for name, changes in trials.items():
        config = copy.deepcopy(base)
        for key, values in changes.items():
            config[key].update(values)
        save_json(directory / f"{name}.json", config)
    save_json(output / "protocol.json", {
        "objective": "mean 2D error below 6 metres on official UJI test, joint building-floor model",
        "baseline": "ReferenceKANMixer spatial, three-seed test 8.058152 +/- 0.306071 metres",
        "branch": "feature/kan-duoi-sau-met", "source_config_sha256": sha256(config_path),
        "split_manifest_sha256": sha256(ROOT / base["split_dir"] / "manifest.json"),
        "screening_seed": 42, "selection_split": "internal_validation_only",
        "test_used_for_selection": False, "known_building_or_floor_at_inference": False,
        "training_buildings": [0, 1, 2], "all_13_classes_in_one_model": True,
        "historical_test_exposure": True,
        "screening_trials": trials,
        "confirmation_policy": "freeze architecture/decoder using validation; confirm multiple seeds before test",
        "causality_policy": "compare one-factor ablations; report unproven interactions as hypotheses",
    })
    record_scope(output)


def trial(output, name, seed=42):
    config_path = output / "configs" / f"{name}.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    protocol = json.loads((output/'protocol.json').read_text())
    config = json.loads(config_path.read_text())
    validate_kan_only(config)
    record_scope(output)
    if sha256(ROOT/config['split_dir']/'manifest.json') != protocol['split_manifest_sha256']:
        raise ValueError('Research trial must retain the frozen joint UJI split and reference KAN.')
    destination = output / "trials" / name / f"seed{seed}"
    if destination.exists():
        raise FileExistsError(f"Trial already exists: {destination}")
    destination.mkdir(parents=True)
    source = destination / "source"
    for path in (ROOT / "src/ips").rglob("*.py"):
        target = source / path.relative_to(ROOT / "src")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    commands = [
        [sys.executable, "-u", "-m", "ips", "train", "--config", str(config_path),
         "--run-dir", str(destination / "training"), "--seed", str(seed), "--device", "cuda"],
        [sys.executable, "-u", "-m", "ips.algorithms.kan.accuracy", "--run-dir",
         str(destination / "training"), "--output-dir", str(destination / "calibration"),
         "--device", "cuda", "--pure-only"],
    ]
    save_json(destination / "launch.json", {"config_sha256": sha256(config_path), "commands": commands,
        "seed": seed, "selection_split": "validation", "test_evaluated": False,
        "kan_only_scope_sha256": sha256(output / "kan_only_scope.json")})
    with (destination / "terminal.log").open("w", encoding="utf-8") as log:
        for stage, command in enumerate(commands):
            snapshot = destination / f"stage{stage}_source_hashes.json"
            save_json(snapshot, {p.relative_to(ROOT).as_posix(): sha256(p)
                for p in (ROOT / "src/ips").rglob("*.py")})
            print("RUN " + " ".join(command), flush=True)
            process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            if process.wait():
                raise RuntimeError(f"Trial failed; inspect {destination / 'terminal.log'}")
    calibration = json.loads((destination / "calibration/calibration.json").read_text())
    if calibration["selected"]["kind"] != "pure_reference_kan_mixer":
        raise ValueError("Research output must be a single, unrefined KAN-Mixer.")
    save_json(destination / "completed.json", {"seed": seed, "name": name,
        "best_validation": calibration["best_pure"], "test_evaluated": False})


def extend(output):
    """Record additional designs before running them; never rewrite a trial."""
    designs = {
        "relative": {"mixer": {"features_per_token": 60}, "positioning": {
            "input_mode": "absolute_relative_mask", "mask_scale": 1.0}},
        "relative_only": {"positioning": {"input_mode": "relative_mask", "mask_scale": 1.0}},
        "aligned_topk": {"loss": {"coordinate_source": "topk"}, "positioning": {"top_k": 20}},
        "auxiliary": {"positioning": {"auxiliary_classifier": True, "class_logit_weight": 1.0},
                      "loss": {"lambda_auxiliary_class": 0.5}},
        "learned_pool": {"positioning": {"pooling": "learned"}},
        "ema": {"training": {"ema_decay": 0.995}},
        "same_point": {"training": {"same_point_mixing": 0.5}},
        "relative_ema": {"mixer": {"features_per_token": 60}, "training": {"ema_decay": 0.995},
                         "positioning": {"input_mode": "absolute_relative_mask", "mask_scale": 1.0}},
        "regularized": {"mixer": {"dropout": 0.2}, "training": {"weight_decay": 0.01}},
        "sigma_tight": {"loss": {"reference_sigma_normalized": 0.025}},
        "sigma_wide": {"loss": {"reference_sigma_normalized": 0.08}},
        "fine_tokens": {"mixer": {"tokens": 52, "features_per_token": 20, "token_hidden": 64}},
        "aligned_aux_relative": {"mixer": {"features_per_token": 60},
            "positioning": {"input_mode": "absolute_relative_mask", "mask_scale": 1.0,
                            "auxiliary_classifier": True, "top_k": 20},
            "loss": {"coordinate_source": "topk", "lambda_auxiliary_class": 0.5}},
        "budget100": {"training": {"patience":100}},
        "same_point_relative": {"training": {"same_point_mixing":0.5},
                                "positioning": {"input_mode":"relative_mask","mask_scale":1.0}},
        "same_point_aux_topk": {"training": {"same_point_mixing":0.5},
            "positioning": {"auxiliary_classifier":True,"top_k":20},
            "loss": {"coordinate_source":"topk","lambda_auxiliary_class":0.5}},
        "kan_head": {"training": {"patience":100}, "positioning": {"head_transform":"kan"}},
        "same_point_kan_head": {"training": {"patience":100,"same_point_mixing":0.5},
                                "positioning": {"head_transform":"kan"}},
        "deeper_same_point": {"training": {"patience":100,"same_point_mixing":0.5},
                              "mixer": {"blocks":3}},
        "same_point_cross_phone": {"training": {"patience":100,"same_point_mixing":0.5,
            "same_point_partner_policy":"different_phone_first"}},
        "cross_phone_consistency": {"training": {"patience":100,"same_point_consistency_weight":1.,
            "same_point_partner_policy":"different_phone_first","consistency_temperature":1.}},
        "same_point_consistency": {"training": {"patience":100,"same_point_mixing":0.5,
            "same_point_consistency_weight":1.,"same_point_partner_policy":"different_phone_first",
            "consistency_temperature":1.}},
    }
    base = json.loads((ROOT / "src/ips/algorithms/kan/configs/accuracy_spatial.json").read_text())
    for name, changes in designs.items():
        config = copy.deepcopy(base)
        for key, values in changes.items():
            config[key].update(values)
        validate_kan_only(config)
        target = output / "configs" / f"{name}.json"
        if target.exists():
            if json.loads(target.read_text()) != config:
                raise FileExistsError(f"Immutable design differs: {target}")
            continue
        save_json(target, config)
        save_json(target.with_suffix('.design.json'), {"name": name, "overrides": changes,
            "baseline": "accuracy_spatial.json", "test_used_for_design": False,
            "source_code_sha256": {p.relative_to(ROOT).as_posix(): sha256(p)
                                    for p in (ROOT / 'src/ips/algorithms/kan').rglob('*.py')}})


def summarize(output):
    """Rebuild validation-only ledger and exact configuration differences."""
    base = json.loads((ROOT/'src/ips/algorithms/kan/configs/accuracy_spatial.json').read_text())
    rows, designs, excluded = [], {}, []
    for config_path in sorted((output/'configs').glob('*.json')):
        if config_path.name.endswith('.design.json'):
            continue
        config = json.loads(config_path.read_text())
        name = config_path.stem
        try:
            validate_kan_only(config)
        except ValueError as error:
            excluded.append({"name": name, "reason": str(error),
                             "config_sha256": sha256(config_path)})
            continue
        differences = {}
        for section in config:
            if isinstance(config[section], dict):
                for key, value in config[section].items():
                    if base.get(section, {}).get(key) != value:
                        differences[f'{section}.{key}'] = {'before':base.get(section,{}).get(key), 'after':value}
            elif base.get(section) != config[section]:
                differences[section] = {'before':base.get(section), 'after':config[section]}
        designs[name] = differences
        for result in sorted((output/'trials'/name).glob('seed*/completed.json')):
            trial = json.loads(result.read_text())
            summary = json.loads((result.parent/'training/summary.json').read_text())
            support_path = result.parent/'validation_support.json'
            support = json.loads(support_path.read_text()) if support_path.exists() else {}
            metadata = json.loads((result.parent/'training/run_metadata.json').read_text())
            rows.append({'name':name,'seed':trial['seed'],
                         'validation_mean_2d':trial['best_validation']['mean_2d'],
                         'validation_joint_accuracy':trial['best_validation']['joint_accuracy'],
                         'k':trial['best_validation']['top_k'],'temperature':trial['best_validation']['temperature'],
                         'epochs':summary['epochs_completed'],'best_epoch':summary['best_epoch'],
                         'parameters':summary['trainable_parameters'],'training_seconds':summary['training_seconds'],
                         'train_views_per_row':metadata.get('training_views_per_dataset_row',1),
                         'seen_coordinate_mean_2d':support.get('seen_train_coordinate',{}).get('mean_2d'),
                         'unseen_coordinate_mean_2d':support.get('unseen_train_coordinate',{}).get('mean_2d'),
                         'test_evaluated':False})
    save_json(output/'exact_configuration_changes.json', designs)
    save_json(output/'screening.json', {'selection_split':'validation','test_evaluated':False,
                                      'scope': KAN_ONLY_SCOPE, 'excluded_designs': excluded,
                                      'results':sorted(rows,key=lambda r:r['validation_mean_2d'])})
    if rows:
        with (output/'screening.csv').open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(sorted(rows,key=lambda r:r['validation_mean_2d']),indent=2),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "extend", "summarize", "trial", "scope"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if args.stage == "prepare":
        prepare(output)
    elif args.stage == "extend":
        extend(output)
    elif args.stage == "summarize":
        summarize(output)
    elif args.stage == "scope":
        record_scope(output)
    else:
        if not args.name:
            parser.error("trial requires --name")
        trial(output, args.name, args.seed)


if __name__ == "__main__":
    main()
