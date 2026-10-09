"""CLI chạy tại repo: python -m ips hoặc ips sau editable install."""
import argparse
import subprocess
import sys
from pathlib import Path

from .data import OUTPUT, ROOT
from .data_access import load_bundle
from .algorithms.kan.models import build_model, parameter_count
from .algorithms.kan.runtime import configure_runtime, read_config
from .algorithms.kan.paths import MODEL_CONFIG, resolve_config_path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Shared UJIIndoorLoc IPS: KAN-Mixer")
    commands = parser.add_subparsers(dest="command", required=True)
    factory_run = commands.add_parser("run", help="Run an IPS algorithm through the shared Factory")
    factory_run.add_argument("--config", type=Path, required=True)
    commands.add_parser("algorithms", help="List implemented IPS algorithm IDs")
    prepare = commands.add_parser("prepare", help="Audit + materialize + verify lesson-5 split")
    prepare.add_argument("--verify-only", action="store_true")
    check = commands.add_parser("check", help="Data integrity, tensor shapes, model parameter counts")
    check.add_argument("--config", type=Path, default=MODEL_CONFIG)
    check.add_argument("--device", default="cpu")
    train_parser = commands.add_parser("train", help="Train with internal validation only")
    train_parser.add_argument("--config", type=Path, default=MODEL_CONFIG)
    train_parser.add_argument("--model", choices=["kan_mixer"], default="kan_mixer")
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--device", default="auto")
    train_parser.add_argument("--epochs", type=int)
    train_parser.add_argument("--smoke", action="store_true")
    train_parser.add_argument("--run-dir", type=Path, required=True)
    prediction = commands.add_parser("predict", help="Infer from CSV; no labels required")
    prediction.add_argument("--checkpoint", type=Path, required=True)
    prediction.add_argument("--csv", type=Path, required=True)
    prediction.add_argument("--output", type=Path, required=True)
    prediction.add_argument("--device", default="auto")
    evaluation = commands.add_parser("evaluate", help="Evaluate saved checkpoint, defaults to internal validation")
    evaluation.add_argument("--checkpoint", type=Path, required=True)
    evaluation.add_argument("--split-dir", type=Path, default=OUTPUT)
    evaluation.add_argument("--split", choices=["validation", "test"], default="validation")
    evaluation.add_argument("--output-dir", type=Path, required=True)
    evaluation.add_argument("--device", default="auto")
    benchmark_parser = commands.add_parser("benchmark", help="Inference latency and memory on fixed RSSI inputs")
    benchmark_parser.add_argument("--checkpoint", type=Path, required=True)
    benchmark_parser.add_argument("--csv", type=Path, required=True)
    benchmark_parser.add_argument("--output", type=Path, required=True)
    benchmark_parser.add_argument("--batch-size", type=int, default=1)
    benchmark_parser.add_argument("--warmup", type=int, default=10)
    benchmark_parser.add_argument("--repeats", type=int, default=100)
    benchmark_parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.command == "algorithms":
        from .experiments.factory import default_factory
        print("Implemented: " + ", ".join(default_factory().available_ids()))
        print("Not implemented: llm, huim, FastKAN, S-KAN")
    elif args.command == "run":
        from .experiments.contracts import ExperimentConfig
        from .experiments.runner import ExperimentRunner
        try:
            ExperimentRunner().run(ExperimentConfig.from_json(resolve_config_path(args.config)))
        except (ValueError, NotImplementedError, FileExistsError) as error:
            parser.error(str(error))
    elif args.command == "prepare":
        scripts = ["verify_splits.py"] if args.verify_only else ["inspect_data.py", "make_splits.py", "verify_splits.py"]
        for name in scripts:
            subprocess.run([sys.executable, str(ROOT / "docs/lessons/chang_05" / name)], check=True)
    elif args.command == "check":
        import torch
        from .algorithms.kan.dataset import FingerprintDataset
        config = read_config(args.config)
        device = configure_runtime(42, args.device, config["training"]["threads"])
        bundle = load_bundle(ROOT / config["split_dir"])
        features = FingerprintDataset.from_split(bundle.split("train", limit=2)).features.to(device)
        counts = {}
        for name in ["kan_mixer"]:
            current = dict(config)
            current["model"] = name
            model = build_model(current, len(bundle.classes)).to(device).eval()
            with torch.inference_mode():
                output = model(features)
            if output["logits"].shape != (2, 13) or output["xy"].shape != (2, 2):
                raise ValueError("Unexpected output shape.")
            counts[name] = parameter_count(model)
            print(f"{name}: parameters={counts[name]}, input={list(features.shape)}, "
                  f"logits={list(output['logits'].shape)}, xy={list(output['xy'].shape)}")
        print("PASS: data integrity and forward shapes.")
    elif args.command == "train":
        from .algorithms.kan.train import train
        config = read_config(args.config)
        config["model"] = args.model
        train(config, args.run_dir, args.seed, args.device, args.epochs, args.smoke)
    elif args.command == "predict":
        from .algorithms.kan.predict import predict
        predict(args.checkpoint, args.csv, args.output, args.device)
    elif args.command == "evaluate":
        from .algorithms.kan.predict import evaluate
        evaluate(args.checkpoint, args.split_dir, args.split, args.output_dir, args.device)
    elif args.command == "benchmark":
        from .algorithms.kan.benchmark import benchmark
        benchmark(args.checkpoint, args.csv, args.output, args.batch_size, args.warmup, args.repeats, args.device)


if __name__ == "__main__":
    main()
