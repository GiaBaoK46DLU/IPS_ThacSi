"""CLI chạy tại repo: python -m ips hoặc ips sau editable install."""
import argparse
import copy
import subprocess
import sys
from pathlib import Path

from .data import OUTPUT, ROOT
from .dataset import load_bundle
from .models import build_model, parameter_count
from .runtime import configure_runtime, read_config


def main(argv=None):
    parser = argparse.ArgumentParser(description="UJIIndoorLoc IPS: MLP / MLP-Mixer / RSSI-KAN-Mixer")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Audit + materialize + verify lesson-5 split")
    prepare.add_argument("--verify-only", action="store_true")
    check = commands.add_parser("check", help="Data integrity, tensor shapes, model parameter counts")
    check.add_argument("--config", type=Path, default=ROOT / "configs/base.json")
    check.add_argument("--device", default="cpu")
    train_parser = commands.add_parser("train", help="Train with internal validation only")
    train_parser.add_argument("--config", type=Path, default=ROOT / "configs/base.json")
    train_parser.add_argument("--model", choices=["mlp", "mlp_mixer", "kan_mixer"], required=True)
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
    report_parser = commands.add_parser("report", help="Audit and summarize a completed nine-run stage-7 suite")
    report_parser.add_argument("--experiment-root", type=Path, required=True)
    report_parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        scripts = ["verify_splits.py"] if args.verify_only else ["inspect_data.py", "make_splits.py", "verify_splits.py"]
        for name in scripts:
            subprocess.run([sys.executable, str(ROOT / "docs/lessons/chang_05" / name)], check=True)
    elif args.command == "check":
        import torch
        config = read_config(args.config)
        device = configure_runtime(42, args.device, config["training"]["threads"])
        bundle = load_bundle(ROOT / config["split_dir"])
        features = bundle.dataset("train", limit=2).features.to(device)
        counts = {}
        for name in ["mlp", "mlp_mixer", "kan_mixer"]:
            current = copy.deepcopy(config)
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
        from .train import train
        config = read_config(args.config)
        config["model"] = args.model
        train(config, args.run_dir, args.seed, args.device, args.epochs, args.smoke)
    elif args.command == "predict":
        from .predict import predict
        predict(args.checkpoint, args.csv, args.output, args.device)
    elif args.command == "evaluate":
        from .predict import evaluate
        evaluate(args.checkpoint, args.split_dir, args.split, args.output_dir, args.device)
    elif args.command == "benchmark":
        from .benchmark import benchmark
        benchmark(args.checkpoint, args.csv, args.output, args.batch_size, args.warmup, args.repeats, args.device)
    elif args.command == "report":
        from .analysis import report
        report(args.experiment_root, args.output_dir)


if __name__ == "__main__":
    main()
