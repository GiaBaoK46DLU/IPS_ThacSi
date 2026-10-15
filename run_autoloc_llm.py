"""AutoLoC LLM Indoor Positioning Experiment Runner.

Replicates the AutoLoC paper methodology with local LLM (Ollama):
- Evaluates data scarcity: 20%, 30%, 40%, 50% training reference memory.
- Evaluates 300 random test queries.
- Computes Building Acc (%), Floor Acc (%), Combined Acc (%), SpaceID Acc (%), and Mean 2D Error (meters).
- Prints a clean summary comparison table.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# Add src to python path if needed
sys.path.insert(0, str(Path(__file__).parent))

from src.ips.algorithms.llm.autoloc import AutoLoCEngine, ScarcityResult, compute_metrics


def load_datasets(
    data_dir: Path,
    test_source: str = "heldout",
    test_samples: int = 300,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Load training and test datasets, isolating WAP columns and target metadata."""
    train_path = data_dir / "trainingData.csv"
    val_path = data_dir / "validationData.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found at {train_path}")

    full_train_df = pd.read_csv(train_path)
    wap_cols = [c for c in full_train_df.columns if c.startswith("WAP")]

    if test_source == "validation":
        if not val_path.exists():
            raise FileNotFoundError(f"Validation data not found at {val_path}")
        test_df = pd.read_csv(val_path)
        # Sample test queries
        if len(test_df) > test_samples:
            test_df = test_df.sample(n=test_samples, random_state=seed).reset_index(drop=True)
        reference_pool_df = full_train_df.copy()
    else:
        # Held-out split from trainingData to preserve ground-truth SpaceID
        rng = np.random.RandomState(seed)
        shuffled_indices = rng.permutation(len(full_train_df))
        test_idx = shuffled_indices[:test_samples]
        train_pool_idx = shuffled_indices[test_samples:]

        test_df = full_train_df.iloc[test_idx].copy().reset_index(drop=True)
        reference_pool_df = full_train_df.iloc[train_pool_idx].copy().reset_index(drop=True)

    return reference_pool_df, test_df, wap_cols


def run_scarcity_experiments(
    reference_pool_df: pd.DataFrame,
    test_df: pd.DataFrame,
    wap_cols: List[str],
    ratios: List[float],
    model_name: str = "llama3.2:1b",
    ollama_url: str = "http://localhost:11434/api/generate",
    max_workers: int = 4,
    seed: int = 42,
) -> List[ScarcityResult]:
    """Execute AutoLoC across each memory scarcity split ratio."""
    results: List[ScarcityResult] = []
    total_ratios = len(ratios)

    print("\n" + "=" * 80)
    print(f"  AUTOLOC EXPERIMENT: DATA SCARCITY BENCHMARK")
    print(f"  Model: {model_name} | Backend: Ollama ({ollama_url})")
    print(f"  Test Queries: {len(test_df)} samples | Parallel Workers: {max_workers}")
    print(f"  Memory Scarcity Ratios: {[f'{int(r*100)}%' for r in ratios]}")
    print("=" * 80 + "\n")

    X_test_raw = test_df[wap_cols].values

    for idx, ratio in enumerate(ratios, 1):
        t0 = time.time()
        # Sample exact percentage of reference pool
        n_train = int(len(reference_pool_df) * ratio)
        memory_df = reference_pool_df.sample(n=n_train, random_state=seed + int(ratio * 100)).reset_index(drop=True)

        print(f"[{idx}/{total_ratios}] Running Memory Split: {int(ratio*100)}% ({n_train:,} reference fingerprints)...")

        # Initialize AutoLoC Engine
        engine = AutoLoCEngine(
            n_components=15,
            n_neighbors=5,
            missing_val=100.0,
            replace_val=-105.0,
            model_name=model_name,
            ollama_url=ollama_url,
            seed=seed,
        )

        # 1. Preprocessing & Latent Memory Projection
        t_fit = time.time()
        engine.fit_memory(memory_df, wap_cols=wap_cols)
        fit_sec = time.time() - t_fit

        # 2. RAG Retrieval & LLM Reasoning
        print(f"       Memory fitted in {fit_sec:.2f}s. Evaluating {len(X_test_raw)} queries via {model_name}...")

        def _progress(done, total):
            if done % 50 == 0 or done == total:
                pct = (done / total) * 100
                print(f"       -> Evaluated {done}/{total} queries ({pct:.0f}%)...", flush=True)

        t_infer = time.time()
        predictions = engine.predict_batch(
            X_test_raw,
            max_workers=max_workers,
            on_progress=_progress,
        )
        infer_sec = time.time() - t_infer

        # 3. Calculate Metrics
        metrics = compute_metrics(predictions, test_df)
        total_sec = time.time() - t0

        res = ScarcityResult(
            split_ratio=ratio,
            train_size=n_train,
            test_size=len(test_df),
            building_acc=metrics["building_acc"],
            floor_acc=metrics["floor_acc"],
            combined_acc=metrics["combined_acc"],
            space_acc=metrics["space_acc"],
            mean_error_meters=metrics["mean_error_meters"],
            median_error_meters=metrics["median_error_meters"],
            fallback_rate=metrics["fallback_rate"],
            total_time_sec=total_sec,
        )
        results.append(res)

        print(f"       Done in {total_sec:.1f}s | Building Acc: {res.building_acc:.2f}% | Floor Acc: {res.floor_acc:.2f}% | Combined: {res.combined_acc:.2f}% | Mean Error: {res.mean_error_meters:.2f}m\n")

    return results


def print_summary_table(results: List[ScarcityResult], model_name: str) -> str:
    """Format and print a markdown summary table."""
    headers = [
        "Memory Ratio",
        "Ref Size",
        "Building Acc (%)",
        "Floor Acc (%)",
        "Combined Acc (%)",
        "SpaceID Acc (%)",
        "Mean 2D Error (m)",
        "Median Error (m)",
        "Fallback (%)",
        "Time (s)",
    ]

    lines = []
    lines.append(f"\n### AutoLoC Indoor Positioning Benchmark Results (LLM: {model_name})")
    lines.append("")
    header_str = "| " + " | ".join(headers) + " |"
    sep_str = "| " + " | ".join(["---"] * len(headers)) + " |"
    lines.append(header_str)
    lines.append(sep_str)

    for r in results:
        row = [
            f"{int(r.split_ratio * 100)}%",
            f"{r.train_size:,}",
            f"{r.building_acc:.2f}%",
            f"{r.floor_acc:.2f}%",
            f"{r.combined_acc:.2f}%",
            f"{r.space_acc:.2f}%" if r.space_acc > 0 else "N/A (masked)",
            f"**{r.mean_error_meters:.2f} m**",
            f"{r.median_error_meters:.2f} m",
            f"{r.fallback_rate:.1f}%",
            f"{r.total_time_sec:.1f}s",
        ]
        lines.append("| " + " | ".join(row) + " |")

    table_text = "\n".join(lines)
    print(table_text)
    return table_text


def save_results(results: List[ScarcityResult], output_path: Path, model_name: str) -> None:
    """Save results as JSON for record-keeping."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model_name,
        "results": [
            {
                "split_ratio": r.split_ratio,
                "train_size": r.train_size,
                "test_size": r.test_size,
                "building_acc": r.building_acc,
                "floor_acc": r.floor_acc,
                "combined_acc": r.combined_acc,
                "space_acc": r.space_acc,
                "mean_error_meters": r.mean_error_meters,
                "median_error_meters": r.median_error_meters,
                "fallback_rate": r.fallback_rate,
                "total_time_sec": r.total_time_sec,
            }
            for r in results
        ],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="AutoLoC Local LLM Indoor Positioning Experiment")
    parser.add_argument("--data_dir", type=str, default="data/raw/ujiindoorloc", help="Path to UJIIndoorLoc dataset directory")
    parser.add_argument("--model", type=str, default="llama3.2:1b", help="Ollama model name (e.g., llama3.2:1b, phi3:mini)")
    parser.add_argument("--ollama_url", type=str, default="http://localhost:11434/api/generate", help="Ollama API generate endpoint")
    parser.add_argument("--test_source", type=str, choices=["heldout", "validation"], default="heldout", help="Test source: heldout (has valid SpaceID) or validation")
    parser.add_argument("--sample_size", type=int, default=300, help="Number of random test queries to evaluate (default: 300)")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel request workers for Ollama")
    parser.add_argument("--ratios", type=float, nargs="+", default=[0.2, 0.3, 0.4, 0.5], help="Memory scarcity ratios (default: 0.2 0.3 0.4 0.5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default="docs/autoloc_llm_benchmark.json", help="Path to save JSON benchmark output")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    ref_df, test_df, wap_cols = load_datasets(
        data_dir=data_dir,
        test_source=args.test_source,
        test_samples=args.sample_size,
        seed=args.seed,
    )

    results = run_scarcity_experiments(
        reference_pool_df=ref_df,
        test_df=test_df,
        wap_cols=wap_cols,
        ratios=args.ratios,
        model_name=args.model,
        ollama_url=args.ollama_url,
        max_workers=args.workers,
        seed=args.seed,
    )

    print_summary_table(results, model_name=args.model)
    save_results(results, Path(args.output), model_name=args.model)


if __name__ == "__main__":
    main()

