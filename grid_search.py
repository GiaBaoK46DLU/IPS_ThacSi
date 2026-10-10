import json
import subprocess
import shutil
import os
from pathlib import Path
import sys

configs = []
for k in [1, 3, 5, 7, 9]:
    for metric in ["euclidean", "manhattan"]:
        name = f"hierarchical_k{k}_{metric}"
        output_dir = Path("runs") / name
        if output_dir.exists():
            shutil.rmtree(output_dir)
            
        config = {
            "algorithm_id": "hierarchical",
            "seed": 42,
            "device": "cpu",
            "output_dir": str(output_dir),
            "smoke": False,
            "params": {
                "n_estimators": 50,
                "n_neighbors": k,
                "metric": metric,
                "penalty_weight": 1000.0
            }
        }
        
        config_path = Path(f"src/ips/algorithms/hierarchical/configs/{name}.json")
        with config_path.open("w") as f:
            json.dump(config, f, indent=4)
        configs.append((name, config_path, output_dir))

results = []
print("Starting Grid Search for Hierarchical (RF + KNN)...")
print("-" * 60)
print(f"{'K':<5} | {'Metric':<12} | {'Val Mean 2D':<15} | {'Val Joint Acc':<15}")
print("-" * 60)

env = os.environ.copy()
env["PYTHONPATH"] = "src"

for name, config_path, output_dir in configs:
    # Run the experiment
    result = subprocess.run(
        [sys.executable, "-m", "ips.cli", "run", "--config", str(config_path)],
        env=env,
        capture_output=True,
        text=True
    )
    
    # Read result
    result_path = output_dir / "run_result.json"
    if result_path.exists():
        with result_path.open("r") as f:
            data = json.load(f)
            val_metrics = data.get("validation_metrics", {})
            mean_2d = val_metrics.get("mean_2d_error", "N/A")
            joint_acc = val_metrics.get("joint_accuracy", "N/A")
            
            k = name.split("_")[1].replace("k", "")
            metric = name.split("_")[2]
            
            mean_2d_str = f"{mean_2d:.4f}" if isinstance(mean_2d, float) else str(mean_2d)
            acc_str = f"{joint_acc:.4%}" if isinstance(joint_acc, float) else str(joint_acc)
            
            print(f"{k:<5} | {metric:<12} | {mean_2d_str:<15} | {acc_str:<15}", flush=True)
            results.append((k, metric, mean_2d, joint_acc, config_path))
    else:
        print(f"Failed: {name} - STDERR: {result.stderr}", flush=True)

print("-" * 60)
# Find best by mean_2d
best = min(results, key=lambda x: x[2] if isinstance(x[2], float) else float('inf'))
print(f"BEST CONFIG: K={best[0]}, Metric={best[1]} -> Mean 2D Error = {best[2]:.4f}")

# Cleanup all except best
for k, metric, mean_2d, acc, config_path in results:
    if k != best[0] or metric != best[1]:
        config_path.unlink()
        
print(f"Kept best config at: {best[4]}")
