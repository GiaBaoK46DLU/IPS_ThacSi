import json
from pathlib import Path

configs = []
for k in [1, 3, 5, 7, 9]:
    for metric in ["euclidean", "manhattan"]:
        name = f"hierarchical_k{k}_{metric}"
        configs.append(name)

results = []
print("Starting Grid Search for Hierarchical (RF + KNN)...")
print("-" * 60)
print(f"{'K':<5} | {'Metric':<12} | {'Val Mean 2D':<15} | {'Val Joint Acc':<15}")
print("-" * 60)

for name in configs:
    output_dir = Path("runs") / name
    result_path = output_dir / "run_result.json"
    if result_path.exists():
        with result_path.open("r") as f:
            data = json.load(f)
            val_metrics = data.get("metrics", {}).get("validation", {})
            mean_2d = val_metrics.get("mean_2d", "N/A")
            joint_acc = val_metrics.get("joint_accuracy", "N/A")
            
            k = name.split("_")[1].replace("k", "")
            metric = name.split("_")[2]
            
            mean_2d_str = f"{mean_2d:.4f}" if isinstance(mean_2d, float) else str(mean_2d)
            acc_str = f"{joint_acc:.4%}" if isinstance(joint_acc, float) else str(joint_acc)
            
            print(f"{k:<5} | {metric:<12} | {mean_2d_str:<15} | {acc_str:<15}")
            results.append((k, metric, mean_2d, joint_acc))
    else:
        print(f"Failed: {name}")

print("-" * 60)
best = min(results, key=lambda x: x[2] if isinstance(x[2], float) else float('inf'))
print(f"BEST CONFIG: K={best[0]}, Metric={best[1]} -> Mean 2D Error = {best[2]:.4f}")
