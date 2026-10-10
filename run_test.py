import json
from pathlib import Path
import subprocess
import os
import sys

output_dir = Path("runs/hierarchical_final_test")
if output_dir.exists():
    import shutil
    shutil.rmtree(output_dir)

config = {
    "algorithm_id": "hierarchical",
    "seed": 42,
    "device": "cpu",
    "output_dir": str(output_dir),
    "smoke": False,
    "evaluation_splits": ["validation", "test"],
    "params": {
        "n_estimators": 50,
        "n_neighbors": 9,
        "metric": "manhattan",
        "penalty_weight": 1000.0
    }
}

config_path = Path("src/ips/algorithms/hierarchical/configs/hierarchical_test.json")
with config_path.open("w") as f:
    json.dump(config, f, indent=4)

env = os.environ.copy()
env["PYTHONPATH"] = "src"

print("Running final test evaluation...")
subprocess.run(
    [sys.executable, "-m", "ips.cli", "run", "--config", str(config_path)],
    env=env
)
