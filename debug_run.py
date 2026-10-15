import json
import subprocess
import shutil
import os
from pathlib import Path
import sys

config_path = Path("src/ips/algorithms/hierarchical/configs/hierarchical_k1_euclidean.json")
env = os.environ.copy()
env["PYTHONPATH"] = "src"

result = subprocess.run(
    [sys.executable, "-m", "ips.cli", "run", "--config", str(config_path)],
    env=env,
    capture_output=True,
    text=True
)
print("STDOUT:", result.stdout)
print("STDERR:", result.stderr)
