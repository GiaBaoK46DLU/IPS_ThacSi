"""KAN-owned defaults and compatibility for former configuration paths."""
from pathlib import Path

from ...data import ROOT

PACKAGE_DIR = Path(__file__).resolve().parent
MODEL_CONFIG = PACKAGE_DIR / "configs/model.json"
EXPERIMENT_CONFIG = PACKAGE_DIR / "configs/experiment_smoke.json"


def resolve_config_path(path: Path) -> Path:
    path = Path(path)
    if path.is_file():
        return path
    legacy = {ROOT/"configs/kan_mixer.json": MODEL_CONFIG,
              ROOT/"configs/experiments/kan_mixer_smoke.json": EXPERIMENT_CONFIG}
    return legacy.get(path.resolve(), path)
