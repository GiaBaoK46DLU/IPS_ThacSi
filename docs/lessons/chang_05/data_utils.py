"""Compatibility wrapper: implementation lives in src/ips/data.py."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from ips.data import *  # noqa: F403
