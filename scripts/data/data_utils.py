"""Import the shared data implementation when scripts run from a checkout."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
# Windows redirected terminals may default to cp1252; these tools print Vietnamese.
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")
from ips.data import *  # noqa: F403
