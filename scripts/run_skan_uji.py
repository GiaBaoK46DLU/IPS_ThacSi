"""Matched V4/S-KAN UJI protocol: prepare, run, or audit (see backend README)."""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from ips.algorithms.kan.variants.selectable.experiment import main

if __name__=='__main__':
    main()
