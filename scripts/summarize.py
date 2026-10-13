"""Aggregate per-seed metrics and compute paired comparisons."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ha_afsnet.statistics import main

if __name__ == "__main__":
    main()
