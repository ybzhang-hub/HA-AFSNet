"""Evaluate a validation-selected checkpoint without changing training files."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from ha_afsnet.trainer import main

if __name__ == "__main__":
    sys.argv.insert(1, "--eval-only")
    main()
