"""Train HA-AFSNet or either control."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from ha_afsnet.trainer import main

if __name__ == "__main__":
    main()
