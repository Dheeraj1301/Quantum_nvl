import sys
from pathlib import Path

# Ensure repository root is on the import path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
