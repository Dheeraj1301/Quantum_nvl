# train_meta_model.py

import json
import sys
from pathlib import Path

# Allow running this script directly from the notebooks folder
# We need the repository root (two levels up) on the Python path so that the
# ``quantumflow_ai`` package can be imported when this file is executed as a
# standalone script.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quantumflow_ai.modules.q_energy.meta_scheduler import MetaScheduler

def train_meta_from_profiles(
    profiles,
    data_dir=None,
):
    """Train the meta scheduling model from hardware profiles.

    Parameters
    ----------
    profiles : list[str]
        Names of the hardware profile JSON files (without extension).
    data_dir : str | Path | None, optional
        Directory containing the ``*.json`` profile data.  If ``None``, the
        directory adjacent to this script named ``profiles`` will be used.  A
        ``RuntimeError`` is raised if no training data can be loaded.
    """

    if data_dir is None:
        data_dir = Path(__file__).resolve().parent / "profiles"
    else:
        data_dir = Path(data_dir)
    all_X, all_y = [], []

    for profile in profiles:
        path = data_dir / f"{profile}.json"
        if not path.exists():
            print(f"⚠️ Skipping {profile}, file not found.")
            continue
        with open(path) as f:
            samples = json.load(f)
        for d in samples:
            all_X.append(d["features"])
            cost = d["cost"]
            # Basic heuristic
            if cost < 25:
                all_y.append("qbm")
            elif cost < 60:
                all_y.append("classical")
            else:
                all_y.append("qaoa")

    if not all_X:
        raise RuntimeError("No training data found.")

    meta = MetaScheduler()
    meta.train(all_X, all_y)
    print("✅ Meta strategy model trained and saved.")


if __name__ == "__main__":
    train_meta_from_profiles(["a100", "h100", "gb200"])
