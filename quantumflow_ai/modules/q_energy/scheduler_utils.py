# scheduler_utils.py

def normalize_energy_profile(energy_profile: dict) -> dict:
    """
    Normalize energy values to [0,1] range
    """
    max_val = max(energy_profile.values())
    return {k: v / max_val for k, v in energy_profile.items()}
