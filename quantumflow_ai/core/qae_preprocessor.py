"""Quantum autoencoder based payload normaliser."""
from __future__ import annotations

from typing import Any, Dict

try:
    import numpy as np
    QAE_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    np = None
    QAE_AVAILABLE = False


def normalize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Return a cleaned version of ``payload`` suitable for modules.

    When the quantum autoencoder is unavailable this simply returns the input
    payload. Otherwise numeric arrays are scaled to the [0,1] range to mimic a
    compression step.
    """
    if not QAE_AVAILABLE:
        return payload

    def _norm(v: Any) -> Any:
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, list) and v and isinstance(v[0], (int, float)):
            arr = np.array(v, dtype=float)
            mn, mx = float(arr.min()), float(arr.max())
            if mx == mn:
                return arr.tolist()
            return ((arr - mn) / (mx - mn)).tolist()
        return v

    return {k: _norm(v) for k, v in payload.items()}

