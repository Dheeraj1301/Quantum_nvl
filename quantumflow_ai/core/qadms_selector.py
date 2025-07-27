"""Quantum-Aware Dynamic Module Selector (QADMS).

This module chooses between quantum and classical variants of a module
based on simple heuristics or metadata. The heavy quantum optimisation
is represented here with a placeholder so tests do not require QML libs.
"""
from __future__ import annotations

from typing import Dict

try:
    import pennylane as qml  # noqa: F401
    QML_AVAILABLE = True
except Exception:  # pragma: no cover - optional
    QML_AVAILABLE = False


def select_module(name: str, metadata: Dict[str, float], available: Dict[str, object]) -> str:
    """Return the variant of ``name`` best suited for the metadata.

    The selector prefers quantum variants when ``latency`` and ``queue_size``
    are low and a quantum backend is available. Variants are expected to be
    registered with ``_quantum`` or ``_classical`` suffixes. If a matching
    variant is not registered the original name is returned.
    """
    latency = float(metadata.get("latency", 0.0))
    queue = float(metadata.get("queue_size", 0.0))

    use_quantum = QML_AVAILABLE and latency < 0.5 and queue < 10

    q_variant = f"{name}_quantum"
    c_variant = f"{name}_classical"

    if use_quantum and q_variant in available:
        return q_variant
    if not use_quantum and c_variant in available:
        return c_variant

    # Fallback to the base name if specific variant not registered
    return name

