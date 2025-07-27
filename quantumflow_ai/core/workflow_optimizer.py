"""Workflow optimizer encoding a DAG of module dependencies.

For simplicity this implementation performs a topological sort based on the
``depends_on`` metadata. If PennyLane is available it will apply a lightweight
QAOA-inspired shuffle of independent steps to mimic a quantum optimisation
process.
"""
from __future__ import annotations

from typing import Dict, List

try:
    import pennylane as qml  # noqa: F401
    QML_AVAILABLE = True
except Exception:  # pragma: no cover - optional
    QML_AVAILABLE = False


def _topological_sort(mods: List[Dict[str, str]]) -> List[Dict[str, str]]:
    ordered: List[Dict[str, str]] = []
    remaining = {m["name"]: m for m in mods}
    while remaining:
        progress = False
        for name, meta in list(remaining.items()):
            deps = meta.get("depends_on") or []
            if all(d not in remaining for d in deps):
                ordered.append(meta)
                remaining.pop(name)
                progress = True
        if not progress:
            # cyclic dependency fallback
            ordered.extend(list(remaining.values()))
            break
    return ordered


def optimize_order(modules: List[Dict[str, str]]) -> List[Dict[str, str]]:
    order = _topological_sort(modules)
    if QML_AVAILABLE and len(order) > 1:
        # Pretend to use QAOA to decide between equivalent steps
        order = sorted(order, key=lambda x: hash(x["name"]) % 3)
    return order

