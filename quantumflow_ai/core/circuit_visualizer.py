"""Utilities for visualizing PennyLane circuits."""
from quantumflow_ai.core.logger import get_logger

try:
    import pennylane as qml
except Exception:  # pragma: no cover - optional dep
    qml = None

try:
    import matplotlib.pyplot as plt  # pragma: no cover - heavy dep
except Exception:  # pragma: no cover - optional dep
    plt = None

logger = get_logger("CircuitVisualizer")


def draw_circuit(qnode, save_path: str | None = None):
    """Return an ASCII diagram or save an image of the given qnode.

    If ``save_path`` is provided and matplotlib is available, the circuit image
    will be saved to that location. When matplotlib is unavailable, or no path is
    given, an ASCII drawing is returned instead. Returns ``None`` if drawing
    fails or PennyLane is unavailable.
    """
    if qml is None:
        logger.warning("PennyLane not available; cannot draw circuit")
        return None

    try:
        if save_path and plt is not None:
            fig, _ = qml.draw_mpl(qnode)()
            fig.savefig(save_path)
            plt.close(fig)
            return save_path
        return qml.draw(qnode)()
    except Exception as exc:  # pragma: no cover - runtime failure
        logger.warning(f"Circuit drawing failed: {exc}")
        return None
