"""Utility helpers for non-quantum execution paths."""

from typing import Callable, Any


def with_fallback(quantum_fn: Callable[..., Any], classical_fn: Callable[..., Any]):
    """Return a wrapper that tries a quantum implementation and falls back.

    Parameters
    ----------
    quantum_fn:
        Function implementing the quantum logic. It may raise ``ImportError`` if
        the required backend is not available.
    classical_fn:
        Classical alternative used when the quantum implementation fails.
    """

    def wrapper(*args, **kwargs):
        try:
            return quantum_fn(*args, **kwargs)
        except Exception:
            return classical_fn(*args, **kwargs)

    return wrapper
