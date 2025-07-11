"""Compatibility package for tests expecting a top-level 'modules' package."""

from pathlib import Path

# Point this package's search path to 'quantumflow_ai/modules'
__path__ = [str(Path(__file__).resolve().parent.parent / "quantumflow_ai" / "modules")]

