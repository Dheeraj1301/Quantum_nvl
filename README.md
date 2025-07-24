# Quantum_nvl

This project contains a lightweight demonstration of quantum-inspired modules for token routing and energy scheduling. It includes simple training utilities and a small FastAPI application.

The compression demo now supports optional depolarizing noise. When using the web UI, enable **Inject Noise** and adjust the **Noise Level** slider (0.0–0.3) to simulate noisy circuits.

## Running tests

```bash
pytest -q
```

## Training models

A command line interface is available via the package entry point. To see the options run:

```bash
python -m quantumflow_ai --help
```
You can also execute the CLI script directly from the repository:
```bash
python quantumflow_ai/__main__.py --help
```
