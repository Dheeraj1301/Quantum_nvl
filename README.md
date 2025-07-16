# Quantum_nvl

This project contains a lightweight demonstration of quantum-inspired modules for token routing and energy scheduling. It includes simple training utilities and a small FastAPI application.

## Running tests

```bash
pytest -q
```

## Training models

A command line interface is available via the package entry point. To see the options run:

```bash
python -m quantumflow_ai --help
```

For example, to train the GNN energy predictor:

```bash
python -m quantumflow_ai train-gnn
```

To train the meta scheduler model:

```bash
python -m quantumflow_ai train-meta
```
