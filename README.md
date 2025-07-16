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
You can also execute the CLI script directly from the repository:
```bash
python quantumflow_ai/__main__.py --help
```
The CLI recognises the training script paths as shorthand for the
corresponding commands, so the following are equivalent:
```bash
python -m quantumflow_ai notebooks/train_meta_model.py --help
python -m quantumflow_ai train-meta --help
```

For example, to train the GNN energy predictor:

```bash
python -m quantumflow_ai train-gnn
```

To train the meta scheduler model:

```bash
python -m quantumflow_ai train-meta
```
