# Quantum_nvl

This project contains a lightweight demonstration of quantum-inspired modules for token routing and energy scheduling. It includes simple training utilities and a small FastAPI application.

### New Features

* **Quantum dropout** - the compression autoencoder can now randomly skip
  entangling layers during training and inference.  Enable the option through the
  frontend toggle or via the API parameters ``use_dropout`` and ``dropout_prob``.

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
