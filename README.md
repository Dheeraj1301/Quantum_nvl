# Quantum_nvl

This project contains a lightweight demonstration of quantum-inspired modules for token routing and energy scheduling. It includes simple training utilities and a small FastAPI application.

Some modules, such as the Q-NVLinkOpt optimizer, rely on optional quantum
machine learning libraries. These can be installed using the additional
`qml-requirements.txt` file:

```bash
pip install -r quantumflow_ai/qml-requirements.txt
```

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
