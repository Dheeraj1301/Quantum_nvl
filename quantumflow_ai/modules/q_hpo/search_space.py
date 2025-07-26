# quantumflow_ai/modules/q_hpo/search_space.py

def get_default_search_space() -> dict:
    return {
        "lr": [1e-5, 1e-4, 1e-3, 1e-2],
        "batch_size": [32, 64, 128, 256],
        "dropout": [0.1, 0.2, 0.3, 0.5],
        "weight_decay": [0.0, 0.01, 0.05, 0.1]
    }
