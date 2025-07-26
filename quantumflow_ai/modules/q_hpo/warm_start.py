# quantumflow_ai/modules/q_hpo/warm_start.py
MODEL_CACHE = {
    "llama-7b": {"lr": 2e-4, "dropout": 0.1, "batch_size": 128, "weight_decay": 0.01},
    "bert-base": {"lr": 3e-5, "dropout": 0.1, "batch_size": 64, "weight_decay": 0.01}
}

def get_warm_start_config(model_name: str):
    return MODEL_CACHE.get(model_name.lower(), {})
