# quantumflow_ai/modules/q_hpo/warm_start.py
# quantumflow_ai/modules/q_hpo/warm_start.py

MODEL_CACHE = {
    # ✅ Open LLMs (Meta / HuggingFace / Mistral)
    "llama-2-7b":       {"lr": 2e-4, "dropout": 0.1, "batch_size": 128, "weight_decay": 0.01},
    "llama-2-13b":      {"lr": 1.5e-4, "dropout": 0.1, "batch_size": 96, "weight_decay": 0.015},
    "llama-3-70b":      {"lr": 1e-4, "dropout": 0.05, "batch_size": 48, "weight_decay": 0.02},
    "mistral-7b":        {"lr": 3e-4, "dropout": 0.15, "batch_size": 96, "weight_decay": 0.01},
    "mixtral-8x7b":      {"lr": 2.5e-4, "dropout": 0.1, "batch_size": 64, "weight_decay": 0.015},

    # ✅ Google / DeepMind
    "gemma-7b":          {"lr": 2e-4, "dropout": 0.1, "batch_size": 128, "weight_decay": 0.01},
    "gemini-1.5-pro":    {"lr": 1.2e-4, "dropout": 0.12, "batch_size": 64, "weight_decay": 0.012},

    # ✅ OpenAI Models
    "gpt-3.5-turbo":     {"lr": 1e-4, "dropout": 0.05, "batch_size": 128, "weight_decay": 0.005},
    "gpt-4":             {"lr": 8e-5, "dropout": 0.05, "batch_size": 96, "weight_decay": 0.01},

    # ✅ HuggingFace BERT / RoBERTa / T5
    "bert-base":         {"lr": 3e-5, "dropout": 0.1, "batch_size": 64, "weight_decay": 0.01},
    "bert-large":        {"lr": 2e-5, "dropout": 0.1, "batch_size": 32, "weight_decay": 0.01},
    "roberta-base":      {"lr": 1e-4, "dropout": 0.1, "batch_size": 64, "weight_decay": 0.01},
    "t5-small":          {"lr": 2e-4, "dropout": 0.1, "batch_size": 64, "weight_decay": 0.005},
    "t5-3b":             {"lr": 1.5e-4, "dropout": 0.1, "batch_size": 32, "weight_decay": 0.015},

    # ✅ NVIDIA Models (optional)
    "megatron-gpt":      {"lr": 1.5e-4, "dropout": 0.1, "batch_size": 128, "weight_decay": 0.02},

    # ✅ Default fallback
    "default":           {"lr": 1e-3, "dropout": 0.2, "batch_size": 64, "weight_decay": 0.01},
}

def get_warm_start_config(model_name: str) -> dict:
    name = model_name.lower().replace(" ", "-")
    return MODEL_CACHE.get(name, MODEL_CACHE["default"])
