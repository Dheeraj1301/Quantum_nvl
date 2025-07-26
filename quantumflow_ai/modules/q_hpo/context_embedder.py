# quantumflow_ai/modules/q_hpo/context_embedder.py
import numpy as np

def build_input_vector(config, hardware_meta):
    return [
        np.log10(config["lr"]),
        config["batch_size"] / 256.0,
        config["dropout"],
        config["weight_decay"],
        hardware_meta.get("gpu_type_id", 0),
        hardware_meta.get("model_size_mb", 0) / 1000.0,
        hardware_meta.get("seq_len", 0) / 2048.0
    ]
