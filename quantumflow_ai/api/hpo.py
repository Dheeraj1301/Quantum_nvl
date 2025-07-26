from fastapi import APIRouter, Body
from pydantic import BaseModel
from typing import Optional, Dict
from quantumflow_ai.modules.q_hpo.hybrid_optimizer import HybridHPO
from quantumflow_ai.modules.q_hpo.search_space import get_default_search_space
from quantumflow_ai.modules.q_hpo.meta_lstm_predictor import MetaLSTMPredictor
from quantumflow_ai.modules.q_hpo.warm_start import get_warm_start_config
from quantumflow_ai.modules.q_hpo.context_embedder import build_input_vector
from quantumflow_ai.modules.q_hpo.vqc_regressor import VQCLossRegressor
from quantumflow_ai.modules.q_hpo.gumbel_sampler import GumbelSoftmaxSampler
from quantumflow_ai.modules.q_hpo.quantum_kernel_decoder import QuantumKernelDecoder

router = APIRouter()

class HPORequest(BaseModel):
    use_hybrid: bool = True
    use_meta_lstm: bool = False
    use_warm_start: Optional[str] = None  # model name
    use_kernel_decoder: bool = False
    use_gumbel: bool = False
    hardware_context: Optional[Dict] = None

@router.post("/q-hpo/optimize")
def run_q_hpo(request: HPORequest):
    # Safe-switching logic: mutually exclusive modules
    if request.use_kernel_decoder:
        request.use_hybrid = False
        request.use_meta_lstm = False

    if request.use_meta_lstm:
        request.use_hybrid = False
        request.use_kernel_decoder = False

    search_space = get_default_search_space()

    # Optional warm start config
    if request.use_warm_start:
        warm_config = get_warm_start_config(request.use_warm_start)
        search_space.update({k: [v] for k, v in warm_config.items()})

    # Optional Gumbel Sampling
    if request.use_gumbel:
        sampler = GumbelSoftmaxSampler(search_space)
        config = sampler.sample()
        return {"strategy": "gumbel_softmax", "config": config}

    # Optional Quantum Kernel Decoder
    if request.use_kernel_decoder:
        dummy_configs = [[0.001, 128, 0.2, 0.01], [0.0005, 64, 0.1, 0.02]]
        dummy_losses = [0.4, 0.25]
        decoder = QuantumKernelDecoder(dummy_configs)
        decoded_loss_pred = decoder.train(dummy_losses)
        return {"strategy": "quantum_kernel", "predicted_loss_matrix": decoded_loss_pred.tolist()}

    # Optional MetaLSTM Predictor
    if request.use_meta_lstm:
        model = MetaLSTMPredictor()
        dummy_sequence = [[[0.001, 128, 0.2, 0.01]], [[0.0005, 64, 0.1, 0.02]]]
        prediction = model.predict_next(dummy_sequence)
        return {"strategy": "meta_lstm", "predicted_next_loss": prediction}

    # Default Hybrid VQC+Surrogate
    optimizer = HybridHPO(search_space)
    best_config = optimizer.optimize()

    # Optional Hardware Context
    if request.hardware_context:
        context_vec = build_input_vector(best_config, request.hardware_context)
        return {"strategy": "hybrid_vqc", "best_config": best_config, "context_vec": context_vec}

    return {"strategy": "hybrid_vqc", "best_config": best_config}