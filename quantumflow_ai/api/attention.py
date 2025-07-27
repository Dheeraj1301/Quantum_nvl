# quantumflow_ai/modules/q_attention/q_attention_backend.py

from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
import numpy as np

# Import modules
from quantumflow_ai.modules.q_attention import (
    qka_kernel_attention,
    vqc_reweight_attention,
    qsvd_head_pruner,
    qpe_position_encoder,
    contrastive_trainer,
    qaoa_sparse_attention,
    hybrid_transformer_layer,
    classical_attention,
    utils
)

router = APIRouter()

class AttentionInput(BaseModel):
    query: list[list[float]]
    key: list[list[float]]
    value: list[list[float]]
    use_quantum_kernel: bool = True
    use_vqc: bool = False
    use_qaoa_sampling: bool = False
    use_positional_encoding: bool = True
    contrastive_test: bool = False

@router.post("/q-attention/run")
async def run_q_attention(payload: AttentionInput):
    try:
        Q = np.array(payload.query)
        K = np.array(payload.key)
        V = np.array(payload.value)

        # Positional Encoding
        if payload.use_positional_encoding:
            for i in range(Q.shape[0]):
                Q[i] += qpe_position_encoder.quantum_positional_encoding(i, Q.shape[1])
                K[i] += qpe_position_encoder.quantum_positional_encoding(i, K.shape[1])

        # QAOA Sampling (Sparse Attention)
        if payload.use_qaoa_sampling:
            pairs = qaoa_sparse_attention.sample_sparse_attention(Q, K)
            reduced_Q = np.array([Q[i] for i, _ in pairs])
            reduced_K = np.array([K[j] for _, j in pairs])
            reduced_V = np.array([V[j] for _, j in pairs])
            Q, K, V = reduced_Q, reduced_K, reduced_V

        # Quantum Kernel Attention
        if payload.use_quantum_kernel:
            output = qka_kernel_attention.quantum_kernel_attention(Q, K, V)
            if output is None:
                output = classical_attention.classical_linear_attention(Q, K, V)
        else:
            output = classical_attention.classical_linear_attention(Q, K, V)

        # VQC Attention Reweighting
        if payload.use_vqc:
            concat_qk = np.concatenate([Q, K], axis=1)
            weights = vqc_reweight_attention.vqc_reweight(concat_qk)
            output = output * weights[:, None]

        # Contrastive Attention Testing (optional)
        if payload.contrastive_test:
            corrupted = output + np.random.normal(0, 0.01, output.shape)
            loss = contrastive_trainer.contrastive_loss(output.flatten(), corrupted.flatten())
        else:
            loss = None

        return {
            "attention_output": output.tolist(),
            "contrastive_loss": loss
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Q-Attention failed: {str(e)}")
