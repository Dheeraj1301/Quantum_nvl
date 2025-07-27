import pytest
np = pytest.importorskip("numpy")

from quantumflow_ai.modules.q_attention.classical_attention import classical_linear_attention
from quantumflow_ai.modules.q_attention.contrastive_trainer import contrastive_loss
from quantumflow_ai.modules.q_attention.utils import scale_and_pad
from quantumflow_ai.modules.q_attention.qka_kernel_attention import quantum_kernel_attention
from quantumflow_ai.modules.q_attention.qpe_position_encoder import quantum_positional_encoding
from quantumflow_ai.modules.q_attention.qsvd_head_pruner import prune_heads_with_qsvd
from quantumflow_ai.modules.q_attention.qaoa_sparse_attention import sample_sparse_attention
from quantumflow_ai.modules.q_attention.vqc_reweight_attention import vqc_reweight
from quantumflow_ai.modules.q_attention.hybrid_transformer_layer import HybridTransformerLayer


def test_classical_linear_attention_shape():
    q = np.ones((2, 3))
    k = np.ones((3, 3))
    v = np.ones((3, 2))
    out = classical_linear_attention(q, k, v)
    assert out.shape == (2, 2)


def test_contrastive_loss_zero():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert contrastive_loss(a, b) == 0


def test_scale_and_pad():
    arr = np.ones((2, 2))
    padded = scale_and_pad(arr, 4)
    assert padded.shape == (2, 4)


def test_quantum_kernel_attention_fallback():
    q = np.ones((2, 2))
    k = np.ones((2, 2))
    v = np.ones((2, 2))
    result = quantum_kernel_attention(q, k, v)
    assert result is None


def test_quantum_positional_encoding_length():
    vec = quantum_positional_encoding(1, 6)
    assert len(vec) == 6


def test_prune_heads_with_qsvd():
    heads = [np.eye(2) * 0.1, np.eye(2) * 1.0]
    pruned = prune_heads_with_qsvd(heads, threshold=0.5)
    assert len(pruned) == 1


def test_sample_sparse_attention_len():
    q = np.eye(3)
    k = np.eye(3)
    pairs = sample_sparse_attention(q, k, top_k=2)
    assert len(pairs) == 2


def test_vqc_reweight_fallback():
    data = np.ones((2, 4))
    result = vqc_reweight(data)
    assert np.all(result == 1)


def test_hybrid_transformer_layer_forward():
    torch = pytest.importorskip("torch")
    layer = HybridTransformerLayer(dim=4)
    x = torch.ones(2, 4)
    out = layer(x)
    assert out.shape == (2, 4)


