import pytest
np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")

from quantumflow_ai.modules.q_hpo.classical_optimizer import ClassicalHPO
from quantumflow_ai.modules.q_hpo.hybrid_optimizer import HybridHPO
from quantumflow_ai.modules.q_hpo.meta_lstm_predictor import MetaLSTMPredictor
from quantumflow_ai.modules.q_hpo.quantum_kernel_decoder import QuantumKernelDecoder
from quantumflow_ai.modules.q_hpo.gumbel_sampler import GumbelSoftmaxSampler
from quantumflow_ai.modules.q_hpo.search_space import get_default_search_space
from quantumflow_ai.modules.q_hpo.context_embedder import build_input_vector
from quantumflow_ai.modules.q_hpo.warm_start import get_warm_start_config
from quantumflow_ai.modules.q_hpo.vqc_optimizer import VQCOptimizer
from quantumflow_ai.modules.q_hpo.vqc_regressor import VQCLossRegressor


SEARCH = {
    "lr": [1e-3, 1e-2],
    "batch_size": [32, 64],
    "dropout": [0.1, 0.2],
    "weight_decay": [0.0, 0.1],
}


def test_classical_hpo_returns_config():
    opt = ClassicalHPO(SEARCH, trials=2)
    result = opt.optimize()
    assert set(result).issuperset(SEARCH)


def test_hybrid_hpo_encode_update():
    hhpo = HybridHPO(SEARCH)
    cfg = SEARCH
    loss = 1.0
    hhpo.update_surrogate(cfg, loss)
    assert hhpo.encoded_configs


def test_meta_lstm_predictor():
    model = MetaLSTMPredictor()
    seq = np.zeros((1, 1, 4))
    out = model.predict_next(seq)
    assert isinstance(out, float)


def test_quantum_kernel_decoder_matrix():
    decoder = QuantumKernelDecoder([[0.0, 1.0], [1.0, 0.0]])
    km = decoder.compute_kernel_matrix()
    assert km.shape == (2, 2)


def test_gumbel_sampler_sample():
    sampler = GumbelSoftmaxSampler({"a": [1, 2]})
    result = sampler.sample()
    assert "a" in result


def test_get_default_search_space():
    sp = get_default_search_space()
    assert "lr" in sp


def test_build_input_vector():
    cfg = {
        "lr": 1e-3,
        "batch_size": 32,
        "dropout": 0.1,
        "weight_decay": 0.0,
    }
    meta = {"gpu_type_id": 1, "model_size_mb": 10, "seq_len": 100}
    vec = build_input_vector(cfg, meta)
    assert len(vec) > 0


def test_get_warm_start_config():
    cfg = get_warm_start_config("llama-2-7b")
    assert "lr" in cfg


def test_vqc_optimizer_fallback():
    opt = VQCOptimizer(SEARCH, max_iter=1)
    result = opt.optimize()
    assert set(result) >= set(SEARCH)


def test_vqc_regressor_build_qnode():
    qml = pytest.importorskip("pennylane")
    reg = VQCLossRegressor()
    qnode = reg.build_qnode()
    res = qnode(np.zeros(4))
    assert isinstance(res, float)

