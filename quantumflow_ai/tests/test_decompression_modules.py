import pytest
np = pytest.importorskip("numpy")

from quantumflow_ai.modules.q_decompression.hhl_solver import HHLSolver
from quantumflow_ai.modules.q_decompression.qft_decoder import QFTDecoder
from quantumflow_ai.modules.q_decompression.qsvt_solver import QSVTSolver
from quantumflow_ai.modules.q_decompression.ae_refiner import AERefiner


def test_hhl_solver_fallback():
    A = np.eye(2)
    b = np.ones(2)
    solver = HHLSolver(use_quantum=True)
    out = solver.solve(A, b)
    assert out.shape == (2,)


def test_qft_decoder_classical():
    dec = QFTDecoder(num_qubits=4, use_quantum=False)
    result = dec.decode(np.ones(4))
    assert len(result) == 4


def test_qsvt_solver_classical():
    A = np.eye(2)
    b = np.ones(2)
    solver = QSVTSolver(use_quantum=False)
    out = solver.solve(A, b)
    assert out.shape == (2,)


def test_ae_refiner_shape():
    torch = pytest.importorskip("torch")
    ref = AERefiner(dim=4)
    data = np.ones((1, 4))
    refined = ref.refine(data)
    assert refined.shape == (1, 4)


