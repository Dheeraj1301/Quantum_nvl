# api/decompressor.py

import numpy as np
import io
import logging
from fastapi import APIRouter, File, UploadFile, Form
from pydantic import BaseModel

from quantumflow_ai.modules.q_decompression.qft_decoder import QFTDecoder
from quantumflow_ai.modules.q_decompression.hhl_solver import HHLSolver
from quantumflow_ai.modules.q_decompression.ae_refiner import AERefiner
from quantumflow_ai.modules.q_decompression.lstm_enhancer import LSTMEnhancer

router = APIRouter()
logger = logging.getLogger("QDecompressionAPI")

class DecompressionResult(BaseModel):
    decoded: list
    features: list
    refined: list | None = None
    amplitudes: list | None = None

@router.post("/q-decompress", response_model=DecompressionResult)
async def decompress(
    file: UploadFile = File(...),
    use_qft: bool = Form(True),
    learnable_qft: bool = Form(False),
    amplitude_estimate: bool = Form(False),
    use_hhl: bool = Form(True),
    alpha: float = Form(0.5),
    use_lstm: bool = Form(False),
    use_ae: bool = Form(False)
):
    logger.info("Starting decompression request")

    content = await file.read()
    filename = file.filename

    # Parse input: CSV or NPZ
    if filename.endswith(".csv"):
        arr = np.loadtxt(io.StringIO(content.decode()))
    elif filename.endswith(".npz"):
        arr = np.load(io.BytesIO(content))["arr_0"]
    else:
        raise ValueError("Unsupported file format")

    logger.info(f"Input shape: {arr.shape}")

    # Optional: LSTM enhancement
    if use_lstm:
        enhancer = LSTMEnhancer()
        arr = enhancer.enhance(arr)

    # QFT Decode
    qft = QFTDecoder(
        num_qubits=len(arr),
        use_quantum=use_qft,
        learnable=learnable_qft,
        amplitude_estimate=amplitude_estimate
    )
    decoded = qft.decode(arr)

    # Solve inverse problem with HHL
    hhl = HHLSolver(use_quantum=use_hhl, alpha=alpha)
    A = np.identity(len(decoded))
    features = hhl.solve(A, decoded)

    # Optional: Autoencoder post-refinement
    refined = None
    if use_ae:
        ae = AERefiner(dim=len(decoded))
        refined = ae.refine(decoded)

    # Amplitude estimation (from QFT)
    amplitudes = None
    if amplitude_estimate and isinstance(decoded, np.ndarray) and decoded.ndim == 2:
        amplitudes = decoded.tolist()
        decoded = np.argmax(decoded, axis=-1)  # convert probs to index (argmax)

    return {
        "decoded": decoded.tolist(),
        "features": features.tolist(),
        "refined": refined.tolist() if refined is not None else None,
        "amplitudes": amplitudes
    }
