import numpy as np
from .q_autoencoder import QuantumAutoencoder
from .classical_compressor import ClassicalCompressor
from quantumflow_ai.core.logger import get_logger

logger = get_logger("CompressorSimulator")

def simulate_compression(data: np.ndarray, use_quantum: bool = True):
    latent_qubits = 4
    if use_quantum:
        qae = QuantumAutoencoder(n_qubits=data.shape[1], latent_qubits=latent_qubits)
        weights = qae.train(data[:10], steps=50)
        compressed = qae.encode(data, weights)
        q_loss = qae.cost_fn(weights, data[:10])

        c = ClassicalCompressor(n_components=latent_qubits)
        c.fit(data)
        pca_loss = np.mean((data - c.inverse_transform(c.transform(data)))**2)

        compression_ratio = round(latent_qubits / data.shape[1], 3)
        logger.info(f"[SIM] Quantum Loss: {q_loss}, PCA Loss: {pca_loss}")
        logger.info(f"[SIM] Compression Ratio: {compression_ratio}")
        return compressed
    else:
        c = ClassicalCompressor(n_components=latent_qubits)
        c.fit(data)
        return c.transform(data)
