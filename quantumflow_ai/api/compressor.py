from fastapi import UploadFile
import pandas as pd
import numpy as np
from quantumflow_ai.modules.q_compression.q_autoencoder import QuantumAutoencoder
from quantumflow_ai.modules.q_compression.classical_compressor import ClassicalCompressor
from quantumflow_ai.core.logger import get_logger

logger = get_logger("CompressorAPI")

def read_csv_as_array(file: UploadFile) -> np.ndarray:
    try:
        df = pd.read_csv(file.file)
        data = df.to_numpy()
        logger.info(f"[Compressor] Loaded data shape: {data.shape}")
        return data
    except Exception as e:
        logger.error(f"[Compressor] Error reading file: {e}")
        raise ValueError("Invalid CSV file format.")

def run_compression(data: np.ndarray, use_quantum: bool = True) -> dict:
    latent_qubits = 4

    if use_quantum:
        qae = QuantumAutoencoder(n_qubits=data.shape[1], latent_qubits=latent_qubits)
        weights = qae.train(data[:10], steps=50)
        compressed = qae.encode(data, weights)
        q_loss = qae.cost_fn(weights, data[:10])

        classical = ClassicalCompressor(n_components=latent_qubits)
        classical.fit(data)
        pca_loss = np.mean((data - classical.inverse_transform(classical.transform(data)))**2)

        compression_ratio = round(latent_qubits / data.shape[1], 3)
        logger.info(f"[Compressor] Quantum Loss: {q_loss:.4f}, PCA Loss: {pca_loss:.4f}, Ratio: {compression_ratio}")

        return {
            "mode": "quantum",
            "quantum_loss": float(q_loss),
            "pca_loss": float(pca_loss),
            "compression_ratio": compression_ratio,
            "compressed_vectors": compressed,
        }

    else:
        classical = ClassicalCompressor(n_components=latent_qubits)
        classical.fit(data)
        compressed = classical.transform(data)
        logger.info("[Compressor] Classical compression complete.")
        return {
            "mode": "classical",
            "compression_ratio": round(latent_qubits / data.shape[1], 3),
            "compressed_vectors": compressed.tolist()
        }
