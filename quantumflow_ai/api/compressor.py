from fastapi import UploadFile
import pandas as pd
import numpy as np
from quantumflow_ai.modules.q_compression.q_autoencoder import QuantumAutoencoder
from quantumflow_ai.modules.q_compression.classical_compressor import ClassicalCompressor
from quantumflow_ai.modules.q_compression.denoiser import LatentDenoiser, TORCH_AVAILABLE
from quantumflow_ai.modules.q_compression.vqc_classifier import VQCClassifier
try:
    import torch
except Exception:
    torch = None  # type: ignore

from quantumflow_ai.core.logger import get_logger

logger = get_logger("CompressorAPI")


def read_csv_as_array(file: UploadFile) -> np.ndarray:
    """Load an uploaded CSV into a numeric numpy array with error handling."""
    try:
        df = pd.read_csv(file.file)
    except Exception as e:
        logger.error(f"[Compressor] Error reading file: {e}")
        raise ValueError("Invalid CSV file format.")

    data = df.select_dtypes(include=[np.number]).to_numpy()
    if data.size == 0:
        raise ValueError("CSV does not contain numeric data.")

    logger.info(f"[Compressor] Loaded data shape: {data.shape}")
    return data


def run_compression(
    data: np.ndarray,
    use_quantum: bool = True,
    *,
    use_denoiser: bool = False,
    noise: bool = False,
    noise_level: float = 0.0,
    predict_first: bool = False,
) -> dict:
    latent_qubits = 4

    # Clip to safe ranges
    noise_level = max(0.0, min(noise_level, 0.3))

    predictions = None

    if predict_first:
        classical = ClassicalCompressor(n_components=latent_qubits)
        classical.fit(data)
        recon = classical.inverse_transform(classical.transform(data))
        errors = np.mean((data - recon) ** 2, axis=1)
        threshold = float(np.median(errors))
        labels = (errors <= threshold).astype(int)
        try:
            vqc = VQCClassifier(n_qubits=data.shape[1])
            vqc.train(list(data), labels, steps=20)
            predictions = vqc.predict(list(data))
        except Exception:
            logger.exception("VQC training failed; falling back to labels")
            predictions = labels.tolist()

    if use_quantum:
        try:
            qae = QuantumAutoencoder(
                n_qubits=data.shape[1],
                latent_qubits=latent_qubits,
                noise=noise,
                noise_level=noise_level,
                use_dropout=use_dropout,
                dropout_prob=dropout_prob,
            )
            weights = qae.train(data[:10], steps=50)
            compressed = qae.encode(data, weights)

            if use_denoiser:
                if TORCH_AVAILABLE:
                    denoiser = LatentDenoiser(latent_qubits)
                    denoiser.train_denoiser(compressed)
                    compressed = [
                        denoiser(
                            torch.tensor(vec, dtype=torch.float32)
                        ).detach().cpu().numpy().tolist()
                        for vec in compressed
                    ]
                else:
                    logger.warning("PyTorch not available; skipping denoiser")

            q_loss = qae.cost_fn(weights, data[:10])

            classical = ClassicalCompressor(n_components=latent_qubits)
            classical.fit(data)
            pca_loss = np.mean(
                (data - classical.inverse_transform(classical.transform(data))) ** 2
            )

            compression_ratio = round(latent_qubits / data.shape[1], 3)
            logger.info(
                f"[Compressor] Quantum Loss: {q_loss:.4f}, PCA Loss: {pca_loss:.4f}, Ratio: {compression_ratio}"
            )

            return {
                "mode": "quantum",
                "quantum_loss": float(q_loss),
                "pca_loss": float(pca_loss),
                "compression_ratio": compression_ratio,
                "compressed_vectors": [list(vec) for vec in compressed],
                "predictions": predictions,
            }

        except Exception:
            logger.exception("Quantum compression failed; using classical fallback")

    classical = ClassicalCompressor(n_components=latent_qubits)
    classical.fit(data)
    compressed = classical.transform(data)
    logger.info("[Compressor] Classical compression complete.")
    return {
        "mode": "classical",
        "compression_ratio": round(latent_qubits / data.shape[1], 3),
        "compressed_vectors": compressed.tolist(),
        "predictions": predictions,
    }
