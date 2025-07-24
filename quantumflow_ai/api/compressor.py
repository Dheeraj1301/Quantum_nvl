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
    torch = None

from quantumflow_ai.core.logger import get_logger

logger = get_logger("CompressorAPI")


def validate_compression_config(config: dict) -> dict:
    """Sanitize compression settings and resolve conflicting modes."""
    cfg = config.copy()

    compression_mode = cfg.get("compression_mode", "qml")

    if compression_mode == "hybrid":
        if cfg.get("use_denoiser"):
            logger.info("Switched off use_denoiser due to hybrid mode")
            cfg["use_denoiser"] = False
        if cfg.get("use_dropout"):
            logger.info("Switched off use_dropout due to hybrid mode")
            cfg["use_dropout"] = False

    if cfg.get("enable_pruning") and compression_mode != "qml":
        logger.info("enable_pruning forces compression_mode to 'qml'")
        cfg["compression_mode"] = "qml"

    if cfg.get("predict_compressibility"):
        # Ensure classification is triggered first
        if not cfg.get("predict_first"):
            logger.info("predict_compressibility enabled; running classifier before compression")
            cfg["predict_first"] = True

    return cfg

def read_csv_as_array(file: UploadFile) -> np.ndarray:
    """Load an uploaded CSV into a numeric numpy array."""
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
    use_dropout: bool = False,
    dropout_prob: float = 0.0,
    enable_pruning: bool = False,
    pruning_threshold: float = 0.01,
    predict_first: bool = False,
    predict_compressibility: bool = False,
    compression_mode: str = "qml",
    config: dict | None = None,
) -> dict:
    if config:
        use_quantum = config.get("use_quantum", use_quantum)
        use_denoiser = config.get("use_denoiser", use_denoiser)
        noise = config.get("noise", noise)
        noise_level = config.get("noise_level", noise_level)
        use_dropout = config.get("use_dropout", use_dropout)
        dropout_prob = config.get("dropout_prob", dropout_prob)
        enable_pruning = config.get("enable_pruning", enable_pruning)
        pruning_threshold = config.get("pruning_threshold", pruning_threshold)
        predict_first = config.get("predict_first", predict_first)
        predict_compressibility = config.get("predict_compressibility", predict_compressibility)
        compression_mode = config.get("compression_mode", compression_mode)

    latent_qubits = 4
    noise_level = max(0.0, min(noise_level, 0.3))

    if compression_mode == "classical":
        use_quantum = False
    elif compression_mode in {"qml", "hybrid"}:
        use_quantum = True

    if predict_compressibility and not predict_first:
        predict_first = True

    predictions = None

    # Optional pre-classification
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
            logger.exception("VQC training failed; falling back to threshold labels")
            predictions = labels.tolist()

        if predict_compressibility and any(p == 0 for p in predictions):
            logger.info("[Compressor] Data deemed not compressible; skipping compression")
            return {
                "mode": "classification",
                "predictions": predictions,
            }

    # Quantum compression
    if use_quantum:
        try:
            qae = QuantumAutoencoder(
                n_qubits=data.shape[1],
                latent_qubits=latent_qubits,
                noise=noise,
                noise_level=noise_level,
            )
            weights = qae.train(data[:10], steps=50)
            compressed = qae.encode(data, weights)

            kept_qubits: list[int] | None = None
            if enable_pruning:
                kept_qubits, compressed = qae.prune_qubits(
                    compressed, threshold=pruning_threshold
                )

            # Optional denoising
            if use_denoiser:
                if TORCH_AVAILABLE:
                    denoiser = LatentDenoiser(latent_qubits)
                    denoiser.train_denoiser(compressed)
                    compressed = [
                        denoiser(torch.tensor(vec, dtype=torch.float32))
                        .detach().cpu().numpy().tolist()
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

            final_latent = len(kept_qubits) if kept_qubits is not None else latent_qubits
            compression_ratio = round(final_latent / data.shape[1], 3)
            logger.info(f"[Compressor] Quantum Loss: {q_loss:.4f}, PCA Loss: {pca_loss:.4f}, Ratio: {compression_ratio}")

            return {
                "mode": "quantum",
                "quantum_loss": float(q_loss),
                "pca_loss": float(pca_loss),
                "compression_ratio": compression_ratio,
                "compressed_vectors": [list(vec) for vec in compressed],
                "kept_qubits": kept_qubits,
                "predictions": predictions,
            }

        except Exception:
            logger.exception("Quantum compression failed; using classical fallback")

    # Classical fallback
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
