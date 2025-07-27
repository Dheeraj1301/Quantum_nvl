# quantumflow_ai/modules/q_hpo/hybrid_optimizer.py
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from quantumflow_ai.modules.q_hpo.vqc_optimizer import VQCOptimizer
from quantumflow_ai.core.logger import get_logger

logger = get_logger("HybridHPO")

class HybridHPO:
    def __init__(self, search_space, prior_runs=None):
        self.search_space = search_space
        self.vqc = VQCOptimizer(search_space)
        self.gp = GaussianProcessRegressor(kernel=RBF(), normalize_y=True)
        self.config_history = prior_runs or []
        self.encoded_configs = []
        self.losses = []

    def encode(self, config):
        return np.array([
            np.log10(config["lr"]),
            config["batch_size"] / 256.0,
            config["dropout"],
            config["weight_decay"]
        ])

    def update_surrogate(self, config, loss):
        encoded = self.encode(config)

        if not (np.all(np.isfinite(encoded)) and np.isfinite(loss)):
            logger.warning("Skipping non-finite data point in surrogate update")
            return

        self.encoded_configs.append(encoded)
        self.losses.append(loss)

        X = np.array(self.encoded_configs)
        y = np.array(self.losses)
        mask = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
        self.gp.fit(X[mask], y[mask])

    def optimize(self, steps=10):
        best_config = None
        best_score = float("inf")

        for _ in range(steps):
            if len(self.losses) >= 5:
                pred_losses, std = self.gp.predict(np.array(self.encoded_configs), return_std=True)
                explore = np.argmax(std)
                candidate = self.vqc.decode(self.encoded_configs[explore])
                loss = self.vqc.cost(np.array(self.encoded_configs[explore]))
                logger.info(f"Using VQC to explore high-uncertainty config: {candidate}")
            else:
                candidate = self.vqc.optimize()
                loss = self.vqc.best_score

            self.update_surrogate(candidate, loss)
            if loss < best_score:
                best_score = loss
                best_config = candidate

        return best_config
