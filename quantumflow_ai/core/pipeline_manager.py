# Auto-generated stub
from .logger import get_logger
from typing import Callable, Dict, Any, List

from .qadms_selector import select_module
from .failure_predictor import predict_retry
from .workflow_optimizer import optimize_order
from .meta_rl_controller import controller as rl_controller
from .qae_preprocessor import normalize_payload
from .cross_module_attention import CrossModuleAttention

class PipelineManager:
    def __init__(self):
        self.logger = get_logger("PipelineManager")
        self.modules: Dict[str, Callable] = {}
        self.attention = CrossModuleAttention()

    def register(self, name: str, module_callable: Callable):
        self.logger.info(f"Registering module: {name}")
        self.modules[name] = module_callable

    def run(self, name: str, *args, **kwargs):
        if name in self.modules:
            self.logger.info(f"Running module: {name}")
            return self.modules[name](*args, **kwargs)
        else:
            self.logger.error(f"Module {name} not found")
            raise ValueError(f"Module {name} not found")

    # ------------------------------------------------------------------
    # Hybrid quantum/classical orchestration
    # ------------------------------------------------------------------
    def run_pipeline(
        self,
        modules: List[Dict[str, Any]],
        payload: Dict[str, Any],
        metadata: Dict[str, Any] | None = None,
        *,
        use_attention: bool = False,
    ) -> Dict[str, Any]:
        """Run a list of modules with advanced selection and fallback."""

        metadata = metadata or {}

        try:
            payload = normalize_payload(payload)
        except Exception as exc:  # pragma: no cover - best effort
            self.logger.warning(f"Payload normalisation failed: {exc}")

        ordered = optimize_order(modules)
        results: Dict[str, Any] = {}
        attention_modules: List[str] = []
        rewards: List[float] = []

        for mod in ordered:
            name = mod["name"] if isinstance(mod, dict) else str(mod)

            sel_name = select_module(name, metadata, self.modules)

            override = rl_controller.recommend_module(sel_name, metadata)
            if override and override in self.modules:
                sel_name = override

            features = [
                float(metadata.get("latency", 0.0)),
                float(metadata.get("queue_size", 0.0)),
                float(metadata.get("input_size", 0.0)),
            ]
            retry_score = predict_retry(features)

            try:
                result = self.run(sel_name, payload)
                results[name] = result
                reward = 1.0
            except Exception as exc:  # pragma: no cover - runtime errors
                self.logger.error(f"Module {sel_name} failed: {exc}")
                reward = -1.0
                if retry_score > 0.5:
                    fallback = sel_name.replace("_quantum", "_classical")
                    if fallback != sel_name and fallback in self.modules:
                        try:
                            result = self.run(fallback, payload)
                            results[name] = result
                            reward = 0.5
                        except Exception as exc2:  # pragma: no cover
                            self.logger.error(f"Fallback {fallback} failed: {exc2}")
                            results[name] = {"error": str(exc2)}
                    else:
                        results[name] = {"error": str(exc)}
                else:
                    results[name] = {"error": str(exc)}

            rl_controller.record_outcome(sel_name, metadata, reward)
            attention_modules.append(sel_name)
            rewards.append(reward)

        if use_attention:
            try:
                self.attention.enabled = True
                self.attention.log_sequence(attention_modules, rewards)
            except Exception as exc:  # pragma: no cover - optional
                self.logger.debug(f"Attention logging failed: {exc}")

        return results
