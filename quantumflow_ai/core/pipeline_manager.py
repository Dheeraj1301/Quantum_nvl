# Auto-generated stub
from .logger import get_logger
from typing import Callable, Dict

class PipelineManager:
    def __init__(self):
        self.logger = get_logger("PipelineManager")
        self.modules: Dict[str, Callable] = {}

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
