# Auto-generated stub
# core/config.py
from pydantic_settings import BaseSettings  # ✅ NEW

class Settings(BaseSettings):
    quantum_backend: str = "default.qubit"
    use_gpu: bool = False
    log_level: str = "INFO"

    class Config:
        env_file = ".env"

settings = Settings()
