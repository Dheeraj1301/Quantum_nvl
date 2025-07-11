# Auto-generated stub
from pydantic import BaseSettings

class Settings(BaseSettings):
    quantum_backend: str = "default.qubit"
    use_gpu: bool = False
    log_level: str = "INFO"

    class Config:
        env_file = ".env"

settings = Settings()
