from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_HOST: str = "127.0.0.1"
    APP_PORT: int = 8000
    APP_WORKERS: int = 2

    # Middleware settings
    ENABLE_AUTHENTICATION: bool = True
    BASIC_USERNAME: str = "username"  # Credentials to access the application
    BASIC_PASSWORD: str = "password"

    # Environment-related settings
    CHECKPOINT_DIR: Path = Path("checkpoints")  # Path to local checkpoints

    # Behaviour settings
    DEVICE: str = "cpu"


settings = Settings()
