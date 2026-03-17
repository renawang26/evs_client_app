from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import model_validator

class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql://evs:evs@localhost:5432/evs"
    REDIS_URL: str = "redis://localhost:6379/0"
    SECRET_KEY: str = "change-me-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 480  # 8 hours
    AUDIO_UPLOAD_DIR: str = str(Path.home() / "evs_audio")

    model_config = {"env_file": ".env"}

    @model_validator(mode="after")
    def check_secret_key(self):
        if self.SECRET_KEY == "change-me-in-production" and "localhost" not in self.DATABASE_URL:
            raise ValueError(
                "SECRET_KEY must be set to a secure random string in production. "
                "Set SECRET_KEY in your .env file."
            )
        return self

settings = Settings()
