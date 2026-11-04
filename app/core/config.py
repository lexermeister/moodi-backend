from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=REPOSITORY_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: str = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    recordings_dir: Path = Path("./data/recordings")
    max_upload_mb: int = Field(default=100, gt=0)
    openai_api_key: str | None = None
    openai_transcription_model: str = "gpt-transcribe"

    @property
    def recordings_path(self) -> Path:
        if self.recordings_dir.is_absolute():
            return self.recordings_dir.resolve()
        return (REPOSITORY_ROOT / self.recordings_dir).resolve()

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
