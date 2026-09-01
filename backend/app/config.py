from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    supabase_url: str = ""
    supabase_publishable_key: str = ""
    supabase_secret_key: str = ""
    auth_disabled: bool = False
    local_admin_user_id: str = ""
    allowed_origins: str = "http://localhost:3000"

    tc2_host: str = ""
    tc2_user: str = ""
    tc2_port: int = 22
    tc2_remote_root: str = "/scratch-share/<username>/sam-football"
    local_job_root: Path = Path(".cache/jobs")
    poll_interval_seconds: int = 10

    @property
    def origins(self) -> list[str]:
        return [origin.strip() for origin in self.allowed_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
