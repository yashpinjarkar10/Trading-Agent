from typing import List, Union
from pydantic import Field, AliasChoices, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    GEMINI_API_KEY: str = ""
    NEWS_API_KEY: str = ""
    LANGSMITH_API_KEY: str = ""
    LANGSMITH_TRACING: bool = False

    BACKEND_PORT: int = Field(
        default=8000, 
        validation_alias=AliasChoices("PORT", "BACKEND_PORT")
    )
    
    CORS_ORIGINS: List[str] = ["http://localhost:5173", "http://localhost:3000"]

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str):
            return [i.strip() for i in v.split(",") if i.strip()]
        return v

    # Redis (cache + future job broker / rate-limiter store)
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_ENABLED: bool = True

    # Cache TTLs (seconds)
    CACHE_TTL_OHLCV: int = 900           # 15 min
    CACHE_TTL_FUNDAMENTALS: int = 3600   # 1 hour
    CACHE_TTL_NEWS: int = 300            # 5 min
    CACHE_TTL_TICKER_VALIDATION: int = 604800  # 7 days

    # LangGraph persistent checkpoint store (SQLite — zero infra, survives restarts)
    LANGGRAPH_DB_PATH: str = "data/langgraph_checkpoints.sqlite"

    # ── Event Map (read-only Supabase client) ─────────────────────────────
    # Writes happen via /scrapers on GitHub Actions, not from the backend.
    EVENTS_ENABLED: bool = False
    SUPABASE_URL: str = ""
    SUPABASE_SERVICE_ROLE_KEY: str = ""
    SUPABASE_ANON_KEY: str = ""

    DEBUG: bool = False
    LOG_LEVEL: str = "info"

    APP_TITLE: str = "Trading Agent API"
    APP_VERSION: str = "1.0.0"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
