import os
from typing import List
from dotenv import load_dotenv

load_dotenv()

class Settings:
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    NEWS_API_KEY: str = os.getenv("NEWS_API_KEY", "")
    LANGSMITH_API_KEY: str = os.getenv("LANGSMITH_API_KEY", "")
    LANGSMITH_TRACING: bool = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"

    BACKEND_PORT: int = int(os.getenv("BACKEND_PORT", "8000"))
    CORS_ORIGINS: List[str] = [
        o.strip()
        for o in os.getenv(
            "CORS_ORIGINS",
            "http://localhost:5173,http://localhost:3000",
        ).split(",")
        if o.strip()
    ]

    # Redis (cache + future job broker / rate-limiter store)
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    REDIS_ENABLED: bool = os.getenv("REDIS_ENABLED", "true").lower() == "true"

    # Cache TTLs (seconds)
    CACHE_TTL_OHLCV: int = int(os.getenv("CACHE_TTL_OHLCV", "900"))           # 15 min
    CACHE_TTL_FUNDAMENTALS: int = int(os.getenv("CACHE_TTL_FUNDAMENTALS", "3600"))  # 1 hour
    CACHE_TTL_NEWS: int = int(os.getenv("CACHE_TTL_NEWS", "300"))             # 5 min
    CACHE_TTL_TICKER_VALIDATION: int = int(os.getenv("CACHE_TTL_TICKER_VALIDATION", "604800"))  # 7 days

    # LangGraph persistent checkpoint store (SQLite — zero infra, survives restarts)
    LANGGRAPH_DB_PATH: str = os.getenv("LANGGRAPH_DB_PATH", "data/langgraph_checkpoints.sqlite")

    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "info")

    APP_TITLE: str = "Trading Agent API"
    APP_VERSION: str = "1.0.0"

settings = Settings()
