import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.config.settings import settings
from app.events.routes import router as events_router
from app.routes import analysis, chat, health

logging.basicConfig(
    level=settings.LOG_LEVEL.upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    logger.info("Starting %s v%s...", settings.APP_TITLE, settings.APP_VERSION)
    yield
    logger.info("Shutting down %s...", settings.APP_TITLE)


def create_app() -> FastAPI:
    """Application factory for FastAPI app instance."""
    application = FastAPI(
        title=settings.APP_TITLE,
        version=settings.APP_VERSION,
        debug=settings.DEBUG,
        lifespan=lifespan,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=[
            "Authorization",
            "Content-Type",
            "Accept",
            "Origin",
            "X-Requested-With",
            "X-Request-ID",
        ],
        max_age=600,
    )

    # Register Routers
    application.include_router(health.router)
    application.include_router(chat.router)
    application.include_router(analysis.router)
    application.include_router(events_router)

    @application.get("/")
    async def root():
        """Root status and metadata endpoint"""
        return {
            "message": settings.APP_TITLE,
            "version": settings.APP_VERSION,
            "docs": "/docs",
        }

    return application


app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=settings.BACKEND_PORT,
        reload=settings.DEBUG,
    )
