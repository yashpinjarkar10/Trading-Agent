import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.config.settings import settings
from app.routes import analysis, chat, health

logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.APP_TITLE,
    version=settings.APP_VERSION,
    debug=settings.DEBUG
)

# Bug #13: explicit allow-list for methods and headers; never use "*" with
# allow_credentials=True (browsers reject it and it's an attack-surface anyway).
app.add_middleware(
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

app.include_router(analysis.router)
app.include_router(chat.router)
app.include_router(health.router)

# Event Map (Phase 1+) — only mounted when EVENTS_ENABLED=true to keep the
# rest of the app bootable on machines without a Supabase connection.
if settings.EVENTS_ENABLED:
    from app.events.routes import router as events_router
    app.include_router(events_router)
    logger.info("Event Map enabled — /api/events router mounted")
else:
    logger.info("Event Map disabled (set EVENTS_ENABLED=true to mount)")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Trading Agent API",
        "version": settings.APP_VERSION,
        "docs": "/docs"
    }

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=settings.BACKEND_PORT,
        reload=settings.DEBUG
    )
