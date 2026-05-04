from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.config.settings import settings
from app.routes import analysis, chat, health

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
