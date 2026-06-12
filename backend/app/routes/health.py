from fastapi import APIRouter, Depends
from datetime import datetime

from app.config.settings import Settings, get_settings
from app.models.schemas import HealthResponse, TickersResponse

router = APIRouter(prefix="/api", tags=["health"])

POPULAR_TICKERS = [
    {"symbol": "AAPL", "name": "Apple Inc."},
    {"symbol": "MSFT", "name": "Microsoft Corporation"},
    {"symbol": "GOOGL", "name": "Alphabet Inc."},
    {"symbol": "AMZN", "name": "Amazon.com Inc."},
    {"symbol": "TSLA", "name": "Tesla Inc."},
    {"symbol": "META", "name": "Meta Platforms Inc."},
    {"symbol": "NVDA", "name": "NVIDIA Corporation"},
    {"symbol": "NFLX", "name": "Netflix Inc."},
    {"symbol": "AMD", "name": "Advanced Micro Devices"},
    {"symbol": "PYPL", "name": "PayPal Holdings Inc."}
]

@router.get("/health", response_model=HealthResponse)
async def health_check(settings: Settings = Depends(get_settings)):
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version=settings.APP_VERSION
    )

@router.get("/tickers", response_model=TickersResponse)
async def get_popular_tickers():
    """Get list of popular stock tickers"""
    return TickersResponse(tickers=POPULAR_TICKERS)
