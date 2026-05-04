from fastapi import APIRouter
from fastapi.responses import JSONResponse
from datetime import datetime

from app.config.settings import settings

router = APIRouter(prefix="/api", tags=["health"])

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return JSONResponse({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": settings.APP_VERSION
    })

@router.get("/tickers")
async def get_popular_tickers():
    """Get list of popular stock tickers"""
    popular_tickers = [
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
    return JSONResponse({"tickers": popular_tickers})
