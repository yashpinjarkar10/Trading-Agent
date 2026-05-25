from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from datetime import datetime

from app.models.schemas import AnalysisRequest
from app.utils.ticker_validator import get_valid_ticker
from app.tradingagents.single_analyst_graph import run_single_analyst

router = APIRouter(prefix="/api/analysis", tags=["analysis"])

@router.post("/market")
async def market_analysis(request: AnalysisRequest):
    """Perform market/technical analysis on a stock"""
    try:
        validated_ticker = get_valid_ticker(request.ticker)
        print(f"🔍 Running market analysis for {validated_ticker} (original: {request.ticker})")
        result = run_single_analyst(validated_ticker, "market")
        return JSONResponse({
            "success": True,
            "analysis_type": "market",
            "ticker": request.ticker,
            "timestamp": datetime.now().isoformat(),
            "result": result
        })
    except Exception as e:
        print(f"❌ Market analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Market analysis failed: {str(e)}")

@router.post("/fundamentals")
async def fundamentals_analysis(request: AnalysisRequest):
    """Perform fundamental analysis on a stock"""
    try:
        validated_ticker = get_valid_ticker(request.ticker)
        print(f"📊 Running fundamentals analysis for {validated_ticker} (original: {request.ticker})")
        result = run_single_analyst(validated_ticker, "fundamentals")
        return JSONResponse({
            "success": True,
            "analysis_type": "fundamentals",
            "ticker": request.ticker,
            "timestamp": datetime.now().isoformat(),
            "result": result
        })
    except Exception as e:
        print(f"❌ Fundamentals analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Fundamentals analysis failed: {str(e)}")

@router.post("/news")
async def news_analysis(request: AnalysisRequest):
    """Perform news analysis on a stock"""
    try:
        validated_ticker = get_valid_ticker(request.ticker)
        print(f"📰 Running news analysis for {validated_ticker} (original: {request.ticker})")
        result = run_single_analyst(validated_ticker, "news")
        return JSONResponse({
            "success": True,
            "analysis_type": "news",
            "ticker": request.ticker,
            "timestamp": datetime.now().isoformat(),
            "result": result
        })
    except Exception as e:
        print(f"❌ News analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"News analysis failed: {str(e)}")

@router.post("/sentiment")
async def sentiment_analysis(request: AnalysisRequest):
    """Perform sentiment analysis on a stock"""
    try:
        validated_ticker = get_valid_ticker(request.ticker)
        print(f"🧠 Running sentiment analysis for {validated_ticker} (original: {request.ticker})")
        result = run_single_analyst(validated_ticker, "sentiment")
        return JSONResponse({
            "success": True,
            "analysis_type": "sentiment",
            "ticker": request.ticker,
            "timestamp": datetime.now().isoformat(),
            "result": result
        })
    except Exception as e:
        print(f"❌ Sentiment analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Sentiment analysis failed: {str(e)}")
