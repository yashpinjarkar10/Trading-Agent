from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from datetime import datetime

from app.models.schemas import AnalysisRequest
from app.utils.ticker_validator import get_valid_ticker
from app.core.technical import analyze_stock_technical
from app.core.news import analyze_stock_news
from app.core.fundamental import analyze_stock_fundamentals

router = APIRouter(prefix="/api/analysis", tags=["analysis"])

@router.post("/technical")
async def technical_analysis(request: AnalysisRequest):
    """Perform technical analysis on a stock"""
    try:
        validated_ticker = get_valid_ticker(request.ticker)
        print(f"🔍 Running technical analysis for {validated_ticker} (original: {request.ticker})")
        result = analyze_stock_technical(
            ticker=validated_ticker,
            period=request.period
        )
        return JSONResponse({
            "success": True,
            "analysis_type": "technical",
            "ticker": request.ticker,
            "timestamp": datetime.now().isoformat(),
            "result": result
        })
    except Exception as e:
        print(f"❌ Technical analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Technical analysis failed: {str(e)}")

@router.post("/fundamental")
async def fundamental_analysis(request: AnalysisRequest):
    """Perform fundamental analysis on a stock"""
    try:
        validated_ticker = get_valid_ticker(request.ticker)
        print(f"📊 Running fundamental analysis for {validated_ticker} (original: {request.ticker})")
        result = analyze_stock_fundamentals(ticker=validated_ticker)
        return JSONResponse({
            "success": True,
            "analysis_type": "fundamental",
            "ticker": request.ticker,
            "timestamp": datetime.now().isoformat(),
            "result": result
        })
    except Exception as e:
        print(f"❌ Fundamental analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Fundamental analysis failed: {str(e)}")

@router.post("/news")
async def news_analysis(request: AnalysisRequest):
    """Perform news sentiment analysis on a stock"""
    try:
        validated_ticker = get_valid_ticker(request.ticker)
        print(f"📰 Running news analysis for {validated_ticker} (original: {request.ticker})")
        result = analyze_stock_news(
            ticker=validated_ticker,
            days_back=request.days_back,
            max_articles=request.max_articles
        )
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
