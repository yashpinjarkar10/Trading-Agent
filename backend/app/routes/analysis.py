import asyncio
import logging
from datetime import datetime

from fastapi import APIRouter, HTTPException

from app.models.schemas import AnalysisRequest, AnalysisResponse
from app.utils.ticker_validator import get_valid_ticker
from app.tradingagents.single_analyst_graph import run_single_analyst

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/analysis", tags=["analysis"])


async def _run_analyst(request: AnalysisRequest, analysis_type: str) -> AnalysisResponse:
    """Shared handler — runs a single analyst in a thread so the event loop stays free."""
    try:
        validated_ticker = get_valid_ticker(request.ticker)
        logger.info("Running %s analysis for %s", analysis_type, validated_ticker)

        # run_single_analyst is synchronous (LLM + data fetching).
        # Push it to a thread so other requests are not blocked.
        result = await asyncio.to_thread(
            run_single_analyst, validated_ticker, analysis_type
        )

        return AnalysisResponse(
            success=True,
            analysis_type=analysis_type,
            ticker=request.ticker,
            timestamp=datetime.now().isoformat(),
            result=result,
        )
    except Exception as e:
        logger.exception("%s analysis failed for %s", analysis_type, request.ticker)
        raise HTTPException(
            status_code=500,
            detail=f"{analysis_type.capitalize()} analysis failed. Please try again.",
        )


@router.post("/market", response_model=AnalysisResponse)
async def market_analysis(request: AnalysisRequest):
    """Run the Market Analyst on a stock."""
    return await _run_analyst(request, "market")


@router.post("/fundamentals", response_model=AnalysisResponse)
async def fundamentals_analysis(request: AnalysisRequest):
    """Run the Fundamentals Analyst on a stock."""
    return await _run_analyst(request, "fundamentals")


@router.post("/news", response_model=AnalysisResponse)
async def news_analysis(request: AnalysisRequest):
    """Run the News Analyst on a stock."""
    return await _run_analyst(request, "news")


@router.post("/sentiment", response_model=AnalysisResponse)
async def sentiment_analysis(request: AnalysisRequest):
    """Run the Sentiment Analyst on a stock."""
    return await _run_analyst(request, "sentiment")
