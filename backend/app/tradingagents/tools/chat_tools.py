import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Literal

from langchain_core.tools import tool
from langchain_core.runnables.config import RunnableConfig

from app.tradingagents.default_config import DEFAULT_CONFIG
from app.tradingagents.graph.trading_graph import TradingAgentsGraph
from app.tradingagents.single_analyst_graph import run_single_analyst
from app.utils.ticker_validator import get_valid_ticker

ANALYST_TYPES = {"market", "fundamentals", "news", "sentiment"}
COMPARE_FOCUS_VALUES = ANALYST_TYPES | {"general"}
MAX_COMPARE_TICKERS = 4
GENERAL_COMPARE_ANALYSTS = ("market", "fundamentals")


def _normalize_ticker(ticker: str) -> str:
    """Normalize and validate a stock ticker string."""
    normalized = get_valid_ticker(ticker)
    if not normalized:
        raise ValueError("empty ticker")
    return normalized


def _run_single_analyst_report(ticker: str, analyst_type: str) -> str:
    """Run an individual analyst sub-pipeline and format the markdown report."""
    try:
        normalized = _normalize_ticker(ticker)
        report = run_single_analyst(normalized, analyst_type)
        return f"## {analyst_type.title()} Analysis for {normalized}\n\n{report}"
    except Exception as exc:
        return f"Could not run {analyst_type} analysis for {ticker}: {exc}"


def _run_compare_job(index: int, ticker: str, analyst_type: str) -> tuple[int, str, str, str]:
    """Execute a single comparison worker task."""
    normalized = ticker
    try:
        normalized = _normalize_ticker(ticker)
        report = run_single_analyst(normalized, analyst_type)
        return index, normalized, analyst_type, report
    except Exception as exc:
        return index, normalized, analyst_type, f"Could not analyze {ticker}: {exc}"


@tool
def run_comprehensive_analysis(ticker: str, config: RunnableConfig) -> str:
    """
    Run a comprehensive, multi-agent analysis on a stock ticker.
    This spawns an entire team of AI agents to debate and analyze Market, Sentiment, News, and Fundamentals.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
    
    Returns:
        Comprehensive multi-agent analysis report including final decision.
    """
    try:
        normalized = _normalize_ticker(ticker)
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        thread_id = config.get("configurable", {}).get("thread_id", "default")

        config_dict = DEFAULT_CONFIG.copy()
        config_dict["llm_provider"] = "google"
        config_dict["quick_think_llm"] = "gemini-3.7-flash"
        config_dict["deep_think_llm"] = "gemini-3.7-flash"

        trading_graph = TradingAgentsGraph(
            config=config_dict, 
            selected_analysts=["market", "social", "news", "fundamentals"]
        )
        final_state, signal = trading_graph.propagate(
            company_name=normalized, 
            trade_date=today, 
            thread_id=thread_id
        )
        
        report = f"## Comprehensive Analysis for {normalized} ({today})\n\n"
        report += f"### Final Decision: {signal}\n\n"
        report += f"**Detailed Trade Decision:**\n{final_state.get('final_trade_decision', '')}\n\n"
        
        if 'market_report' in final_state:
            report += f"### Market Report\n{final_state.get('market_report', '')}\n\n"
        if 'fundamentals_report' in final_state:
            report += f"### Fundamentals\n{final_state.get('fundamentals_report', '')}\n\n"
        if 'news_report' in final_state:
            report += f"### News\n{final_state.get('news_report', '')}\n\n"
            
        return report
    except Exception as e:
        return f"Could not analyze {ticker}: {e}"


@tool
def run_market_analysis(ticker: str) -> str:
    """
    Run a market and technical analysis for one ticker.

    Args:
        ticker: Stock ticker symbol or company name.
    """
    return _run_single_analyst_report(ticker, "market")


@tool
def run_fundamentals_analysis(ticker: str) -> str:
    """
    Run a fundamentals analysis for one ticker.

    Args:
        ticker: Stock ticker symbol or company name.
    """
    return _run_single_analyst_report(ticker, "fundamentals")


@tool
def run_news_analysis(ticker: str) -> str:
    """
    Run a recent news and catalyst analysis for one ticker.

    Args:
        ticker: Stock ticker symbol or company name.
    """
    return _run_single_analyst_report(ticker, "news")


@tool
def run_sentiment_analysis(ticker: str) -> str:
    """
    Run a social and retail sentiment analysis for one ticker.

    Args:
        ticker: Stock ticker symbol or company name.
    """
    return _run_single_analyst_report(ticker, "sentiment")


@tool
def compare_stocks(
    tickers: list[str],
    focus: Literal["general", "market", "fundamentals", "news", "sentiment"] = "general",
) -> str:
    """
    Compare multiple tickers using targeted analyst reports.

    Args:
        tickers: Ticker symbols or company names to compare. Maximum 4.
        focus: Comparison focus. Use general if the user did not specify a focus.
    """
    if not tickers:
        return "Could not compare stocks: no tickers were provided."

    selected_tickers = [ticker for ticker in tickers if ticker and ticker.strip()][:MAX_COMPARE_TICKERS]
    if not selected_tickers:
        return "Could not compare stocks: no valid tickers were provided."

    normalized_focus = (focus or "general").lower()
    if normalized_focus not in COMPARE_FOCUS_VALUES:
        normalized_focus = "general"

    analyst_types = (
        GENERAL_COMPARE_ANALYSTS
        if normalized_focus == "general"
        else (normalized_focus,)
    )
    jobs = [
        (ticker, analyst_type)
        for ticker in selected_tickers
        for analyst_type in analyst_types
    ]

    results: list[tuple[int, str, str, str]] = []
    max_workers = min(len(jobs), MAX_COMPARE_TICKERS)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_run_compare_job, index, ticker, analyst_type): (ticker, analyst_type)
            for index, (ticker, analyst_type) in enumerate(jobs)
        }
        for future in as_completed(futures):
            results.append(future.result())

    results.sort(key=lambda item: item[0])

    sections = [
        f"## Comparison Data\n\nFocus: {normalized_focus}\nTickers: {', '.join(selected_tickers)}"
    ]
    if len(tickers) > MAX_COMPARE_TICKERS:
        sections.append(f"Note: only the first {MAX_COMPARE_TICKERS} tickers were analyzed.")

    for _, normalized, analyst_type, report in results:
        sections.append(f"### {normalized} - {analyst_type.title()}\n\n{report}")

    return "\n\n".join(sections)


CHAT_TOOLS = [
    run_comprehensive_analysis,
    run_market_analysis,
    run_fundamentals_analysis,
    run_news_analysis,
    run_sentiment_analysis,
    compare_stocks,
]
