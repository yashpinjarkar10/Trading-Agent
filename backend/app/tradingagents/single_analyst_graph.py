import os
from typing import Optional
from datetime import datetime
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from app.tradingagents.agents.utils.agent_states import AgentState
from langchain_google_genai import ChatGoogleGenerativeAI
from app.config.settings import Settings
from app.tradingagents.default_config import DEFAULT_CONFIG

from app.tradingagents.agents.analysts.market_analyst import create_market_analyst
from app.tradingagents.agents.analysts.fundamentals_analyst import create_fundamentals_analyst
from app.tradingagents.agents.analysts.news_analyst import create_news_analyst
from app.tradingagents.agents.analysts.sentiment_analyst import create_sentiment_analyst

from app.tradingagents.agents.utils.agent_utils import (
    get_stock_data,
    get_indicators,
    get_fundamentals,
    get_balance_sheet,
    get_cashflow,
    get_income_statement,
    get_news,
    get_global_news,
    get_insider_transactions
)

def run_single_analyst(ticker: str, analyst_type: str, config: Optional[dict] = None) -> str:
    """Runs a single analyst node from the TradingAgentsGraph and returns its markdown report."""
    config = config or DEFAULT_CONFIG
    
    # Initialize LLM directly
    llm = ChatGoogleGenerativeAI(
        model=config.get("quick_think_llm", "gemini-1.5-flash"),
        api_key=Settings.GEMINI_API_KEY
    )
    
    # Configure tools and analyst node
    tools = []
    analyst_node = None
    report_key = ""
    
    if analyst_type == "market":
        analyst_node = create_market_analyst(llm)
        tools = [get_stock_data, get_indicators]
        report_key = "market_report"
    elif analyst_type == "fundamentals":
        analyst_node = create_fundamentals_analyst(llm)
        tools = [get_fundamentals, get_balance_sheet, get_cashflow, get_income_statement]
        report_key = "fundamentals_report"
    elif analyst_type == "news":
        analyst_node = create_news_analyst(llm)
        tools = [get_news, get_global_news, get_insider_transactions]
        report_key = "news_report"
    elif analyst_type == "sentiment":
        analyst_node = create_sentiment_analyst(llm)
        tools = [get_news]
        report_key = "sentiment_report"
    else:
        raise ValueError(f"Unknown analyst type: {analyst_type}")
        
    tool_node = ToolNode(tools)
    
    # Build graph
    workflow = StateGraph(AgentState)
    workflow.add_node("analyst", analyst_node)
    workflow.add_node("tools", tool_node)
    
    workflow.add_edge(START, "analyst")
    
    def should_continue(state: AgentState):
        messages = state.get("messages", [])
        if messages and hasattr(messages[-1], "tool_calls") and messages[-1].tool_calls:
            return "tools"
        return END
        
    workflow.add_conditional_edges("analyst", should_continue, {"tools": "tools", END: END})
    workflow.add_edge("tools", "analyst")
    
    graph = workflow.compile()
    
    today = datetime.now().strftime("%Y-%m-%d")
    initial_state = {
        "messages": [],
        "company_of_interest": ticker,
        "trade_date": today,
        "asset_type": "stock",
        "market_report": "",
        "sentiment_report": "",
        "news_report": "",
        "fundamentals_report": ""
    }
    
    # Run the graph
    result = graph.invoke(initial_state)
    
    return result.get(report_key, "No report generated.")
