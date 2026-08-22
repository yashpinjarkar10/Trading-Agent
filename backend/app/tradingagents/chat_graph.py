import logging
import os
import pathlib
import warnings
from typing import Annotated, Optional

from typing_extensions import TypedDict
from langchain_core.messages import SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from app.config.settings import settings
from app.tradingagents.tools import CHAT_TOOLS

logger = logging.getLogger(__name__)

# Suppress verbose grpc / glog warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"

if settings.LANGSMITH_API_KEY:
    os.environ["LANGSMITH_API_KEY"] = settings.LANGSMITH_API_KEY
    os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "TradingAgent")
    os.environ["LANGSMITH_TRACING"] = "true" if settings.LANGSMITH_TRACING else "false"
else:
    os.environ["LANGSMITH_TRACING"] = "false"

if not settings.GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

os.environ["GEMINI_API_KEY"] = settings.GEMINI_API_KEY

SYSTEM_MESSAGE = """You are an expert stock trading analyst assistant.

You can answer general finance and trading education questions directly. Use tools only
when the user asks for analysis of specific tickers or companies.

Tool routing rules:
- For broad analysis, final recommendations, buy/sell/hold, or portfolio decisions, call `run_comprehensive_analysis`.
- For technical, price action, indicators, trend, support/resistance, RSI, MACD, or volume questions, call `run_market_analysis`.
- For valuation, earnings, revenue, margins, balance sheet, cash flow, financial statements, or fundamentals questions, call `run_fundamentals_analysis`.
- For recent news, catalysts, events, headlines, macro impact, or insider activity, call `run_news_analysis`.
- For Reddit, StockTwits, social media, market mood, crowd positioning, or sentiment questions, call `run_sentiment_analysis`.
- For compare, versus, ranking, "which is better", or multiple-ticker questions, call `compare_stocks`.

When a tool returns an error-like message, explain the limitation clearly and do not
invent unavailable data. Format final answers in concise Markdown.
"""


class State(TypedDict):
    """Conversation state representing user/agent messages and active ticker."""
    messages: Annotated[list, add_messages]
    current_ticker: Optional[str]


def _init_checkpointer():
    """Initialize persistent SQLite checkpointer with in-memory fallback."""
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
        db_path = pathlib.Path(settings.LANGGRAPH_DB_PATH)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        saver_cm = SqliteSaver.from_conn_string(str(db_path))
        saver = saver_cm.__enter__()
        logger.info("LangGraph SqliteSaver initialized at %s", db_path)
        return saver
    except Exception as exc:
        logger.warning("SqliteSaver unavailable (%s); falling back to MemorySaver", exc)
        return MemorySaver()


def create_chat_graph(checkpointer=None):
    """Factory function to build and compile the Trading Chat Graph."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.7-flash",
        api_key=settings.GEMINI_API_KEY,
        temperature=0.3,
    )
    llm_with_tools = llm.bind_tools(CHAT_TOOLS)

    def tool_calling_llm(state: State):
        messages = state["messages"]
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=SYSTEM_MESSAGE)] + messages
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    builder = StateGraph(State)
    builder.add_node("tool_calling_llm", tool_calling_llm)
    builder.add_node("tools", ToolNode(CHAT_TOOLS))

    # Graph routing edges
    builder.add_edge(START, "tool_calling_llm")
    builder.add_conditional_edges(
        "tool_calling_llm",
        tools_condition,
        {
            "tools": "tools",
            END: END,
        },
    )
    builder.add_edge("tools", "tool_calling_llm")

    memory = checkpointer if checkpointer is not None else _init_checkpointer()
    return builder.compile(checkpointer=memory)


# Default compiled graph instance for direct imports
graph = create_chat_graph()

if __name__ == "__main__":
    logger.info("Trading Agent Graph Initialized Successfully!")
