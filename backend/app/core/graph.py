import os
import pathlib
import warnings
from typing import Annotated, Optional

from typing_extensions import TypedDict
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config.settings import settings
from app.core.technical import analyze_stock_technical
from app.core.news import analyze_stock_news
from app.core.fundamental import analyze_stock_fundamentals

load_dotenv()

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"

# Bug #2: Only configure LangSmith when key is actually present (no None assignment)
if settings.LANGSMITH_API_KEY:
    os.environ["LANGSMITH_API_KEY"] = settings.LANGSMITH_API_KEY
    os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "TradingAgent")
    os.environ["LANGSMITH_TRACING"] = "true" if settings.LANGSMITH_TRACING else "false"
else:
    # Make sure stale tracing flag doesn't try to ship traces with no key
    os.environ["LANGSMITH_TRACING"] = "false"

GEMINI_API_KEY = settings.GEMINI_API_KEY
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    api_key=GEMINI_API_KEY,
    temperature=0.3
)

# System message for the trading agent
SYSTEM_MESSAGE = """You are an expert stock trading analyst with access to three powerful analysis tools:

1. **technical_analyst**: Provides comprehensive technical analysis including indicators, trends, and trading signals
2. **fundamental_analyst**: Analyzes financial health, valuation metrics, and investment scores  
3. **news_analyst**: Evaluates news sentiment and market impact analysis

Your role:
- Always call the appropriate analysis tools based on user requests
- Provide clear, actionable insights combining tool results
- Explain complex financial concepts in understandable terms
- Give balanced perspectives considering multiple analysis types
- Format responses in a professional, structured manner

When users ask about stocks, automatically determine which analysis tools to use and provide comprehensive insights."""

# Define State with analysis tracking
class State(TypedDict):
    messages: Annotated[list, add_messages]
    technical_analysis: Optional[str]
    fundamental_analysis: Optional[str] 
    news_analysis: Optional[str]
    current_ticker: Optional[str]

# Create tools from analysis functions
@tool
def technical_analyst(ticker: str, period: str = "1y") -> str:
    """
    Perform comprehensive technical analysis on a stock.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
        period: Analysis period ('1y', '2y', '6mo', etc.)
    
    Returns:
        Comprehensive technical analysis report
    """
    return analyze_stock_technical(ticker, period)

@tool
def fundamental_analyst(ticker: str) -> str:
    """
    Perform comprehensive fundamental analysis on a stock.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
    
    Returns:
        Comprehensive fundamental analysis report
    """
    return analyze_stock_fundamentals(ticker)

@tool
def news_analyst(ticker: str, days_back: int = 7) -> str:
    """
    Perform comprehensive news sentiment analysis on a stock.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
        days_back: Number of days to look back for news (default: 7)
    
    Returns:
        Comprehensive news analysis report
    """
    return analyze_stock_news(ticker, days_back)

# Bug #8: persistent checkpoint store (SQLite) — survives restarts, zero infra
# https://langchain-ai.github.io/langgraph/concepts/persistence/
try:
    from langgraph.checkpoint.sqlite import SqliteSaver
    _db_path = pathlib.Path(settings.LANGGRAPH_DB_PATH)
    _db_path.parent.mkdir(parents=True, exist_ok=True)
    # `from_conn_string` is the documented constructor; check_same_thread=False
    # for FastAPI's threadpool. Use a context manager pattern compatible singleton.
    _saver_cm = SqliteSaver.from_conn_string(str(_db_path))
    memory = _saver_cm.__enter__()  # keep open for app lifetime
    print(f"✅ LangGraph SqliteSaver initialized at {_db_path}")
except Exception as e:
    print(f"⚠️  SqliteSaver unavailable ({e}); falling back to in-memory MemorySaver")
    from langgraph.checkpoint.memory import MemorySaver
    memory = MemorySaver()

tools = [technical_analyst, fundamental_analyst, news_analyst]
llm_with_tools = llm.bind_tools(tools)

## Node definition
def tool_calling_llm(state: State):
    messages = state["messages"]
    
    # Add system message if not present
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM_MESSAGE)] + messages
    
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

## Graph
builder = StateGraph(State)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode(tools))

## Add Edges
builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges(
    "tool_calling_llm",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition
)
builder.add_edge("tools", "tool_calling_llm")

## Compile the graph
graph = builder.compile(checkpointer=memory)

# Simple execution for testing
if __name__ == "__main__":
    print("🚀 Trading Agent Graph Initialized Successfully!")
    print("For interactive chatbot, run: python chatbot.py")
    user_query = input("Enter your query (e.g., 'Analyze AAPL fundamentals'): ")
    # Simple test
    try:
        response = graph.invoke(
            {"messages": [HumanMessage(content=user_query)]},
            config={"configurable": {"thread_id": "test"}}
        )
        print("✅ Graph test successful!")
        print("Response preview:", response["messages"][-1].content[:] + "...")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("Please install dependencies: pip install langgraph langchain-core langchain-google-genai")