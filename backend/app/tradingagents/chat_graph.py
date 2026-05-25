import os
import pathlib
import warnings
import datetime
from typing import Annotated, Optional

from typing_extensions import TypedDict
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.runnables.config import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config.settings import settings
from app.tradingagents.graph.trading_graph import TradingAgentsGraph
from app.tradingagents.default_config import DEFAULT_CONFIG

load_dotenv()

# Suppress warnings for cleaner output
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

GEMINI_API_KEY = settings.GEMINI_API_KEY
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

# Set it in os environment for the trading agents graph to pick up
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    api_key=GEMINI_API_KEY,
    temperature=0.3
)

# System message for the trading agent
SYSTEM_MESSAGE = """You are an expert stock trading analyst assistant.
You have access to a very powerful tool: `run_comprehensive_analysis`.

Your role:
- When the user asks to analyze a stock, extract the ticker symbol and call `run_comprehensive_analysis`.
- Once you receive the comprehensive report, present it to the user. You can format it nicely using Markdown.
- If the user asks general questions, just answer them. 

The comprehensive analysis tool will automatically spawn a team of AI agents (Market, News, Sentiment, Fundamentals) to debate and give a final trading recommendation.
"""

# Define State with analysis tracking
class State(TypedDict):
    messages: Annotated[list, add_messages]
    current_ticker: Optional[str]

# New comprehensive analysis tool
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
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    thread_id = config.get("configurable", {}).get("thread_id", "default")
    
    config_dict = DEFAULT_CONFIG.copy()
    config_dict["llm_provider"] = "google"
    config_dict["quick_think_llm"] = "gemini-1.5-flash"
    config_dict["deep_think_llm"] = "gemini-1.5-flash"  # Use flash if pro is not strictly needed, but let's stick to flash for both for speed.
    
    try:
        trading_graph = TradingAgentsGraph(config=config_dict, selected_analysts=["market", "social", "news", "fundamentals"])
        final_state, signal = trading_graph.propagate(company_name=ticker, trade_date=today, thread_id=thread_id)
        
        report = f"## Comprehensive Analysis for {ticker} ({today})\n\n"
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
        return f"Error executing comprehensive analysis: {str(e)}"


try:
    from langgraph.checkpoint.sqlite import SqliteSaver
    _db_path = pathlib.Path(settings.LANGGRAPH_DB_PATH)
    _db_path.parent.mkdir(parents=True, exist_ok=True)
    _saver_cm = SqliteSaver.from_conn_string(str(_db_path))
    memory = _saver_cm.__enter__()  
    print(f"LangGraph SqliteSaver initialized at {_db_path}")
except Exception as e:
    print(f"SqliteSaver unavailable ({e}); falling back to in-memory MemorySaver")
    from langgraph.checkpoint.memory import MemorySaver
    memory = MemorySaver()

tools = [run_comprehensive_analysis]
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
    tools_condition
)
builder.add_edge("tools", "tool_calling_llm")

## Compile the graph
graph = builder.compile(checkpointer=memory)

if __name__ == "__main__":
    print("Trading Agent Graph Initialized Successfully!")