from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import os
import warnings
from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI
import dotenv

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '2'

os.environ["LANGSMITH_PROJECT"] = "TradingAgent"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"

# Import the analysis modules
from Technical_Analyst import analyze_stock_technical
from News_Analyst import analyze_stock_news
from Fundamentals import analyze_stock_fundamentals

dotenv.load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

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