from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
import os
import json
from datetime import datetime

# Import analysis modules
from Technical_Analyst import analyze_stock_technical
from News_Analyst import analyze_stock_news
from Fundamentals import analyze_stock_fundamentals
from graph import graph
from langchain_core.messages import HumanMessage

app = FastAPI(title="Trading Agent Web Interface", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except:
    # Static directory doesn't exist yet, will be created
    pass

# Pydantic models for request/response
class AnalysisRequest(BaseModel):
    ticker: str
    period: Optional[str] = "1y"
    days_back: Optional[int] = 7
    max_articles: Optional[int] = 50

class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = "default"

class ChatResponse(BaseModel):
    response: str
    thread_id: str

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main web interface"""
    try:
        with open("templates/index.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        return html_content
    except FileNotFoundError:
        return HTMLResponse("""
        <html>
        <head><title>Trading Agent</title></head>
        <body style="background: #0a0b0f; color: white; font-family: Inter;">
        <div style="padding: 40px; text-align: center;">
        <h1>🚀 Trading Agent Interface</h1>
        <p>Setting up web interface files...</p>
        <p>Please ensure templates/index.html exists.</p>
        </div>
        </body>
        </html>
        """)

@app.post("/api/analysis/technical")
async def technical_analysis(request: AnalysisRequest):
    """Perform technical analysis on a stock"""
    try:
        print(f"🔍 Running technical analysis for {request.ticker}")
        result = analyze_stock_technical(
            ticker=request.ticker,
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

@app.post("/api/analysis/fundamental")
async def fundamental_analysis(request: AnalysisRequest):
    """Perform fundamental analysis on a stock"""
    try:
        print(f"📊 Running fundamental analysis for {request.ticker}")
        result = analyze_stock_fundamentals(ticker=request.ticker)
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

@app.post("/api/analysis/news")
async def news_analysis(request: AnalysisRequest):
    """Perform news sentiment analysis on a stock"""
    try:
        print(f"📰 Running news analysis for {request.ticker}")
        result = analyze_stock_news(
            ticker=request.ticker,
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

@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_agent(request: ChatRequest):
    """Chat with the LangGraph trading agent"""
    try:
        print(f"💬 Processing chat message: {request.message[:50]}...")
        
        # Use the LangGraph agent from graph.py
        response = graph.invoke(
            {"messages": [HumanMessage(content=request.message)]},
            config={"configurable": {"thread_id": request.thread_id}}
        )
        
        # Extract the assistant's response
        assistant_message = response["messages"][-1].content
        
        return ChatResponse(
            response=assistant_message,
            thread_id=request.thread_id
        )
    except Exception as e:
        print(f"❌ Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return JSONResponse({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    })

@app.get("/api/tickers")
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)