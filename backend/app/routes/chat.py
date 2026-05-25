from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
import asyncio
import json

from app.models.schemas import ChatRequest, ChatResponse
from app.tradingagents.chat_graph import graph
from app.utils.progress import get_progress, clear_progress

router = APIRouter(prefix="/api", tags=["chat"])

@router.post("/chat", response_model=ChatResponse)
async def chat_with_agent(request: ChatRequest):
    """Chat with the LangGraph trading agent"""
    try:
        print(f"💬 Processing chat message: {request.message[:50]}...")
        
        response = graph.invoke(
            {"messages": [HumanMessage(content=request.message)]},
            config={"configurable": {"thread_id": request.thread_id}}
        )
        
        assistant_message = response["messages"][-1].content
        
        return ChatResponse(
            response=assistant_message,
            thread_id=request.thread_id
        )
    except Exception as e:
        print(f"❌ Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")
    finally:
        clear_progress(request.thread_id)

@router.get("/chat/stream/{thread_id}")
async def stream_chat_progress(thread_id: str):
    """SSE endpoint to stream progress of the TradingAgentGraph"""
    async def event_generator():
        last_progress = []
        try:
            # Poll for up to 3 minutes
            for _ in range(360):
                current_progress = get_progress(thread_id)
                if current_progress != last_progress:
                    last_progress = list(current_progress)
                    yield f"data: {json.dumps({'nodes': current_progress})}\n\n"
                
                # Heartbeat to keep connection alive
                yield ": heartbeat\n\n"
                await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            pass
            
    return StreamingResponse(event_generator(), media_type="text/event-stream")
