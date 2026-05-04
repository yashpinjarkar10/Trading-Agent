from fastapi import APIRouter, HTTPException
from langchain_core.messages import HumanMessage

from app.models.schemas import ChatRequest, ChatResponse
from app.core.graph import graph

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
