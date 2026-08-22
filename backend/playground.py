import uvicorn
from fastapi import FastAPI
from sse_starlette.sse import EventSourceResponse

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.7-flash",
    api_key=GEMINI_API_KEY
)

app = FastAPI()


@app.get("/see_test")
async def chat(query: str):

    async def event_generator():
        async for chunk in llm.astream(query):
            yield {"data": chunk.text}

    return EventSourceResponse(event_generator())


if __name__ == "__main__":
    uvicorn.run(
        "playground:app",
        host="0.0.0.0",
        port=5000,
        reload=True
    )