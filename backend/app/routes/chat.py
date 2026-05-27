import asyncio
import json
import logging

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage

from app.models.schemas import ChatRequest
from app.tradingagents.chat_graph import graph
from app.utils.progress import get_progress, clear_progress

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat")
async def chat_stream(request: ChatRequest):
    """
    Single streaming endpoint for the AI trading assistant.

    Returns an SSE stream with three event types:
      • {"type": "token",    "content": "…"}   — LLM tokens as they are generated
      • {"type": "progress", "nodes": [...]}   — which inner-graph agents are active
      • {"type": "done",     "response": "…"}  — full accumulated response at the end
      • {"type": "error",    "message": "…"}   — if something goes wrong
    """
    thread_id = request.thread_id
    config = {"configurable": {"thread_id": thread_id}}
    input_data = {"messages": [HumanMessage(content=request.message)]}

    async def event_generator():
        queue: asyncio.Queue = asyncio.Queue()
        full_response = ""

        # ── Producer 1: graph events (LLM tokens, node starts) ──────────
        async def stream_graph():
            try:
                async for event in graph.astream_events(
                    input_data, config, version="v2"
                ):
                    await queue.put(("graph", event))
            except Exception as exc:
                await queue.put(("error", str(exc)))
            finally:
                await queue.put(("graph_done", None))

        # ── Producer 2: inner-graph progress polling ────────────────────
        async def poll_progress():
            last: list[str] = []
            try:
                while True:
                    await asyncio.sleep(0.5)
                    current = get_progress(thread_id)
                    if current != last:
                        last = list(current)
                        await queue.put(("progress", current))
            except asyncio.CancelledError:
                pass

        graph_task = asyncio.create_task(stream_graph())
        progress_task = asyncio.create_task(poll_progress())

        try:
            while True:
                try:
                    msg_type, data = await asyncio.wait_for(
                        queue.get(), timeout=300  # 5-min hard cap
                    )
                except asyncio.TimeoutError:
                    yield _sse({"type": "error", "message": "Request timed out after 5 minutes"})
                    break

                if msg_type == "graph_done":
                    break

                if msg_type == "error":
                    yield _sse({"type": "error", "message": data})
                    break

                if msg_type == "progress":
                    yield _sse({"type": "progress", "nodes": data})

                if msg_type == "graph":
                    event = data
                    kind = event.get("event", "")

                    # Real LLM token streaming
                    if kind == "on_chat_model_stream":
                        chunk = event.get("data", {}).get("chunk")
                        if chunk and hasattr(chunk, "content") and chunk.content:
                            token = chunk.content
                            full_response += token
                            yield _sse({"type": "token", "content": token})

                    # Node lifecycle — only interesting, non-internal names
                    elif kind == "on_chain_start":
                        name = event.get("name", "")
                        if _is_interesting_node(name):
                            yield _sse({"type": "status", "node": name})

            # Final aggregated response
            yield _sse({"type": "done", "response": full_response})

        except Exception as exc:
            logger.exception("SSE stream error")
            yield _sse({"type": "error", "message": "Internal server error"})

        finally:
            progress_task.cancel()
            graph_task.cancel()
            clear_progress(thread_id)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",      # Disable nginx buffering
        },
    )


# ── Helpers ──────────────────────────────────────────────────────────────

def _sse(payload: dict) -> str:
    """Format a single SSE data frame."""
    return f"data: {json.dumps(payload)}\n\n"


_INTERNAL_PREFIXES = (
    "Runnable", "LangGraph", "ChannelWrite", "ChannelRead",
    "__", "Branch", "CompiledGraph",
)

def _is_interesting_node(name: str) -> bool:
    """Filter out LangGraph internal plumbing nodes."""
    if not name:
        return False
    return not any(name.startswith(p) for p in _INTERNAL_PREFIXES)
