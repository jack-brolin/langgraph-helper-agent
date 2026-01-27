import logging
import uuid

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.api.schemas import ChatRequest
from app.core.config import Settings, get_settings
from app.services.chat_service import ChatService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger("langgraph_agent")
router = APIRouter()


@router.post("/chat")
async def chat(payload: ChatRequest, settings: Settings = Depends(get_settings)):
    """
    Chat endpoint with streaming support.
    
    Returns Server-Sent Events (SSE) stream with:
    - event: token - Incremental response tokens
    - event: tool_call - When agent calls a tool
    - event: tool_result - Tool execution result
    - event: done - Final message with thread_id
    - event: error - Error message
    """
    thread_id = payload.thread_id or str(uuid.uuid4())
    service = ChatService(settings)
    
    return StreamingResponse(
        service.stream_chat(payload.question, thread_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
