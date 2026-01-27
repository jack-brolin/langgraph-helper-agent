from typing import Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(description="User's question about LangGraph/LangChain")
    thread_id: Optional[str] = Field(default=None, description="Thread ID for conversation continuity")
