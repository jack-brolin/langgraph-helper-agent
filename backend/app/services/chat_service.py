import json
import uuid
from typing import AsyncGenerator

from agents.executor import AgentExecutor
from app.core.config import Settings


class ChatService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.executor = AgentExecutor(settings)

    async def stream_chat(self, question: str, thread_id: str = None) -> AsyncGenerator[str, None]:
        """
        Stream chat response as Server-Sent Events.
        
        Converts agent events to SSE format:
        - event: token
        - event: tool_call
        - event: tool_result
        - event: done
        - event: error
        """
        thread_id = thread_id or str(uuid.uuid4())

        async for event in self.executor.run(question, thread_id):
            event_type = event["type"]
            event_data = event["data"]
            
            yield f"event: {event_type}\ndata: {json.dumps(event_data)}\n\n"
