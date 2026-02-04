from typing import Annotated, TypedDict, Optional

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    State for the LangGraph helper agent with optimized Self-RAG.
    
    Attributes:
        messages: Conversation history with add_messages reducer
        iteration_count: Number of tool-calling iterations completed
        max_iterations: Maximum allowed iterations before forcing response
        should_continue_research: Flag indicating if research should continue
        previous_doc_ids: Set of chunk content hashes from previous iteration (to detect repeats)
    """

    messages: Annotated[list, add_messages]
    iteration_count: int
    max_iterations: int
    should_continue_research: Optional[bool]
    previous_doc_ids: Optional[set[str]]
