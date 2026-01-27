from typing import Annotated, TypedDict, Optional

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    State for the LangGraph helper agent.
    
    Attributes:
        messages: Conversation history with add_messages reducer
        iteration_count: Number of tool-calling iterations completed
        max_iterations: Maximum allowed iterations before forcing response
        evaluation_result: Result of the last evaluation ("continue" or "sufficient")
    """

    messages: Annotated[list, add_messages]
    iteration_count: int
    max_iterations: int
    evaluation_result: Optional[str]
