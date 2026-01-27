import logging
from typing import Literal

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

from agents.prompts import (
    OFFLINE_SYSTEM_PROMPT,
    ONLINE_SYSTEM_PROMPT,
    EVALUATION_PROMPT,
    SYNTHESIS_SYSTEM,
    SYNTHESIS_INSTRUCTION,
)
from agents.state import AgentState

logger = logging.getLogger("langgraph_agent")


async def agent_node(state: AgentState, llm_with_tools, is_online: bool = False) -> dict:
    messages = state.get("messages", [])

    system_prompt = ONLINE_SYSTEM_PROMPT if is_online else OFFLINE_SYSTEM_PROMPT
    full_messages = [SystemMessage(content=system_prompt)] + list(messages)

    response = await llm_with_tools.ainvoke(full_messages)
    return {"messages": [response]}


def should_continue(state: AgentState) -> Literal["tools", "evaluate"]:
    """Route to tools if the last message has tool calls, otherwise to evaluate."""
    messages = state.get("messages", [])
    
    if messages and hasattr(messages[-1], "tool_calls") and messages[-1].tool_calls:
        return "tools"
    return "evaluate"


def _extract_user_question(messages: list) -> str:
    """Extract the original user question from messages."""
    for msg in messages:
        if isinstance(msg, HumanMessage):
            return msg.content
    return "Unknown question"


def _extract_tool_results(messages: list) -> str:
    """Extract tool results from recent messages."""
    results = []
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            tool_name = getattr(msg, "name", "unknown")
            content = msg.content[:1500] if len(msg.content) > 1500 else msg.content
            results.append(f"[{tool_name}]:\n{content}")
        elif isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
            break
    return "\n\n".join(reversed(results)) if results else "No tool results found"


def _parse_evaluation_response(response: str) -> str:
    response_upper = response.upper()
    
    if "CONTINUE" in response_upper:
        return "continue"
    elif "SUFFICIENT" in response_upper:
        return "sufficient"
    
    return "sufficient"


async def evaluate_node(state: AgentState, llm) -> dict:
    """Evaluate whether the gathered research is sufficient to answer the question."""
    messages = state.get("messages", [])
    iteration_count = state.get("iteration_count", 0) + 1
    max_iterations = state.get("max_iterations", 3)
    
    if iteration_count >= max_iterations:
        return {
            "iteration_count": iteration_count,
            "evaluation_result": "sufficient"
        }

    eval_prompt = EVALUATION_PROMPT.format(
        user_question=_extract_user_question(messages),
        tool_results=_extract_tool_results(messages),
        iteration=iteration_count,
        max_iterations=max_iterations,
    )

    eval_response = await llm.ainvoke([HumanMessage(content=eval_prompt)])
    eval_content = eval_response.content if hasattr(eval_response, "content") else str(eval_response)

    decision = _parse_evaluation_response(eval_content)
    
    return {
        "iteration_count": iteration_count,
        "evaluation_result": decision
    }


def should_continue_after_eval(state: AgentState) -> Literal["agent", "respond"]:
    """Route after evaluation: continue researching or proceed to final response."""
    evaluation_result = state.get("evaluation_result", "sufficient")
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 3)

    if evaluation_result == "continue" and iteration_count < max_iterations:
        return "agent"
    return "respond"


async def respond_node(state: AgentState, llm) -> dict:
    """Generate the final response after research is complete."""
    messages = state.get("messages", [])

    full_messages = [SystemMessage(content=SYNTHESIS_SYSTEM)] + list(messages) + [HumanMessage(content=SYNTHESIS_INSTRUCTION)]
    response = await llm.ainvoke(full_messages)

    return {"messages": [response]}
