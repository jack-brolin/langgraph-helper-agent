import json
import logging
from typing import Literal

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import END
from langgraph.prebuilt import ToolNode

from agents.prompts import (
    OFFLINE_SYSTEM_PROMPT,
    ONLINE_SYSTEM_PROMPT,
    SYNTHESIS_SYSTEM,
    SYNTHESIS_WITH_ASSESSMENT,
)
from agents.state import AgentState

logger = logging.getLogger("langgraph_agent")


async def agent_node(state: AgentState, llm_with_tools, is_online: bool = False) -> dict:
    """
    Agent node: Gathers information by calling tools.
    
    This is the only node that calls tools. When respond_node determines more research
    is needed, it sends guidance back as a HumanMessage which the agent will read
    and act upon by making appropriate tool calls.
    """
    messages = state.get("messages", [])
    
    system_prompt = ONLINE_SYSTEM_PROMPT if is_online else OFFLINE_SYSTEM_PROMPT
    full_messages = [SystemMessage(content=system_prompt)] + list(messages)

    response = await llm_with_tools.ainvoke(full_messages)
    return {"messages": [response]}


def should_continue_agent(state: AgentState) -> Literal["tools", "respond"]:
    """Route after agent: execute tools or assess in respond."""
    messages = state.get("messages", [])
    
    if messages and hasattr(messages[-1], "tool_calls") and messages[-1].tool_calls:
        logger.debug(f"Agent routing to TOOLS ({len(messages[-1].tool_calls)} tool calls)")
        return "tools"
    
    logger.debug("Agent routing to RESPOND for assessment")
    return "respond"


def should_continue_respond(state: AgentState) -> Literal["agent", END]:
    """Route after respond: back to agent for more research or END with final answer."""
    messages = state.get("messages", [])
    
    if not messages:
        return END
    
    last_message = messages[-1]
    
    if isinstance(last_message, HumanMessage):
        logger.debug("Respond routing to AGENT (guidance provided)")
        return "agent"
    
    logger.debug("Respond routing to END (final answer)")
    return END



async def respond_node(state: AgentState, llm) -> dict:
    """
    Assess research completeness and either guide agent or generate final answer.
    
    This node evaluates gathered information:
    1. Checks if search results sufficiently answer the question
    2. If INSUFFICIENT and NOT at max iterations: Returns guidance for agent
    3. If SUFFICIENT or AT max iterations: Generates final answer
    
    Uses base LLM (no tools) - only the agent calls tools.
    The prompt itself handles the max iteration check and forces answer generation.
    """
    messages = state.get("messages", [])
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 3)
    
    if iteration_count >= max_iterations:
        iteration_warning = "⚠️ **MAX ITERATIONS REACHED** - You MUST generate a final answer with available information. DO NOT request more research."
    else:
        iteration_warning = f"You have {max_iterations - iteration_count} iterations remaining for research if needed."
    
    synthesis_with_assessment = SYNTHESIS_WITH_ASSESSMENT.format(
        iteration=iteration_count,
        max_iterations=max_iterations,
        iteration_warning=iteration_warning
    )
    
    full_messages = [SystemMessage(content=SYNTHESIS_SYSTEM)] + list(messages) + [HumanMessage(content=synthesis_with_assessment)]
    response = await llm.ainvoke(full_messages)
    
    response_text = response.content if hasattr(response, "content") else str(response)
    
    needs_research_indicators = [
        "NEED MORE RESEARCH",
        "MISSING INFORMATION",
        "CONTINUE RESEARCH",
        "SEARCH FOR:"
    ]
    
    if any(indicator in response_text.upper() for indicator in needs_research_indicators):
        logger.info(f"Assessment: More research needed (iteration {iteration_count}/{max_iterations}), routing back to agent")
        return {"messages": [HumanMessage(content=response_text)]}
    else:
        logger.info(f"Assessment: Final answer generated (iteration {iteration_count}/{max_iterations})")
        return {"messages": [response]}

def route_after_tools(state: AgentState) -> Literal["respond", END]:
    """
    Route after tools execute: Track chunks, detect repetition, update iteration count.
    
    Replaces the old quality_check_node by doing state updates and routing inline.
    Relevance filtering now happens in the tools themselves.
    
    Returns "respond" to continue processing, or END if appropriate.
    """
    messages = state.get("messages", [])
    iteration_count = state.get("iteration_count", 0) + 1
    max_iterations = state.get("max_iterations", 3)
    previous_doc_ids = state.get("previous_doc_ids") or set()
    current_doc_ids = set()

    tool_results = []
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            content = msg.content
            if isinstance(content, str):
                try:
                    results = json.loads(content)
                    if isinstance(results, list):
                        tool_results.extend(results)
                except:
                    tool_results.append({"content": content})
        elif isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
            break

    for doc in tool_results:
        if "error" in doc or not doc.get("content", doc.get("snippet", "")):
            continue
        
        chunk_content = doc.get("content") or doc.get("snippet", "")
        chunk_id = chunk_content[:500] if chunk_content else ""
        if chunk_id:
            current_doc_ids.add(chunk_id)
    
    logger.info(f"After tools: {len(current_doc_ids)} chunks retrieved (iteration {iteration_count}/{max_iterations})")
    
    if iteration_count > 1 and previous_doc_ids and current_doc_ids:
        overlap = len(current_doc_ids & previous_doc_ids)
        overlap_ratio = overlap / len(current_doc_ids)
        
        if overlap_ratio > 0.8:
            logger.warning(f"Repetitive chunks detected ({overlap_ratio:.0%} overlap). Stopping research.")
            state["iteration_count"] = iteration_count
            state["previous_doc_ids"] = current_doc_ids
            state["should_continue_research"] = False
            return "respond"
    
    if iteration_count >= max_iterations:
        logger.info(f"Max iterations reached. Proceeding to response.")
        state["iteration_count"] = iteration_count
        state["previous_doc_ids"] = current_doc_ids
        state["should_continue_research"] = False
        return "respond"
    
    state["iteration_count"] = iteration_count
    state["previous_doc_ids"] = current_doc_ids
    state["should_continue_research"] = False
    return "respond"
