import logging
from functools import partial
from pathlib import Path

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from agents.nodes import (
    agent_node,
    should_continue,
    evaluate_node,
    should_continue_after_eval,
    respond_node,
)
from agents.state import AgentState
from agents.tools import get_tools_for_mode
from app.core.config import Settings
from retrieval.store import create_store_manager

logger = logging.getLogger("langgraph_agent")

_checkpointer: MemorySaver | None = None

DEFAULT_MAX_ITERATIONS = 3


def get_checkpointer() -> MemorySaver:
    """Get or create the global checkpointer."""
    global _checkpointer
    if _checkpointer is None:
        _checkpointer = MemorySaver()
    return _checkpointer


def _check_index_status(settings: Settings) -> None:
    try:
        store = create_store_manager(
            data_dir=Path(settings.data_dir),
            google_api_key=settings.google_api_key,
            chroma_host=settings.chroma_host,
            chroma_port=settings.chroma_port,
        )
        if store.index_exists():
            stats = store.get_index_stats()
            logger.info(f"Vector store: {stats.get('document_count', 0)} documents")
        else:
            logger.warning("Index not found. Run prepare_docs script.")
    except Exception as e:
        logger.warning(f"Could not check index status: {e}")


def create_agent_graph(settings: Settings, max_iterations: int = DEFAULT_MAX_ITERATIONS) -> StateGraph:
    """
    Create the LangGraph agent with evaluation loop.

    Graph flow: START -> agent -> tools -> evaluate -> (agent loop OR respond -> END)
    """
    _check_index_status(settings)

    llm = ChatGoogleGenerativeAI(
        model=settings.llm_model,
        google_api_key=settings.google_api_key,
        temperature=0,
        streaming=True,
    )

    tools = get_tools_for_mode(settings.is_online)
    mode = "online" if settings.is_online else "offline"
    logger.info(f"Creating agent: {len(tools)} tools, mode={mode}, max_iter={max_iterations}")

    llm_with_tools = llm.bind_tools(tools)

    graph = StateGraph(AgentState)
    graph.add_node("agent", partial(agent_node, llm_with_tools=llm_with_tools, is_online=settings.is_online))
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("evaluate", partial(evaluate_node, llm=llm))
    graph.add_node("respond", partial(respond_node, llm=llm))

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "evaluate": "evaluate"})
    graph.add_edge("tools", "evaluate")
    graph.add_conditional_edges("evaluate", should_continue_after_eval, {"agent": "agent", "respond": "respond"})
    graph.add_edge("respond", END)

    compiled = graph.compile(checkpointer=get_checkpointer())
    logger.info("Agent graph compiled successfully")

    return compiled
