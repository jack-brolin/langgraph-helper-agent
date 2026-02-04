import logging
import os
from pathlib import Path

from langchain_core.tools import tool
from tavily import TavilyClient

from app.core.config import get_settings
from retrieval.store import create_store_manager

logger = logging.getLogger("langgraph_agent")


def _get_store():
    """Create a vector store manager instance."""
    settings = get_settings()
    return create_store_manager(
        data_dir=Path(settings.data_dir),
        google_api_key=settings.google_api_key,
        chroma_host=settings.chroma_host,
        chroma_port=settings.chroma_port,
    )


@tool
def search_docs(query: str) -> list[dict]:
    """
    Search the local LangGraph and LangChain documentation index.

    Best for core concepts, API references, and foundational knowledge.
    Consider supplementing with web_search for comprehensive answers.

    Args:
        query: The search query (e.g., "how to add memory to a LangGraph agent")

    Returns:
        A list of relevant documentation chunks with source information
    """
    try:
        try:
            from langsmith.run_helpers import get_current_run_tree
            run_tree = get_current_run_tree()
            if run_tree:
                run_tree.add_metadata({"search_type": "documentation", "tool": "search_docs"})
        except Exception:
            pass
        
        store = _get_store()

        if not store.index_exists():
            return [{"error": "Documentation index not found.", "suggestion": "Run 'python -m scripts.prepare_docs'"}]

        results = store.search_with_scores(query, k=5)

        docs = []
        filtered_count = 0
        for doc, score in results:
            if score < 0.3:
                filtered_count += 1
                continue
                
            docs.append({
                "content": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "url": doc.metadata.get("url", ""),
                "section": doc.metadata.get("section", ""),
                "relevance_score": round(score, 4),
            })
        
        if filtered_count > 0:
            logger.info(f"Filtered {filtered_count} low-relevance docs (< 0.3)")

        return docs if docs else [{"message": "No relevant documentation found. Try rephrasing your query."}]

    except Exception as e:
        logger.error(f"search_docs error: {e}")
        return [{"error": f"Documentation search failed: {str(e)}"}]


@tool
def web_search(query: str) -> list[dict]:
    """
    Search the web for LangGraph and LangChain information.

    Best for latest updates, real-world examples, tutorials, and troubleshooting.
    Combine with search_docs for comprehensive answers.

    Args:
        query: The search query

    Returns:
        A list of web search results with URLs and snippets
    """
    try:
        try:
            from langsmith.run_helpers import get_current_run_tree
            run_tree = get_current_run_tree()
            if run_tree:
                run_tree.add_metadata({"search_type": "web", "tool": "web_search"})
        except Exception:
            pass
        
        settings = get_settings()

        if not settings.is_online:
            return [{"error": "Web search is not available in offline mode."}]

        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            return [{"error": "TAVILY_API_KEY not configured."}]

        client = TavilyClient(api_key=api_key)
        response = client.search(query=query, max_results=5)

        results = []
        filtered_count = 0
        for item in response.get("results", []):
            score = item.get("score", 0.7)
            
            if score < 0.3:
                filtered_count += 1
                continue
                
            results.append({
                "url": item.get("url", ""),
                "title": item.get("title", ""),
                "snippet": item.get("content", ""),
                "relevance_score": score
            })
        
        if filtered_count > 0:
            logger.info(f"Filtered {filtered_count} low-relevance web results (< 0.3)")

        return results if results else [{"message": "No results found"}]

    except Exception as e:
        logger.error(f"web_search error: {e}")
        return [{"error": f"Web search failed: {str(e)}"}]


def get_tools_for_mode(is_online: bool) -> list:
    """Get tools based on mode: search_docs always, web_search if online."""
    tools = [search_docs]
    if is_online:
        tools.append(web_search)
    return tools
