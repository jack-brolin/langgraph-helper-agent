"""Constants for retrieval module."""

# ChromaDB collection names
PARENT_COLLECTION = "langgraph_docs_parents"
CHILD_COLLECTION = "langgraph_docs_children"

# Batch processing settings
DEFAULT_BATCH_SIZE = 50
DEFAULT_BATCH_DELAY = 1.0
MAX_RETRIES = 3

# Documentation sources
DOCS_SOURCES = {
    "langgraph_llms": {
        "url": "https://langchain-ai.github.io/langgraph/llms.txt",
        "description": "LangGraph concise overview",
        "priority": "primary",
    },
    "langgraph_llms_full": {
        "url": "https://langchain-ai.github.io/langgraph/llms-full.txt",
        "description": "LangGraph full documentation",
        "priority": "primary",
    },
    "langchain_llms": {
        "url": "https://docs.langchain.com/llms.txt",
        "description": "LangChain concise overview",
        "priority": "secondary",
    },
    "langchain_llms_full": {
        "url": "https://docs.langchain.com/llms-full.txt",
        "description": "LangChain full documentation",
        "priority": "secondary",
    },
}
