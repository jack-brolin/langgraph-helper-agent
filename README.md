# LangGraph Helper Agent

An AI-powered assistant that helps answer questions about LangGraph and LangChain using retrieval-augmented generation (RAG). The agent uses locally indexed documentation combined with Google Gemini LLM to provide accurate, contextual answers. It features an advanced parent-child chunking strategy for better retrieval, a modern React-based chat interface, and optional web search capabilities for the latest information.

## Prerequisites

- Docker and Docker Compose
- Google API Key (for Gemini API) - Get it from [Google AI Studio](https://makersuite.google.com/app/apikey)
- Tavily API Key (optional, only for online mode) - Sign up at [Tavily](https://tavily.com/)
- LangSmith API Key (optional, for tracing and observability) - Sign up at [LangSmith](https://smith.langchain.com/)
- ~2GB free disk space (for ChromaDB and documentation)
- Internet connection (for initial documentation download)

## Quick Start

```bash
# 1. Clone and navigate to the project
git clone <repository-url>
cd opsfleet-langgraph-helper-agent

# 2. Set up environment variables
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY

# 3. Start all services
docker-compose up --build

# 4. Index documentation (in a new terminal, wait for services to be ready)
# Note: Initial processing takes ~1000 seconds (~16 minutes)
python -m backend.scripts.prepare_docs --chroma-host localhost --chroma-port 8001 --force

# 5. Open http://localhost:3000 and start asking questions!
```

## Architecture Overview

### Graph Design
The agent uses a **StateGraph** with three nodes in a streamlined research-assessment loop:
- **Flow**: `START → agent → tools → respond → [agent (loop back) OR END]`
- **Routing logic**:
  - `agent` → calls tools OR proceeds to `respond` (if no tools called)
  - `tools` → proceeds to `respond` OR `END` (if max iterations or repetitive results)
  - `respond` → loops back to `agent` (if more research needed) OR `END` (final answer ready)
- **Iteration control**: Maximum 3 research iterations to prevent infinite loops
- **Repetition detection**: Tracks retrieved chunks to detect when agent is getting repetitive results
- **Persistence**: Compiled with `MemorySaver` checkpointer for multi-turn conversations

### State Management
The `AgentState` TypedDict manages conversation state with five fields:
- **`messages`**: Conversation history with `add_messages` reducer (automatically merges new messages)
- **`iteration_count`**: Tracks the number of research iterations completed
- **`max_iterations`**: Limits research loops (default: 3)
- **`should_continue_research`**: Boolean flag indicating if more research is needed (optional)
- **`previous_doc_ids`**: Set of previously retrieved document chunk IDs for repetition detection (optional)

### Node Structure
Three specialized nodes handle different aspects of the agent workflow:

**1. `agent_node`**: Gathers information by calling tools
- Receives system prompt (offline or online mode)
- Invokes LLM with bound tools to decide which searches to perform
- Can respond to guidance from `respond_node` by performing suggested searches
- Returns tool call messages to state

**2. `tools`**: Executes tool calls with built-in quality filtering
- **`search_docs`**: Local documentation search with vector similarity filtering
  - Retrieves top 3 most relevant document chunks from ChromaDB
  - Filters by relevance score: only returns chunks with score ≥ 0.3
  - Low-relevance results are automatically discarded before reaching the LLM
- **`web_search`**: Real-time web search with relevance filtering (online mode only)
  - Uses Tavily API to fetch search results
  - Filters by relevance score: only returns results with score ≥ 0.3
  - Ensures only high-quality web sources are used
- Returns filtered results as `ToolMessage` objects
- **Routing**: After tools execute, checks for max iterations or chunk repetition before proceeding

**3. `respond_node`**: Assesses research completeness and generates responses
- **Assessment phase**: Evaluates if gathered information is sufficient to answer the question
- **Two output modes**:
  - **Guidance mode**: Returns `HumanMessage` with specific search suggestions when more research is needed
  - **Answer mode**: Returns `AIMessage` with final synthesized answer when information is complete
- **Max iteration enforcement**: Prompt-driven logic forces final answer at iteration limit
- Uses synthesis system prompt to ensure answers are grounded in sources

## Operating Modes

### Offline Mode
Offline mode uses **only local documentation** for answering questions:
- **Data source**: Pre-indexed LangGraph and LangChain documentation stored in ChromaDB
- **Tool available**: `search_docs` (searches the local vector store with parent-child chunking)
- **API calls**: Only to Google Gemini for LLM inference (no web search APIs)
- **Best for**: Core concepts, API references, and foundational LangGraph/LangChain knowledge
- **Advantages**: Faster responses, no external dependencies beyond Gemini, works offline (after initial setup)

### Online Mode
Online mode combines **local documentation with live web search**:
- **Data sources**: 
  - ChromaDB index (same as offline mode)
  - Live web search via Tavily API
- **Tools available**: 
  - `search_docs` (local documentation)
  - `web_search` (real-time web queries)
- **API calls**: Google Gemini (LLM) + Tavily (web search)
- **Best for**: Latest updates, recent features, real-world examples, tutorials, and troubleshooting
- **Advantages**: Access to current information beyond indexed documentation

### Switching Between Modes
To switch operating modes:

1. **Edit `.env` file**:
```bash
# For offline mode
AGENT_MODE=offline

# For online mode (requires TAVILY_API_KEY)
AGENT_MODE=online
```

2. **Restart services**:
```bash
docker-compose down
docker-compose up --build
```

**Note**: Online mode requires a valid `TAVILY_API_KEY` in your `.env` file.

## Data Strategy

### Data Preparation with Parent-Child Chunking
The documentation undergoes a sophisticated preparation process using **parent-child chunking** to optimize retrieval quality:

**1. Document Loading**:
- Downloads LangGraph and LangChain documentation from official sources
- Stores raw markdown files in `data/raw/` directory

**2. Parent-Child Chunking** (solves the RAG trade-off):
- **Parent chunks** (2000 characters): Large chunks for complete LLM context
- **Child chunks** (500 characters): Small chunks for precise retrieval
- **Code block preservation**: Custom separator (`\n````) prevents splitting code examples
- **Hierarchical linking**: Children reference parents via `parent_id` in metadata

**3. Dual Indexing**:
- Creates two ChromaDB collections:
  - `langgraph_docs_children`: Searchable index of 300-400 small chunks
  - `langgraph_docs_parents`: Context store of 75-100 large chunks
- Both collections use Google `text-embedding-004` embeddings

**How it works at query time**:
```
Query → Search children (precise match) → Extract parent_id → Return parent (full context) → LLM
```

This approach delivers **precise retrieval** (small chunks find exact matches) with **complete context** (large parent chunks give LLM full information).

### Services Used

**ChromaDB** - Vector database for document embeddings:
- **Why**: Open-source, production-ready, supports persistent storage and server mode
- **How**: Stores embeddings in two collections (parent/child) with metadata linking
- **Deployment**: Runs as Docker container with persistent volume for data retention
- **Interface**: HTTP API for remote access from backend service

**Tavily** - Web search API (online mode only):
- **Why**: Specialized for AI applications, returns clean, structured results optimized for LLMs
- **How**: Provides real-time web search when local documentation is insufficient
- **Integration**: Called via `web_search` tool when agent needs current information
- **Fallback**: System works fully without Tavily in offline mode

**LangSmith** - Observability and tracing platform (optional):
- **Why**: Debug, monitor, and trace LLM applications in production
- **What's traced**: All LLM calls, tool executions, agent decisions, and graph transitions
- **Tracing method**: Uses `@traceable` decorators on key functions and automatic LangChain integration
- **Setup**: Add `LANGSMITH_API_KEY` to `.env` and set `LANGSMITH_TRACING=true`
- **Region support**: Configure `LANGSMITH_ENDPOINT` for EU region (`https://eu.api.smith.langchain.com`)
- **Benefits**: Visualize agent workflow, measure latency, debug failures, and track token usage
- **Fallback**: System works fully without LangSmith; tracing is disabled by default

## Next Steps

- **Guardrails**: Add input/output validation to ensure safe and appropriate responses
