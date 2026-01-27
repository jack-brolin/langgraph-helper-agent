# LangGraph Helper Agent

An AI-powered assistant that helps answer questions about LangGraph and LangChain using retrieval-augmented generation (RAG). The agent uses locally indexed documentation combined with Google Gemini LLM to provide accurate, contextual answers. It features an advanced parent-child chunking strategy for better retrieval, a modern React-based chat interface, and optional web search capabilities for the latest information.

## Prerequisites

- Docker and Docker Compose
- Google API Key (for Gemini API) - Get it from [Google AI Studio](https://makersuite.google.com/app/apikey)
- Tavily API Key (optional, only for online mode) - Sign up at [Tavily](https://tavily.com/)
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
The agent uses a **StateGraph** with four nodes connected in a research-evaluation loop:
- **Flow**: `START → agent → tools → evaluate → [agent (loop back) OR respond → END]`
- **Conditional routing**: After the agent calls tools, an evaluation node decides whether to continue researching (loop back to agent) or generate the final answer (proceed to respond node)
- **Iteration control**: Maximum 3 research iterations to prevent infinite loops
- **Persistence**: Compiled with `MemorySaver` checkpointer for multi-turn conversations

### State Management
The `AgentState` TypedDict manages conversation state with four fields:
- **`messages`**: Conversation history with `add_messages` reducer (automatically merges new messages)
- **`iteration_count`**: Tracks the number of research iterations completed
- **`max_iterations`**: Limits research loops (default: 3)
- **`evaluation_result`**: Stores evaluation decision (`"continue"` or `"sufficient"`)

### Node Structure
Four specialized nodes handle different aspects of the agent workflow:

**1. `agent_node`**: Orchestrates research by deciding which tools to call
- Receives system prompt (offline or online mode)
- Invokes LLM with bound tools to generate tool calls
- Returns tool call messages to state

**2. `tools`**: Executes tool calls using LangGraph's `ToolNode`
- Runs `search_docs` (local documentation search)
- Runs `web_search` (Tavily API, online mode only)
- Returns tool results as `ToolMessage` objects

**3. `evaluate_node`**: Assesses research completeness
- Extracts user question and tool results from state
- Asks LLM if information is sufficient to answer
- Routes to agent (continue) or respond (sufficient)

**4. `respond_node`**: Generates the final synthesized answer
- Uses synthesis system prompt
- Combines all gathered research
- Streams response back to user

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

## Next Steps

- **Guardrails**: Add input/output validation to ensure safe and appropriate responses
