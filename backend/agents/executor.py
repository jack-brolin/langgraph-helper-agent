import logging
from typing import AsyncGenerator

from agents.graph import create_agent_graph, DEFAULT_MAX_ITERATIONS
from app.core.config import Settings

logger = logging.getLogger("langgraph_agent")

# Display messages for different nodes in the graph
# TODO: Maybe make these configurable or move to config file
NODE_MESSAGES = {
    "agent": ("agent", "Iteration {}: Analyzing and deciding next action..."),
    "tools": ("tools", "Executing tools..."),
    "evaluate": ("evaluate", "Evaluating research quality..."),
    "respond": ("respond", "Research complete! Generating final answer..."),
}


class AgentExecutor:
    """Executes the LangGraph agent and yields events."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.graph = create_agent_graph(settings)

    async def run(self, question: str, thread_id: str) -> AsyncGenerator[dict, None]:
        """
        Run the agent and yield events.

        Event types: reasoning, token, final_answer_start, done, error
        """
        logger.info(f"Request: thread={thread_id[:8]}, mode={self.settings.agent_mode.value}")

        config = {"configurable": {"thread_id": thread_id}}
        state = {
            "current_node": None,
            "iteration": 0,
            "in_respond": False,
            "answer_started": False,
            "streamed": False,
            "sources": [],
        }

        try:
            yield {"type": "reasoning", "data": {"step": "start", "message": "Starting research..."}}

            # Initialize state with all required fields
            initial_state = {
                "messages": [("user", question)],
                "iteration_count": 0,
                "max_iterations": DEFAULT_MAX_ITERATIONS,
                "evaluation_result": None,
            }

            async for event in self.graph.astream_events(
                initial_state,
                config=config,
                version="v1",
            ):
                result = self._process_event(event, state)
                if result:
                    for r in result:
                        yield r

            if state["sources"]:
                for source in state["sources"]:
                    yield {"type": "citation", "data": source}

            yield {
                "type": "done",
                "data": {"thread_id": thread_id, "mode": self.settings.agent_mode.value, "iterations": state["iteration"]},
            }

        except ValueError as e:
            logger.error(f"Validation error: {e}")
            yield {"type": "error", "data": {"error": str(e)}}
        except Exception as e:
            logger.exception(f"Internal error: {e}")
            yield {"type": "error", "data": {"error": f"Internal error: {str(e)}"}}

    def _process_event(self, event: dict, state: dict) -> list[dict] | None:
        """Process a single event and return any results to yield."""
        event_type = event.get("event")
        
        if event_type == "on_chain_start":
            return self._handle_node_transition(event, state)
        elif event_type == "on_tool_start":
            return self._handle_tool_start(event)
        elif event_type == "on_tool_end":
            return self._handle_tool_end(event, state)
        elif event_type == "on_chat_model_stream":
            return self._handle_llm_stream(event, state)
        elif event_type == "on_chat_model_end":
            return self._handle_llm_completion(event, state)
        
        return None

    def _handle_node_transition(self, event: dict, state: dict) -> list[dict] | None:
        """Handle transitions between graph nodes."""
        node_name = event.get("name", "")
        if not node_name or node_name == state["current_node"] or node_name not in NODE_MESSAGES:
            return None

        state["current_node"] = node_name
        step, msg_template = NODE_MESSAGES[node_name]
        results = []

        if node_name == "agent":
            state["iteration"] += 1
            msg = msg_template.format(state["iteration"])
            results.append({
                "type": "reasoning",
                "data": {"step": step, "iteration": state["iteration"], "message": msg}
            })
        elif node_name == "respond":
            state["in_respond"] = True
            results.append({"type": "reasoning", "data": {"step": step, "message": msg_template}})
        else:
            results.append({"type": "reasoning", "data": {"step": step, "message": msg_template}})

        return results

    def _handle_tool_start(self, event: dict) -> list[dict]:
        """Handle tool call events."""
        tool_name = event.get("name", "unknown")
        tool_input = event.get("data", {}).get("input", {})
        
        # Extract query from input
        if isinstance(tool_input, dict):
            query = tool_input.get("query", str(tool_input))
        else:
            query = str(tool_input)

        return [{
            "type": "reasoning",
            "data": {
                "step": "tool_call",
                "tool": tool_name,
                "query": query,
                "message": f"  → Calling {tool_name}: \"{query}\""
            }
        }]

    def _handle_tool_end(self, event: dict, state: dict) -> list[dict]:
        """Handle tool result events and extract sources."""
        tool_name = event.get("name", "unknown")
        output = event.get("data", {}).get("output", [])
        
        # Tools return list[dict], extract sources from them
        if isinstance(output, list):
            for item in output:
                if isinstance(item, dict) and ("url" in item or "source" in item):
                    source_url = item.get("url") or item.get("source", "")
                    if source_url and source_url not in [s.get("source_url") for s in state["sources"]]:
                        state["sources"].append({
                            "source_url": source_url,
                            "title": item.get("title") or item.get("section", "Documentation"),
                            "snippet": item.get("snippet") or item.get("content", "")[:200],
                        })

        count = len(output) if isinstance(output, list) else 1
        return [{
            "type": "reasoning",
            "data": {
                "step": "tool_result",
                "tool": tool_name,
                "result_count": count,
                "message": f"  ✓ {tool_name} returned {count} result(s)"
            }
        }]

    def _handle_llm_stream(self, event: dict, state: dict) -> list[dict] | None:
        """Handle streaming LLM tokens (final answer only)."""
        if not state["in_respond"]:
            return None

        chunk = event.get("data", {}).get("chunk")
        if not chunk or not hasattr(chunk, "content") or not chunk.content:
            return None

        content = str(chunk.content) if chunk.content else ""
        if not content:
            return None

        results = []
        if not state["answer_started"]:
            results.append({"type": "final_answer_start", "data": {"message": "\n\n[FINAL ANSWER]\n\n"}})
            state["answer_started"] = True
        
        state["streamed"] = True
        results.append({"type": "token", "data": {"content": content}})
        return results

    def _handle_llm_completion(self, event: dict, state: dict) -> list[dict] | None:
        """Handle LLM completion events."""
        output = event.get("data", {}).get("output")
        if not output or not hasattr(output, "content"):
            return None

        content = str(output.content) if output.content else ""
        if not content:
            return None

        results = []

        if state["current_node"] == "evaluate":
            if "CONTINUE" in content.upper():
                results.append({
                    "type": "reasoning",
                    "data": {
                        "step": "evaluation_decision",
                        "decision": "continue",
                        "message": "  → Need more info, continuing research..."
                    }
                })
            elif "SUFFICIENT" in content.upper():
                results.append({
                    "type": "reasoning",
                    "data": {
                        "step": "evaluation_decision",
                        "decision": "sufficient",
                        "message": "  → Research is sufficient, generating answer..."
                    }
                })

        if state["in_respond"] and not state["streamed"]:
            if not state["answer_started"]:
                results.append({"type": "final_answer_start", "data": {"message": "\n\n[FINAL ANSWER]\n\n"}})
                state["answer_started"] = True
            results.append({"type": "token", "data": {"content": content}})

        return results if results else None
