import os
import logging
from app.core.config import Settings

logger = logging.getLogger("langgraph_agent")


def configure_langsmith(settings: Settings) -> None:
    """Configure LangSmith tracing environment variables."""
    if settings.langsmith_enabled:
        os.environ["LANGSMITH_TRACING_V2"] = "true"
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
        os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint
        logger.info(f"LangSmith tracing enabled: project={settings.langsmith_project}")
    else:
        os.environ["LANGSMITH_TRACING_V2"] = "false"
        logger.info("LangSmith tracing disabled")
