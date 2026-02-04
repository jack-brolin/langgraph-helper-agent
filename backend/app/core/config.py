from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class AgentMode(str, Enum):
    OFFLINE = "offline"
    ONLINE = "online"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    agent_mode: AgentMode = AgentMode.OFFLINE

    google_api_key: str
    tavily_api_key: str = ""

    llm_model: str = "gemini-2.5-flash"
    embedding_model: str = "models/text-embedding-004"

    data_dir: str = "./data"

    chroma_host: Optional[str] = None
    chroma_port: int = 8000

    max_results: int = 5

    batch_size: int = 50
    batch_delay: float = 1.0

    # LangSmith Configuration
    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "langgraph-helper-agent"
    langsmith_endpoint: str = ""

    @property
    def is_online(self) -> bool:
        return self.agent_mode == AgentMode.ONLINE
    
    @property
    def langsmith_enabled(self) -> bool:
        """Check if LangSmith is properly configured."""
        return self.langsmith_tracing and bool(self.langsmith_api_key)

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir)

    @property
    def use_chroma_server(self) -> bool:
        return self.chroma_host is not None and self.chroma_host != ""


@lru_cache
def get_settings() -> Settings:
    return Settings()
