"""
Centralized application configuration.
All environment-dependent values are declared here and nowhere else,
so the rest of the codebase never touches os.environ directly.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Groq
    groq_api_key: str
    groq_model: str = "llama-3.1-8b-instant"

    # HuggingFace Inference API
    hf_api_token: str
    hf_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # LangSmith
    langchain_tracing_v2: bool = True
    langchain_api_key: str | None = None
    langchain_project: str = "pdf-rag-agent"

    # Lakera Guard
    lakera_api_key: str | None = None
    lakera_api_url: str = "https://api.lakera.ai/v2/guard"

    # NeMo Guardrails
    nemo_config_path: str = "guardrails_config"

    # App behaviour
    app_env: str = "development"
    log_level: str = "INFO"
    max_memory_turns: int = 5
    retrieval_top_k: int = 5
    relevance_score_threshold: float = 0.35
    chroma_persist_dir: str = "data/chroma"
    upload_dir: str = "data/uploads"

    # MCP
    mcp_server_script: str = "mcp_server/server.py"


@lru_cache
def get_settings() -> Settings:
    return Settings()
