import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def get_secret(key: str, default: str | None = None) -> str | None:
    val = os.getenv(key)
    if val:
        return val
    try:
        import streamlit as st

        return st.secrets.get(key, default)
    except Exception:
        return default


@dataclass
class EmbeddingConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    api_key: str | None = None
    max_workers: int = 4
    max_retries: int = 3
    retry_delay: float = 1.5

    def __post_init__(self):
        self.api_key = self.api_key or get_secret("HUGGINGFACE_API_KEY")


@dataclass
class PineconeConfig:
    api_key: str | None = None
    index_name: str = "rag-chatbot"
    namespace: str = "default"
    dimension: int = 384
    metric: str = "cosine"
    cloud: str = "aws"
    region: str = "us-east-1"

    def __post_init__(self):
        self.api_key = self.api_key or get_secret("PINECONE_API_KEY")
        self.index_name = self.index_name or get_secret(
            "PINECONE_INDEX_NAME", "rag-chatbot"
        )


@dataclass
class GroqConfig:
    api_key: str | None = None
    model: str = "llama-3.1-8b-instant"
    temperature: float = 0.1
    max_tokens: int = 1024
    request_timeout: int = 30

    def __post_init__(self):
        self.api_key = self.api_key or get_secret("GROQ_API_KEY")
        self.model = self.model or get_secret("GROQ_MODEL", "llama-3.1-8b-instant")


@dataclass
class ChunkingConfig:
    child_chunk_size: int = 300
    child_chunk_overlap: int = 50
    parent_chunk_size: int = 1200
    parent_chunk_overlap: int = 200
    separators: list = field(
        default_factory=lambda: ["\n## ", "\n### ", "\n\n", "\n", " ", ""]
    )


@dataclass
class RetrievalConfig:
    dense_top_k: int = 10
    sparse_top_k: int = 10
    rrf_k: int = 60
    final_top_k: int = 3
    reranker_model: str = "BAAI/bge-reranker-base"
    reranker_threshold: float = 0.3
    min_relevance_score: float = 0.25


@dataclass
class AgentConfig:
    query_rewriter_model: str = "llama-3.1-8b-instant"
    query_rewriter_temperature: float = 0.0
    intent_confidence_threshold: float = 0.7
    crag_confidence_threshold: float = 0.3
    web_fallback_enabled: bool = True
    web_search_max_results: int = 5


@dataclass
class ObservabilityConfig:
    enabled: bool = False
    public_key: str | None = None
    secret_key: str | None = None
    host: str = "https://cloud.langfuse.com"
    project_name: str = "rag-engine"

    def __post_init__(self):
        self.enabled = self.enabled or (
            get_secret("LANGFUSE_ENABLED", "false").lower() == "true"
        )
        self.public_key = self.public_key or get_secret("LANGFUSE_PUBLIC_KEY")
        self.secret_key = self.secret_key or get_secret("LANGFUSE_SECRET_KEY")
        self.host = self.host or get_secret(
            "LANGFUSE_HOST", "https://cloud.langfuse.com"
        )


@dataclass
class AppConfig:
    max_question_length: int = 2000
    max_questions_per_session: int = 50
    max_file_upload_mb: int = 50
    max_file_content_chars: int = 200000
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    pinecone: PineconeConfig = field(default_factory=PineconeConfig)
    groq: GroqConfig = field(default_factory=GroqConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)


_config_instance: AppConfig | None = None


def get_config() -> AppConfig:
    global _config_instance
    if _config_instance is None:
        _config_instance = AppConfig()
    return _config_instance


def reset_config():
    global _config_instance
    _config_instance = None
