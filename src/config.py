from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    nvidia_nim_api_key: str
    groq_api_key: str
    gemini_api_key: str
    qdrant_url: str
    qdrant_api_key: str
    logfire_token: str | None = None

    groq_main_model: str = "openai/gpt-oss-120b"
    nim_main_model: str = "openai/gpt-oss-120b"
    gemini_main_model: str = "gemini-2.5-flash"
    nim_planner_model: str = "meta/llama-3.1-8b-instruct"
    groq_planner_model: str = "openai/gpt-oss-20b"
    gemini_planner_model: str = "gemini-2.5-flash-lite"
    groq_eval_judge_model: str = "openai/gpt-oss-120b"

    nemoguard_topic_model: str = "nvidia/llama-3.1-nemoguard-8b-topic-control"
    nemoguard_safety_model: str = "nvidia/llama-3.1-nemoguard-8b-content-safety"

    gemini_embedding_model: str = "gemini-embedding-001"
    embedding_dim: int = 768

    semantic_cache_similarity_threshold: float = 0.95

    rerank_score_threshold: float = 0.5

    # how long a cached "no grounded documentation" answer is trusted before
    # it's re-checked against retrieval; keeps re-ingested corpora from being
    # shadowed by a stale no-context verdict for the same question
    no_context_cache_ttl_seconds: int = 3600

    qdrant_docs_collection: str = "kubernetes_docs"
    qdrant_cache_collection: str = "semantic_cache"

    nim_base_url: str = "https://integrate.api.nvidia.com/v1"
    groq_base_url: str = "https://api.groq.com/openai/v1"

    rerank_top_k: int = 20
    rerank_model: str = "ms-marco-MiniLM-L-12-v2"

settings = Settings()