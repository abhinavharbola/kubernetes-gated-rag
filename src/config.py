from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    nvidia_nim_api_key: str
    groq_api_key: str
    # second Groq account, used as the 2nd link in generate_main's chain
    # (groq -> groq_secondary -> nim). Protects against per-model/per-key
    # rate caps and single-account throttling, not a full Groq-platform
    # outage — nim is still there as the cross-vendor fallback.
    groq_api_key_secondary: str
    gemini_api_key: str
    qdrant_url: str
    qdrant_api_key: str
    logfire_token: str | None = None

    # generation chain: groq (primary) -> groq, second account (same model,
    # redundant key) -> nim (cross-vendor fallback). Gemini is intentionally
    # not in this chain, see gemini_eval_judge_model / gemini_embedding_model
    # below for its only two roles in this project.
    groq_main_model: str = "openai/gpt-oss-120b"
    groq_main_model_secondary: str = "openai/gpt-oss-120b"
    nim_main_model: str = "openai/gpt-oss-120b"

    # planner chain: nim (primary) -> groq (fallback). Reversed from the
    # generation chain on purpose — this is the chain the safety/topic gates
    # used to ride on generate_planner's classifier calls before NeMoGuard
    # took that job over directly; rewrite/canonicalize/ingestion-relevance
    # still use it.
    nim_planner_model: str = "meta/llama-3.1-8b-instruct"
    groq_planner_model: str = "openai/gpt-oss-20b"

    # eval judge stays on a separate model family from both live chains
    # (Groq gpt-oss and NIM llama) so a model's own family never grades its
    # own output.
    gemini_eval_judge_model: str = "gemini-3.5-flash"

    # safety/topic gates: purpose-built NeMoGuard classifiers, called
    # directly against nim_client (see src/guardrails/gates.py), not routed
    # through generate_planner's failover chain — NeMoGuard only exists on
    # NIM, so there's nothing to fail over to; an error here fails closed
    # like every other gate check.
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