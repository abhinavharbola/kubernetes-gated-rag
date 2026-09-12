from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    nvidia_nim_api_key: str
    groq_api_key: str
    groq_api_key_secondary: str
    gemini_api_key: str
    qdrant_url: str
    qdrant_api_key: str
    logfire_token: str | None = None
    # Off by default: turn_span() logs only a length + hash of the user
    # message when this is False, so raw user input (which may contain PII,
    # per the safety taxonomy's own S9 category) isn't shipped to Logfire.
    # Set True only for local debugging with a private/local Logfire sink.
    tracing_log_raw_messages: bool = False

    # Deliberately the same open-weights model string across all three
    # links of generate_main's chain: the failover this project needs is
    # provider/account diversity (Groq primary account, Groq secondary
    # account, NVIDIA NIM), not model diversity — the goal is "keep
    # answering if one hosted endpoint is down or rate-limited", not "try a
    # different model". If you intend these to actually be different
    # models, set them explicitly via env vars; leaving them identical here
    # is intentional, not a copy-paste leftover.
    groq_main_model: str = "openai/gpt-oss-120b"
    groq_main_model_secondary: str = "openai/gpt-oss-120b"
    nim_main_model: str = "openai/gpt-oss-120b"
    nim_planner_model: str = "nvidia/nemotron-3-super-120b-a12b"
    groq_planner_model: str = "openai/gpt-oss-20b"
    gemini_eval_judge_model: str = "gemini-3.5-flash"

    nemoguard_topic_model: str = "nvidia/llama-3.1-nemoguard-8b-topic-control"
    nemoguard_safety_model: str = "nvidia/llama-3.1-nemoguard-8b-content-safety"
    guardrail_skip_nemoguard_safety: bool = False
    # NeMoGuard topic-control has been the one reliably crashing (a
    # recurring server-side TensorRT-LLM/CUDA error on NVIDIA's hosted
    # endpoint, not something a client-side retry or timeout fixes).
    # Default true: use the existing fallback classifier (Groq, via
    # TOPIC_POLICY_PROMPT) as topic's primary path instead of paying for a
    # call to a model that's reliably failing. Flip to false to give
    # NeMoGuard topic-control another try once NVIDIA's instance is
    # confirmed healthy again.
    guardrail_skip_nemoguard_topic: bool = True
    guardrail_timeout_seconds: float = 3.0
    guardrail_circuit_failure_threshold: int = 2
    guardrail_circuit_recovery_seconds: float = 30.0

    gemini_embedding_model: str = "gemini-embedding-001"
    embedding_dim: int = 768
    embedding_cache_ttl_seconds: int = 86400

    semantic_cache_similarity_threshold: float = 0.95
    rerank_score_threshold: float = 0.5
    no_context_cache_ttl_seconds: int = 3600

    qdrant_docs_collection: str = "kubernetes_docs"
    qdrant_cache_collection: str = "semantic_cache"

    nim_base_url: str = "https://integrate.api.nvidia.com/v1"
    groq_base_url: str = "https://api.groq.com/openai/v1"

    rerank_top_k: int = 20
    rerank_model: str = "ms-marco-MiniLM-L-12-v2"
    rerank_fail_closed: bool = True

    generation_timeout_seconds: float = 15.0
    planner_timeout_seconds: float = 4.0
    # 2.0s (the previous hardcoded value) was too tight for real hosted
    # Qdrant Cloud latency, especially free-tier, and caused ReadTimeouts
    # on otherwise-healthy requests rather than only on genuine outages.
    # 5s still keeps a stalled Qdrant from eating the whole interactive
    # budget the way the original 60s did, with realistic headroom.
    qdrant_timeout_seconds: float = 5.0
    # Gemini's client previously had no timeout configured at all, unlike
    # every other provider client here — a slow embed_content call could
    # hang for however long the underlying SDK/transport defaults to.
    embedding_timeout_seconds: float = 10.0

    # Circuit breaker for the shared provider chain in src/providers/llm.py
    # (nim/groq/groq-secondary), separate from guardrail_circuit_* above,
    # which only covers the two NeMoGuard classifier calls. Without this,
    # an unhealthy NIM meant every independent generate_planner/generate_main
    # call in a turn re-paid the full planner/generation timeout discovering
    # the same outage from scratch — e.g. rewrite_with_history and the topic
    # gate's fallback classifier both hitting a dead NIM for 4s each, in the
    # same turn, with no memory of the first failure.
    provider_circuit_failure_threshold: int = 2
    provider_circuit_recovery_seconds: float = 20.0

    cache_schema_version: str = "3"
    cache_policy_version: str = "2"
    corpus_version: str = "1"


settings = Settings()



