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
    # meta/llama-3.1-8b-instruct was retired from NVIDIA's hosted NIM API
    # catalog on 2026-08-26 (self-host NIM containers of it still exist,
    # the hosted endpoint this project calls does not). Replaced with
    # meta/llama-3.2-3b-instruct: same weight class and provider, still
    # live on the hosted API as of this writing, and Meta's own model card
    # lists "query and prompt rewriting" as an intended use case, which
    # covers everything this chain is actually used for here.
    # meta/llama-3.2-3b-instruct was also later retired from NVIDIA's
    # hosted API catalog (same fate as meta/llama-3.1-8b-instruct before
    # it). nvidia/nemotron-3-super-120b-a12b confirmed working via a real
    # trace log ("served by nim (nvidia/nemotron-3-super-120b-a12b)")
    # rather than another guess at a model string, it's NVIDIA's own
    # model, not a third-party one they might deprecate on the same kind
    # of schedule. It's a larger, reasoning-capable model rather than a
    # small terse one, which is exactly why _parse_binary_verdict and
    # _parse_safety_json were made more lenient elsewhere in this file,
    # see their docstrings.
    nim_planner_model: str = "nvidia/nemotron-3-super-120b-a12b"
    groq_planner_model: str = "openai/gpt-oss-20b"

    # eval judge stays on a separate model family from both live chains
    # (Groq gpt-oss and NIM llama) so a model's own family never grades its
    # own output.
    gemini_eval_judge_model: str = "gemini-3.5-flash"

    # safety/topic gates: purpose-built NeMoGuard classifiers, called
    # directly against nim_client (see src/guardrails/gates.py). NeMoGuard
    # only exists on NIM, so a failed call retries and then falls back to
    # generate_planner's chain rather than failing closed immediately, see
    # _call_nemoguard's docstring in gates.py for why.
    nemoguard_topic_model: str = "nvidia/llama-3.1-nemoguard-8b-topic-control"
    nemoguard_safety_model: str = "nvidia/llama-3.1-nemoguard-8b-content-safety"

    # When True, skip straight to the generate_planner fallback classifier
    # instead of first attempting (and retrying 3x against) the NeMoGuard
    # models above. Default False: NeMoGuard's purpose-tuned classifiers
    # are the better judgment when they're healthy, so the normal path
    # should still try them first. Flip this on only as a stopgap during a
    # sustained NVIDIA-side outage on those specific hosted models (the
    # symptom: every turn's log shows "topic gate's NeMoGuard call failed
    # after retries" before it recovers via the fallback), paying ~9s of
    # guaranteed-to-fail retries on every single turn while that endpoint
    # is down is pure waste. Flip it back off once NVIDIA's instance
    # recovers, since it means running on a lower-confidence classifier
    # (see _fallback_topic_check's and _fallback_safety_check's docstrings
    # in gates.py) for no reason once the primary is healthy again.
    guardrail_skip_nemoguard: bool = False

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