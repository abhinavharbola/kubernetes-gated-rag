import hashlib

import logfire

from src.config import settings

# send_to_logfire defaults to None (auto-detect), which raises
# LogfireConfigError on startup if there's no token AND no cached
# `logfire auth` credentials — the opposite of "just no-ops without it".
# 'if-token-present' only attempts to send when settings.logfire_token is
# actually set, and runs as a local no-op otherwise, matching the README's
# documented behavior and letting `streamlit run` / `python ingest.py` work
# with zero Logfire setup.
logfire.configure(
    token=settings.logfire_token,
    send_to_logfire="if-token-present",
    service_name="kubernetes-agentic-rag",
)


def turn_span(user_message: str):
    # The safety taxonomy this project checks input against explicitly
    # includes PII/Privacy (S9) — shipping the raw, unredacted question to a
    # third-party observability backend by default would be inconsistent
    # with that. Length and a short hash are enough to correlate a span
    # with a specific question locally (e.g. against exact-cache keys)
    # without exporting its contents. Full raw content is still available
    # for local debugging via settings.tracing_log_raw_messages, off by
    # default.
    attributes = {
        "user_message_length": len(user_message),
        "user_message_hash": hashlib.sha256(user_message.encode("utf-8")).hexdigest()[:12],
    }
    if settings.tracing_log_raw_messages:
        attributes["user_message"] = user_message
    return logfire.span("user_turn", **attributes)


def node_span(name: str, **attributes):
    return logfire.span(f"node:{name}", **attributes)


def provider_call_span(provider: str, model: str, role: str):
    return logfire.span("provider_call", provider=provider, model=model, role=role)


def log_cache_decision(layer: str, hit: bool):
    logfire.info("cache_decision", layer=layer, hit=hit)



