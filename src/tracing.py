import hashlib

import logfire

from src.config import settings

logfire.configure(
    token=settings.logfire_token,
    send_to_logfire="if-token-present",
    service_name="kubernetes-gated-rag",
)


def turn_span(user_message: str):
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
