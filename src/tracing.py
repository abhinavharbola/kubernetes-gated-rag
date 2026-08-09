import logfire

from src.config import settings

logfire.configure(token=settings.logfire_token, service_name="kubernetes-agentic-rag")


def turn_span(user_message: str):
    return logfire.span("user_turn", user_message=user_message)


def node_span(name: str, **attributes):
    return logfire.span(f"node:{name}", **attributes)


def provider_call_span(provider: str, model: str, role: str):
    return logfire.span("provider_call", provider=provider, model=model, role=role)


def log_cache_decision(layer: str, hit: bool):
    logfire.info("cache_decision", layer=layer, hit=hit)