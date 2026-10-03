import logging
import threading
from dataclasses import dataclass

from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    InternalServerError,
    NotFoundError,
    OpenAI,
    PermissionDeniedError,
    RateLimitError,
)

from src.config import settings
from src.providers.circuit_breaker import CircuitBreaker
from src.providers.clients import groq_client, groq_client_secondary, nim_client
from src.tracing import provider_call_span

logger = logging.getLogger(__name__)


class EmptyCompletionError(RuntimeError):
    pass


class IncompleteCompletionError(RuntimeError):
    pass


RETRYABLE = (
    APITimeoutError,
    RateLimitError,
    APIConnectionError,
    InternalServerError,
    AuthenticationError,
    PermissionDeniedError,
    NotFoundError,
    EmptyCompletionError,
    IncompleteCompletionError,
)

_provider_breakers: dict[str, CircuitBreaker] = {}
_provider_breakers_lock = threading.Lock()


def _breaker_for(name: str, model: str) -> CircuitBreaker:
    key = f"{name}:{model}"
    with _provider_breakers_lock:
        breaker = _provider_breakers.get(key)
        if breaker is None:
            breaker = CircuitBreaker(settings.provider_circuit_failure_threshold, settings.provider_circuit_recovery_seconds)
            _provider_breakers[key] = breaker
        return breaker


@dataclass
class CompletionResult:
    content: str
    provider: str
    model: str


def _call_openai(
    client: OpenAI,
    model: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    provider_name: str,
    role: str,
    timeout_seconds: float,
) -> str:
    with provider_call_span(provider=provider_name, model=model, role=role):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout_seconds,
        )
        choice = response.choices[0]
        content = choice.message.content
        if content is None or not content.strip():
            raise EmptyCompletionError(f"{provider_name} returned an empty completion")
        if choice.finish_reason == "length":
            raise IncompleteCompletionError(f"{provider_name} completion was cut off at max_tokens={max_tokens}")
        return content


def _openai_link(
    client: OpenAI,
    model: str,
    name: str,
    messages: list,
    temperature: float,
    max_tokens: int,
    role: str,
    timeout_seconds: float,
) -> dict:
    return {
        "name": name,
        "model": model,
        "call": lambda: _call_openai(
            client,
            model,
            messages,
            temperature,
            max_tokens,
            name,
            role,
            timeout_seconds,
        ),
        "is_transient": lambda error: isinstance(error, RETRYABLE),
    }


def _run_chain(chain: list[dict]) -> CompletionResult:
    for i, link in enumerate(chain):
        is_last = i == len(chain) - 1
        breaker = _breaker_for(link["name"], link["model"])
        if not is_last and not breaker.allow():
            logger.warning("%s circuit open, skipping straight to %s", link["name"], chain[i + 1]["name"])
            continue
        try:
            content = link["call"]()
        except Exception as error:
            if not link["is_transient"](error):
                breaker.record_success()
                raise
            breaker.record_failure()
            if is_last:
                names = " -> ".join(step["name"] for step in chain)
                raise RuntimeError(f"all providers in chain ({names}) failed: {error}") from error
            logger.warning(
                "%s failed, falling back immediately to %s: %s",
                link["name"],
                chain[i + 1]["name"],
                error,
            )
            continue
        logger.info("served by %s (%s)", link["name"], link["model"])
        breaker.record_success()
        return CompletionResult(content=content, provider=link["name"], model=link["model"])
    raise RuntimeError("provider chain exhausted")


def generate_main(messages: list[dict], temperature: float = 0.2, max_tokens: int | None = None) -> CompletionResult:
    timeout = settings.generation_timeout_seconds
    max_tokens = max_tokens or settings.generation_max_tokens
    chain = [
        _openai_link(groq_client, settings.groq_main_model, "groq", messages, temperature, max_tokens, "main", timeout),
        _openai_link(
            groq_client_secondary,
            settings.groq_main_model_secondary,
            "groq-secondary",
            messages,
            temperature,
            max_tokens,
            "main",
            timeout,
        ),
        _openai_link(nim_client, settings.nim_main_model, "nim", messages, temperature, max_tokens, "main", timeout),
    ]
    return _run_chain(chain)


def generate_planner(
    messages: list[dict],
    temperature: float = 0.0,
    max_tokens: int | None = None,
    timeout_seconds: float | None = None,
) -> CompletionResult:
    timeout = timeout_seconds or settings.planner_timeout_seconds
    max_tokens = max_tokens or settings.planner_max_tokens
    chain = [
        _openai_link(nim_client, settings.nim_planner_model, "nim", messages, temperature, max_tokens, "planner", timeout),
        _openai_link(groq_client, settings.groq_planner_model, "groq", messages, temperature, max_tokens, "planner", timeout),
    ]
    return _run_chain(chain)
