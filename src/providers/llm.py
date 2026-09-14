import logging
import threading
from dataclasses import dataclass

from openai import OpenAI, APITimeoutError, APIConnectionError, InternalServerError, RateLimitError

from src.providers.circuit_breaker import CircuitBreaker
from src.providers.clients import nim_client, groq_client, groq_client_secondary
from src.config import settings
from src.tracing import provider_call_span

logger = logging.getLogger(__name__)


class EmptyCompletionError(RuntimeError):
    """Raised when a provider returns a completion with no content. Subclasses
    RuntimeError so any old call site catching RuntimeError still works, but
    is listed in RETRYABLE below so _run_chain treats it as failover-worthy
    instead of raising immediately: an empty completion (clipped by a content
    filter, a hosted-model hiccup, etc.) is exactly the kind of single-provider
    flakiness the chain exists to route around, not a sign the whole request
    is malformed the way a 400 is."""


RETRYABLE = (APITimeoutError, RateLimitError, APIConnectionError, InternalServerError, EmptyCompletionError)

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
        content = response.choices[0].message.content
        if content is None:
            raise EmptyCompletionError(f"{provider_name} returned an empty completion")
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
    last_error: Exception | None = None
    for i, link in enumerate(chain):
        is_last = i == len(chain) - 1
        breaker = _breaker_for(link["name"], link["model"])
        if not is_last and not breaker.allow():
            logger.warning("%s circuit open, skipping straight to %s", link["name"], chain[i + 1]["name"])
            continue
        try:
            content = link["call"]()
            logger.info("served by %s (%s)", link["name"], link["model"])
            breaker.record_success()
            return CompletionResult(content=content, provider=link["name"], model=link["model"])
        except Exception as error:
            last_error = error
            if not link["is_transient"](error):
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
    # every link was skipped via an open breaker except (by construction)
    # the last one, which either returned above or raised above — this is
    # unreachable, but guards against a future edit changing that invariant.
    if last_error is not None:
        raise RuntimeError("provider chain exhausted") from last_error
    raise RuntimeError("provider chain was empty")


def generate_main(messages: list[dict], temperature: float = 0.2, max_tokens: int = 1024) -> CompletionResult:
    timeout = settings.generation_timeout_seconds
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
    max_tokens: int = 256,
    timeout_seconds: float | None = None,
) -> CompletionResult:
    timeout = timeout_seconds or settings.planner_timeout_seconds
    chain = [
        _openai_link(nim_client, settings.nim_planner_model, "nim", messages, temperature, max_tokens, "planner", timeout),
        _openai_link(groq_client, settings.groq_planner_model, "groq", messages, temperature, max_tokens, "planner", timeout),
    ]
    return _run_chain(chain)
