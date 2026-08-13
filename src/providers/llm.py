import logging
from dataclasses import dataclass

from openai import (
    OpenAI,
    APITimeoutError,
    APIConnectionError,
    InternalServerError,
    RateLimitError,
)
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.providers.clients import nim_client, groq_client
from src.config import settings
from src.tracing import provider_call_span

logger = logging.getLogger(__name__)

# Only transient/infrastructure failures are retried and failed-over.
# APIError (the base SDK exception) also covers BadRequestError,
# AuthenticationError, PermissionDeniedError, NotFoundError, etc: none
# of those are transient, retrying or failing over to a second provider
# just repeats the same failure twice at double the latency. Those are
# left to propagate immediately instead.
RETRYABLE = (APITimeoutError, RateLimitError, APIConnectionError, InternalServerError)


@dataclass
class CompletionResult:
    content: str
    provider: str
    model: str


@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type(RETRYABLE),
    reraise=True,
)
def _call(
    client: OpenAI,
    model: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    provider_name: str,
    role: str,
) -> str:
    with provider_call_span(provider=provider_name, model=model, role=role):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content


def _complete_with_failover(
    messages: list[dict],
    primary_client: OpenAI,
    primary_model: str,
    primary_name: str,
    fallback_client: OpenAI,
    fallback_model: str,
    fallback_name: str,
    role: str,
    temperature: float = 0.2,
    max_tokens: int = 1024,
) -> CompletionResult:
    try:
        content = _call(primary_client, primary_model, messages, temperature, max_tokens, primary_name, role)
        logger.info("served by %s (%s)", primary_name, primary_model)
        return CompletionResult(content=content, provider=primary_name, model=primary_model)
    except RETRYABLE as primary_error:
        logger.warning("%s failed after retry, falling back to %s: %s", primary_name, fallback_name, primary_error)

    try:
        content = _call(fallback_client, fallback_model, messages, temperature, max_tokens, fallback_name, role)
        logger.info("served by %s (%s)", fallback_name, fallback_model)
        return CompletionResult(content=content, provider=fallback_name, model=fallback_model)
    except RETRYABLE as fallback_error:
        raise RuntimeError(
            f"both {primary_name} and {fallback_name} failed: {fallback_error}"
        ) from fallback_error


def generate_main(messages: list[dict], temperature: float = 0.2, max_tokens: int = 1024) -> CompletionResult:
    return _complete_with_failover(
        messages,
        primary_client=nim_client,
        primary_model=settings.nim_main_model,
        primary_name="nim",
        fallback_client=groq_client,
        fallback_model=settings.groq_main_model,
        fallback_name="groq",
        role="main",
        temperature=temperature,
        max_tokens=max_tokens,
    )


def generate_planner(messages: list[dict], temperature: float = 0.0, max_tokens: int = 256) -> CompletionResult:
    return _complete_with_failover(
        messages,
        primary_client=nim_client,
        primary_model=settings.nim_planner_model,
        primary_name="nim",
        fallback_client=groq_client,
        fallback_model=settings.groq_planner_model,
        fallback_name="groq",
        role="planner",
        temperature=temperature,
        max_tokens=max_tokens,
    )
