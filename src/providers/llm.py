import logging
from dataclasses import dataclass

from openai import OpenAI, APITimeoutError, APIConnectionError, InternalServerError, RateLimitError

from src.providers.clients import nim_client, groq_client, groq_client_secondary
from src.config import settings
from src.tracing import provider_call_span

logger = logging.getLogger(__name__)
RETRYABLE = (APITimeoutError, RateLimitError, APIConnectionError, InternalServerError)


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
            raise RuntimeError(f"{provider_name} returned an empty completion")
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
        try:
            content = link["call"]()
            logger.info("served by %s (%s)", link["name"], link["model"])
            return CompletionResult(content=content, provider=link["name"], model=link["model"])
        except Exception as error:
            if not link["is_transient"](error):
                raise
            if i == len(chain) - 1:
                names = " -> ".join(step["name"] for step in chain)
                raise RuntimeError(f"all providers in chain ({names}) failed: {error}") from error
            logger.warning(
                "%s failed, falling back immediately to %s: %s",
                link["name"],
                chain[i + 1]["name"],
                error,
            )
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
