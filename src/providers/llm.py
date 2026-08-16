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

from src.providers.clients import nim_client, groq_client, groq_client_secondary
from src.config import settings
from src.tracing import provider_call_span

logger = logging.getLogger(__name__)

# Only transient/infrastructure failures are retried and failed-over.
# APIError (the base SDK exception) also covers BadRequestError,
# AuthenticationError, PermissionDeniedError, NotFoundError, etc: none
# of those are transient, retrying or failing over to the next provider
# just repeats the same failure twice (or three times) at multiplied
# latency for something that will never succeed. Those propagate
# immediately instead, at whichever step in the chain they happened.
RETRYABLE = (APITimeoutError, RateLimitError, APIConnectionError, InternalServerError)

# Gemini is deliberately not part of either chain below — it's used only for
# embeddings (src/retrieval/embeddings.py) and as the eval judge
# (eval/run_eval.py), both isolated from these live generation/planning
# paths. See src/config.py for the reasoning.


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
def _call_openai(
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


def _openai_link(client: OpenAI, model: str, name: str, messages: list, temperature: float, max_tokens: int, role: str) -> dict:
    return {
        "name": name,
        "model": model,
        "call": lambda: _call_openai(client, model, messages, temperature, max_tokens, name, role),
        "is_transient": lambda error: isinstance(error, RETRYABLE),
    }


def _run_chain(chain: list[dict]) -> CompletionResult:
    for i, link in enumerate(chain):
        is_last = i == len(chain) - 1
        try:
            content = link["call"]()
            logger.info("served by %s (%s)", link["name"], link["model"])
            return CompletionResult(content=content, provider=link["name"], model=link["model"])
        except Exception as error:
            if not link["is_transient"](error):
                # non-transient failure (bad request, auth, ...): this
                # provider will never succeed for this request, and neither
                # will re-sending the identical request to the next one.
                # Propagate immediately rather than burning the rest of the
                # chain on something that can't be fixed by switching hosts.
                raise
            if is_last:
                names = " -> ".join(step["name"] for step in chain)
                raise RuntimeError(f"all providers in chain ({names}) failed: {error}") from error
            logger.warning(
                "%s failed after retry, falling back to %s: %s", link["name"], chain[i + 1]["name"], error
            )
    raise RuntimeError("provider chain was empty")  # unreachable, chains are always non-empty


def generate_main(messages: list[dict], temperature: float = 0.2, max_tokens: int = 1024) -> CompletionResult:
    # groq (primary account) -> groq (secondary account, same model) -> nim.
    # The first two links share a model and only differ by API key/account,
    # so a per-key rate cap on the primary doesn't immediately cost a hop to
    # the slower cross-vendor NIM link; nim is still there for a genuine
    # Groq-platform-wide outage.
    chain = [
        _openai_link(groq_client, settings.groq_main_model, "groq", messages, temperature, max_tokens, "main"),
        _openai_link(
            groq_client_secondary,
            settings.groq_main_model_secondary,
            "groq-secondary",
            messages,
            temperature,
            max_tokens,
            "main",
        ),
        _openai_link(nim_client, settings.nim_main_model, "nim", messages, temperature, max_tokens, "main"),
    ]
    return _run_chain(chain)


def generate_planner(messages: list[dict], temperature: float = 0.0, max_tokens: int = 256) -> CompletionResult:
    # nim primary, groq fallback — reversed from generate_main deliberately.
    chain = [
        _openai_link(nim_client, settings.nim_planner_model, "nim", messages, temperature, max_tokens, "planner"),
        _openai_link(groq_client, settings.groq_planner_model, "groq", messages, temperature, max_tokens, "planner"),
    ]
    return _run_chain(chain)
