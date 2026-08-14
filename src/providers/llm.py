import logging
from dataclasses import dataclass

from google.genai import types
from google.genai.errors import ClientError, ServerError
from openai import (
    OpenAI,
    APITimeoutError,
    APIConnectionError,
    InternalServerError,
    RateLimitError,
)
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, retry_if_exception

from src.providers.clients import nim_client, groq_client, gemini_client
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


def _is_gemini_transient(error: BaseException) -> bool:
    # ServerError covers Gemini's 5xx responses. ClientError covers both
    # 429 (genuinely transient, worth a retry/failover) and non-transient
    # 4xx like bad request or auth — only the former should trigger a
    # retry or move to the next link in the chain.
    if isinstance(error, ServerError):
        return True
    if isinstance(error, ClientError) and getattr(error, "code", None) == 429:
        return True
    return False


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


@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception(_is_gemini_transient),
    reraise=True,
)
def _call_gemini(model: str, messages: list[dict], temperature: float, max_tokens: int, role: str) -> str:
    # Gemini isn't OpenAI-compatible: no chat.completions endpoint, system
    # messages go in a dedicated config field rather than the messages list,
    # and roles are "user"/"model" rather than "user"/"assistant".
    system_instruction = None
    contents = []
    for message in messages:
        if message["role"] == "system":
            system_instruction = message["content"]
        else:
            role_name = "model" if message["role"] == "assistant" else "user"
            contents.append({"role": role_name, "parts": [{"text": message["content"]}]})

    with provider_call_span(provider="gemini", model=model, role=role):
        response = gemini_client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )
        return response.text


def _openai_link(client: OpenAI, model: str, name: str, messages: list, temperature: float, max_tokens: int, role: str) -> dict:
    return {
        "name": name,
        "model": model,
        "call": lambda: _call_openai(client, model, messages, temperature, max_tokens, name, role),
        "is_transient": lambda error: isinstance(error, RETRYABLE),
    }


def _gemini_link(model: str, messages: list, temperature: float, max_tokens: int, role: str) -> dict:
    return {
        "name": "gemini",
        "model": model,
        "call": lambda: _call_gemini(model, messages, temperature, max_tokens, role),
        "is_transient": _is_gemini_transient,
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
    chain = [
        _openai_link(groq_client, settings.groq_main_model, "groq", messages, temperature, max_tokens, "main"),
        _openai_link(nim_client, settings.nim_main_model, "nim", messages, temperature, max_tokens, "main"),
        _gemini_link(settings.gemini_main_model, messages, temperature, max_tokens, "main"),
    ]
    return _run_chain(chain)


def generate_planner(messages: list[dict], temperature: float = 0.0, max_tokens: int = 256) -> CompletionResult:
    chain = [
        _openai_link(groq_client, settings.groq_planner_model, "groq", messages, temperature, max_tokens, "planner"),
        _openai_link(nim_client, settings.nim_planner_model, "nim", messages, temperature, max_tokens, "planner"),
        _gemini_link(settings.gemini_planner_model, messages, temperature, max_tokens, "planner"),
    ]
    return _run_chain(chain)
