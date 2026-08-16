from unittest.mock import MagicMock, patch

import pytest
from openai import APITimeoutError, BadRequestError

from src.providers.llm import generate_main, generate_planner


def _mock_completion(content: str) -> MagicMock:
    completion = MagicMock()
    completion.choices[0].message.content = content
    return completion


def _retryable_openai_error() -> APITimeoutError:
    # src/providers/llm.py's RETRYABLE tuple deliberately only covers
    # transient/infrastructure failures (timeouts, rate limits, connection
    # errors, 5xx) — a generic openai.APIError (which also covers
    # non-transient errors like BadRequestError/AuthenticationError) is
    # NOT retried or failed over, by design. Use an actually-retryable
    # exception so these tests exercise the real failover path.
    return APITimeoutError(request=MagicMock())


# --- generate_main: groq (primary account) -> groq (secondary account) -> nim ---


@patch("src.providers.llm.nim_client")
@patch("src.providers.llm.groq_client_secondary")
@patch("src.providers.llm.groq_client")
def test_uses_groq_primary_account_when_it_succeeds(mock_groq, mock_groq_secondary, mock_nim):
    mock_groq.chat.completions.create.return_value = _mock_completion("groq answer")
    result = generate_main([{"role": "user", "content": "hi"}])
    assert result.provider == "groq"
    assert result.content == "groq answer"
    assert mock_groq_secondary.chat.completions.create.called is False
    assert mock_nim.chat.completions.create.called is False


@patch("src.providers.llm.nim_client")
@patch("src.providers.llm.groq_client_secondary")
@patch("src.providers.llm.groq_client")
def test_falls_back_to_groq_secondary_account_when_primary_fails(mock_groq, mock_groq_secondary, mock_nim):
    mock_groq.chat.completions.create.side_effect = _retryable_openai_error()
    mock_groq_secondary.chat.completions.create.return_value = _mock_completion("secondary groq answer")
    result = generate_main([{"role": "user", "content": "hi"}])
    assert result.provider == "groq-secondary"
    assert result.content == "secondary groq answer"
    assert mock_nim.chat.completions.create.called is False


@patch("src.providers.llm.nim_client")
@patch("src.providers.llm.groq_client_secondary")
@patch("src.providers.llm.groq_client")
def test_falls_back_to_nim_when_both_groq_accounts_fail(mock_groq, mock_groq_secondary, mock_nim):
    mock_groq.chat.completions.create.side_effect = _retryable_openai_error()
    mock_groq_secondary.chat.completions.create.side_effect = _retryable_openai_error()
    mock_nim.chat.completions.create.return_value = _mock_completion("nim answer")
    result = generate_main([{"role": "user", "content": "hi"}])
    assert result.provider == "nim"
    assert result.content == "nim answer"


@patch("src.providers.llm.nim_client")
@patch("src.providers.llm.groq_client_secondary")
@patch("src.providers.llm.groq_client")
def test_raises_clear_error_when_all_three_generation_links_fail(mock_groq, mock_groq_secondary, mock_nim):
    mock_groq.chat.completions.create.side_effect = _retryable_openai_error()
    mock_groq_secondary.chat.completions.create.side_effect = _retryable_openai_error()
    mock_nim.chat.completions.create.side_effect = _retryable_openai_error()
    with pytest.raises(RuntimeError, match=r"all providers in chain \(groq -> groq-secondary -> nim\) failed"):
        generate_main([{"role": "user", "content": "hi"}])


@patch("src.providers.llm.nim_client")
@patch("src.providers.llm.groq_client_secondary")
@patch("src.providers.llm.groq_client")
def test_non_transient_error_propagates_without_touching_rest_of_generation_chain(
    mock_groq, mock_groq_secondary, mock_nim
):
    bad_request = BadRequestError("invalid request", response=MagicMock(status_code=400), body=None)
    mock_groq.chat.completions.create.side_effect = bad_request
    with pytest.raises(BadRequestError):
        generate_main([{"role": "user", "content": "hi"}])
    assert mock_groq_secondary.chat.completions.create.called is False
    assert mock_nim.chat.completions.create.called is False


@patch("src.providers.llm.nim_client")
@patch("src.providers.llm.groq_client_secondary")
@patch("src.providers.llm.groq_client")
def test_non_transient_error_on_last_generation_link_propagates_as_itself(mock_groq, mock_groq_secondary, mock_nim):
    # a non-transient error from the LAST link in the chain should propagate
    # as itself, not get wrapped in the generic "all providers failed"
    # RuntimeError — there's nothing left to fall back to, and the specific
    # error type still carries useful information.
    bad_request = BadRequestError("invalid request", response=MagicMock(status_code=400), body=None)
    mock_groq.chat.completions.create.side_effect = _retryable_openai_error()
    mock_groq_secondary.chat.completions.create.side_effect = _retryable_openai_error()
    mock_nim.chat.completions.create.side_effect = bad_request
    with pytest.raises(BadRequestError):
        generate_main([{"role": "user", "content": "hi"}])


def test_groq_client_and_secondary_are_configured_with_distinct_api_keys():
    # regression guard: the two groq links in generate_main's chain only
    # protect against per-key rate caps if clients.py actually constructed
    # them against two different account keys, not the same key twice.
    from src.config import settings
    from src.providers.clients import groq_client, groq_client_secondary

    assert groq_client.api_key == settings.groq_api_key
    assert groq_client_secondary.api_key == settings.groq_api_key_secondary
    assert groq_client.api_key != groq_client_secondary.api_key


# --- generate_planner: nim (primary) -> groq (fallback) ---


@patch("src.providers.llm.groq_client")
@patch("src.providers.llm.nim_client")
def test_uses_nim_as_planner_primary_when_it_succeeds(mock_nim, mock_groq):
    mock_nim.chat.completions.create.return_value = _mock_completion("nim planner answer")
    result = generate_planner([{"role": "user", "content": "rewrite this"}])
    assert result.provider == "nim"
    assert result.content == "nim planner answer"
    assert mock_groq.chat.completions.create.called is False

    called_model = mock_nim.chat.completions.create.call_args.kwargs["model"]
    from src.config import settings

    assert called_model == settings.nim_planner_model


@patch("src.providers.llm.groq_client")
@patch("src.providers.llm.nim_client")
def test_falls_back_to_groq_when_nim_planner_fails(mock_nim, mock_groq):
    mock_nim.chat.completions.create.side_effect = _retryable_openai_error()
    mock_groq.chat.completions.create.return_value = _mock_completion("groq planner answer")
    result = generate_planner([{"role": "user", "content": "rewrite this"}])
    assert result.provider == "groq"
    assert result.content == "groq planner answer"

    called_model = mock_groq.chat.completions.create.call_args.kwargs["model"]
    from src.config import settings

    assert called_model == settings.groq_planner_model


@patch("src.providers.llm.groq_client")
@patch("src.providers.llm.nim_client")
def test_raises_clear_error_when_both_planner_links_fail(mock_nim, mock_groq):
    mock_nim.chat.completions.create.side_effect = _retryable_openai_error()
    mock_groq.chat.completions.create.side_effect = _retryable_openai_error()
    with pytest.raises(RuntimeError, match=r"all providers in chain \(nim -> groq\) failed"):
        generate_planner([{"role": "user", "content": "rewrite this"}])


@patch("src.providers.llm.groq_client")
@patch("src.providers.llm.nim_client")
def test_non_transient_error_propagates_without_touching_rest_of_planner_chain(mock_nim, mock_groq):
    bad_request = BadRequestError("invalid request", response=MagicMock(status_code=400), body=None)
    mock_nim.chat.completions.create.side_effect = bad_request
    with pytest.raises(BadRequestError):
        generate_planner([{"role": "user", "content": "rewrite this"}])
    assert mock_groq.chat.completions.create.called is False
