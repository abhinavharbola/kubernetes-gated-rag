from unittest.mock import MagicMock, patch

import pytest
from openai import APITimeoutError

from src.providers.llm import generate_main, generate_planner


def _mock_completion(content: str) -> MagicMock:
    completion = MagicMock()
    completion.choices[0].message.content = content
    return completion


def _retryable_error() -> APITimeoutError:
    # src/llm.py's RETRYABLE tuple deliberately only covers transient/
    # infrastructure failures (timeouts, rate limits, connection errors,
    # 5xx) — a generic openai.APIError (which also covers non-transient
    # errors like BadRequestError/AuthenticationError) is NOT retried or
    # failed over, by design. Use an actually-retryable exception here so
    # these tests exercise the real failover path instead of the
    # immediate-propagation path.
    return APITimeoutError(request=MagicMock())


@patch("src.providers.llm.groq_client")
@patch("src.providers.llm.nim_client")
def test_uses_primary_when_it_succeeds(mock_nim, mock_groq):
    mock_nim.chat.completions.create.return_value = _mock_completion("nim answer")
    result = generate_main([{"role": "user", "content": "hi"}])
    assert result.provider == "nim"
    assert result.content == "nim answer"
    assert mock_groq.chat.completions.create.called is False


@patch("src.providers.llm.groq_client")
@patch("src.providers.llm.nim_client")
def test_falls_back_to_groq_when_primary_fails(mock_nim, mock_groq):
    mock_nim.chat.completions.create.side_effect = _retryable_error()
    mock_groq.chat.completions.create.return_value = _mock_completion("groq answer")
    result = generate_main([{"role": "user", "content": "hi"}])
    assert result.provider == "groq"
    assert result.content == "groq answer"


@patch("src.providers.llm.groq_client")
@patch("src.providers.llm.nim_client")
def test_raises_clear_error_when_both_providers_fail(mock_nim, mock_groq):
    mock_nim.chat.completions.create.side_effect = _retryable_error()
    mock_groq.chat.completions.create.side_effect = _retryable_error()
    with pytest.raises(RuntimeError, match="both nim and groq failed"):
        generate_main([{"role": "user", "content": "hi"}])


@patch("src.providers.llm.groq_client")
@patch("src.providers.llm.nim_client")
def test_non_transient_error_propagates_without_failover(mock_nim, mock_groq):
    from openai import BadRequestError

    bad_request = BadRequestError("invalid request", response=MagicMock(status_code=400), body=None)
    mock_nim.chat.completions.create.side_effect = bad_request
    with pytest.raises(BadRequestError):
        generate_main([{"role": "user", "content": "hi"}])
    assert mock_groq.chat.completions.create.called is False


@patch("src.providers.llm.groq_client")
@patch("src.providers.llm.nim_client")
def test_planner_never_touches_main_generation_models(mock_nim, mock_groq):
    mock_nim.chat.completions.create.return_value = _mock_completion("planner answer")
    generate_planner([{"role": "user", "content": "rewrite this"}])
    called_model = mock_nim.chat.completions.create.call_args.kwargs["model"]
    assert "8b" in called_model