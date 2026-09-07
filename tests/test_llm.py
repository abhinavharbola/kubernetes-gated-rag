from unittest.mock import MagicMock, patch

import pytest
from openai import APITimeoutError, BadRequestError

from src.providers.llm import generate_main, generate_planner


def _mock_completion(content: str) -> MagicMock:
    completion = MagicMock()
    completion.choices[0].message.content = content
    return completion


def _retryable_openai_error() -> APITimeoutError:
    return APITimeoutError(request=MagicMock())


@patch("src.providers.llm.nim_client")
@patch("src.providers.llm.groq_client_secondary")
@patch("src.providers.llm.groq_client")
def test_uses_groq_primary_account_when_it_succeeds(mock_groq, mock_groq_secondary, mock_nim):
    mock_groq.chat.completions.create.return_value = _mock_completion("groq answer")
    result = generate_main([{"role": "user", "content": "hi"}])
    assert result.provider == "groq"
    mock_groq_secondary.chat.completions.create.assert_not_called()
    mock_nim.chat.completions.create.assert_not_called()


@patch("src.providers.llm.nim_client")
@patch("src.providers.llm.groq_client_secondary")
@patch("src.providers.llm.groq_client")
def test_falls_back_immediately_to_secondary_on_transient_failure(mock_groq, mock_groq_secondary, mock_nim):
    mock_groq.chat.completions.create.side_effect = _retryable_openai_error()
    mock_groq_secondary.chat.completions.create.return_value = _mock_completion("secondary")
    result = generate_main([{"role": "user", "content": "hi"}])
    assert result.provider == "groq-secondary"
    assert mock_groq.chat.completions.create.call_count == 1


@patch("src.providers.llm.nim_client")
@patch("src.providers.llm.groq_client_secondary")
@patch("src.providers.llm.groq_client")
def test_falls_back_to_nim_when_both_groq_links_fail(mock_groq, mock_groq_secondary, mock_nim):
    mock_groq.chat.completions.create.side_effect = _retryable_openai_error()
    mock_groq_secondary.chat.completions.create.side_effect = _retryable_openai_error()
    mock_nim.chat.completions.create.return_value = _mock_completion("nim")
    result = generate_main([{"role": "user", "content": "hi"}])
    assert result.provider == "nim"
    assert mock_groq.chat.completions.create.call_count == 1
    assert mock_groq_secondary.chat.completions.create.call_count == 1


@patch("src.providers.llm.groq_client")
@patch("src.providers.llm.nim_client")
def test_planner_uses_nim_then_groq(mock_nim, mock_groq):
    mock_nim.chat.completions.create.return_value = _mock_completion("nim planner")
    result = generate_planner([{"role": "user", "content": "rewrite this"}])
    assert result.provider == "nim"
    mock_groq.chat.completions.create.assert_not_called()


@patch("src.providers.llm.groq_client")
@patch("src.providers.llm.nim_client")
def test_non_transient_error_does_not_fail_over(mock_nim, mock_groq):
    bad_request = BadRequestError("invalid request", response=MagicMock(status_code=400), body=None)
    mock_nim.chat.completions.create.side_effect = bad_request
    with pytest.raises(BadRequestError):
        generate_planner([{"role": "user", "content": "rewrite this"}])
    mock_groq.chat.completions.create.assert_not_called()
