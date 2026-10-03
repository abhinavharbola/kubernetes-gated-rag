from unittest.mock import MagicMock, patch

import pytest
from openai import APITimeoutError, AuthenticationError, BadRequestError

import src.providers.llm as llm_module
from src.providers.llm import generate_main, generate_planner


@pytest.fixture(autouse=True)
def reset_provider_breakers():
    for breaker in llm_module._provider_breakers.values():
        breaker.record_success()
    yield
    for breaker in llm_module._provider_breakers.values():
        breaker.record_success()


def _mock_completion(content: str, finish_reason: str = "stop") -> MagicMock:
    completion = MagicMock()
    completion.choices[0].message.content = content
    completion.choices[0].finish_reason = finish_reason
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


@patch("src.providers.llm.nim_client")
@patch("src.providers.llm.groq_client_secondary")
@patch("src.providers.llm.groq_client")
def test_empty_completion_fails_over_to_next_provider(mock_groq, mock_groq_secondary, mock_nim):
    mock_groq.chat.completions.create.return_value = _mock_completion(None)
    mock_groq_secondary.chat.completions.create.return_value = _mock_completion("secondary answer")
    result = generate_main([{"role": "user", "content": "hi"}])
    assert result.provider == "groq-secondary"
    assert result.content == "secondary answer"


@patch("src.providers.llm.groq_client")
@patch("src.providers.llm.nim_client")
def test_non_transient_error_does_not_fail_over(mock_nim, mock_groq):
    bad_request = BadRequestError("invalid request", response=MagicMock(status_code=400), body=None)
    mock_nim.chat.completions.create.side_effect = bad_request
    with pytest.raises(BadRequestError):
        generate_planner([{"role": "user", "content": "rewrite this"}])
    mock_groq.chat.completions.create.assert_not_called()


@patch("src.providers.llm.groq_client")
@patch("src.providers.llm.nim_client")
def test_open_breaker_skips_provider_without_calling_it(mock_nim, mock_groq):
    mock_nim.chat.completions.create.side_effect = _retryable_openai_error()
    mock_groq.chat.completions.create.return_value = _mock_completion("groq 1")
    generate_planner([{"role": "user", "content": "q1"}])
    mock_nim.chat.completions.create.side_effect = _retryable_openai_error()
    mock_groq.chat.completions.create.return_value = _mock_completion("groq 2")
    generate_planner([{"role": "user", "content": "q2"}])
    assert mock_nim.chat.completions.create.call_count == 2

    mock_groq.chat.completions.create.return_value = _mock_completion("groq 3")
    result = generate_planner([{"role": "user", "content": "q3"}])
    assert result.provider == "groq"
    assert mock_nim.chat.completions.create.call_count == 2


@patch("src.providers.llm.groq_client")
@patch("src.providers.llm.nim_client")
def test_open_breaker_never_skips_the_last_link(mock_nim, mock_groq):
    mock_nim.chat.completions.create.side_effect = _retryable_openai_error()
    mock_groq.chat.completions.create.return_value = _mock_completion("groq")
    generate_planner([{"role": "user", "content": "q1"}])
    generate_planner([{"role": "user", "content": "q2"}])

    mock_groq.chat.completions.create.side_effect = _retryable_openai_error()
    with pytest.raises(RuntimeError):
        generate_planner([{"role": "user", "content": "q3"}])
    with pytest.raises(RuntimeError):
        generate_planner([{"role": "user", "content": "q4"}])
    assert mock_groq.chat.completions.create.call_count >= 2


@patch("src.providers.llm.nim_client")
@patch("src.providers.llm.groq_client_secondary")
@patch("src.providers.llm.groq_client")
def test_blank_completion_fails_over(mock_groq, mock_groq_secondary, mock_nim):
    mock_groq.chat.completions.create.return_value = _mock_completion("   ")
    mock_groq_secondary.chat.completions.create.return_value = _mock_completion("real answer")
    result = generate_main([{"role": "user", "content": "hi"}])
    assert result.provider == "groq-secondary"


@patch("src.providers.llm.nim_client")
@patch("src.providers.llm.groq_client_secondary")
@patch("src.providers.llm.groq_client")
def test_truncated_completion_is_never_returned(mock_groq, mock_groq_secondary, mock_nim):
    mock_groq.chat.completions.create.return_value = _mock_completion("half an ans", finish_reason="length")
    mock_groq_secondary.chat.completions.create.return_value = _mock_completion("half an ans", finish_reason="length")
    mock_nim.chat.completions.create.return_value = _mock_completion("half an ans", finish_reason="length")
    with pytest.raises(RuntimeError):
        generate_main([{"role": "user", "content": "hi"}])


@patch("src.providers.llm.nim_client")
@patch("src.providers.llm.groq_client_secondary")
@patch("src.providers.llm.groq_client")
def test_provider_auth_failure_fails_over_to_the_next_provider(mock_groq, mock_groq_secondary, mock_nim):
    mock_groq.chat.completions.create.side_effect = AuthenticationError(
        "bad key", response=MagicMock(status_code=401), body=None
    )
    mock_groq_secondary.chat.completions.create.return_value = _mock_completion("secondary answer")
    result = generate_main([{"role": "user", "content": "hi"}])
    assert result.provider == "groq-secondary"


@patch("src.providers.llm.groq_client")
@patch("src.providers.llm.nim_client")
def test_default_token_budgets_come_from_settings(mock_nim, mock_groq):
    from src.config import settings

    mock_nim.chat.completions.create.return_value = _mock_completion("ok")
    generate_planner([{"role": "user", "content": "x"}])
    assert mock_nim.chat.completions.create.call_args.kwargs["max_tokens"] == settings.planner_max_tokens


@patch("src.providers.llm.groq_client")
@patch("src.providers.llm.nim_client")
def test_non_transient_error_does_not_leave_a_half_open_breaker_stuck(mock_nim, mock_groq):
    bad_request = BadRequestError("invalid request", response=MagicMock(status_code=400), body=None)
    for _ in range(3):
        mock_nim.chat.completions.create.side_effect = bad_request
        with pytest.raises(BadRequestError):
            generate_planner([{"role": "user", "content": "x"}])
    assert mock_nim.chat.completions.create.call_count == 3
