from unittest.mock import MagicMock, patch

import pytest
from openai import APITimeoutError, BadRequestError

import src.providers.llm as llm_module
from src.providers.llm import generate_main, generate_planner


@pytest.fixture(autouse=True)
def reset_provider_breakers():
    # _provider_breakers is module-level and shared across the whole test
    # session; without this, failures recorded by one test (e.g. opening
    # the "groq" breaker) would silently change a later test's behavior.
    for breaker in llm_module._provider_breakers.values():
        breaker.record_success()
    yield
    for breaker in llm_module._provider_breakers.values():
        breaker.record_success()


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


@patch("src.providers.llm.nim_client")
@patch("src.providers.llm.groq_client_secondary")
@patch("src.providers.llm.groq_client")
def test_empty_completion_fails_over_to_next_provider(mock_groq, mock_groq_secondary, mock_nim):
    # Content=None used to raise a bare RuntimeError, which _run_chain's
    # is_transient check didn't recognize as failover-worthy, killing the
    # whole chain on the first provider's empty response instead of trying
    # the next one.
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
    # 2 consecutive transient nim failures should open its breaker
    # (PROVIDER_CIRCUIT_FAILURE_THRESHOLD default = 2).
    mock_nim.chat.completions.create.side_effect = _retryable_openai_error()
    mock_groq.chat.completions.create.return_value = _mock_completion("groq 1")
    generate_planner([{"role": "user", "content": "q1"}])
    mock_nim.chat.completions.create.side_effect = _retryable_openai_error()
    mock_groq.chat.completions.create.return_value = _mock_completion("groq 2")
    generate_planner([{"role": "user", "content": "q2"}])
    assert mock_nim.chat.completions.create.call_count == 2

    # third call: breaker should now be open, nim must not be called at all
    mock_groq.chat.completions.create.return_value = _mock_completion("groq 3")
    result = generate_planner([{"role": "user", "content": "q3"}])
    assert result.provider == "groq"
    assert mock_nim.chat.completions.create.call_count == 2  # unchanged


@patch("src.providers.llm.groq_client")
@patch("src.providers.llm.nim_client")
def test_open_breaker_never_skips_the_last_link(mock_nim, mock_groq):
    # open nim's breaker first
    mock_nim.chat.completions.create.side_effect = _retryable_openai_error()
    mock_groq.chat.completions.create.return_value = _mock_completion("groq")
    generate_planner([{"role": "user", "content": "q1"}])
    generate_planner([{"role": "user", "content": "q2"}])

    # now open groq's breaker too (it's the last link here, so even once
    # open it must still be attempted rather than the whole chain failing
    # with zero real attempts)
    mock_groq.chat.completions.create.side_effect = _retryable_openai_error()
    with pytest.raises(RuntimeError):
        generate_planner([{"role": "user", "content": "q3"}])
    with pytest.raises(RuntimeError):
        generate_planner([{"role": "user", "content": "q4"}])
    assert mock_groq.chat.completions.create.call_count >= 2
