from unittest.mock import MagicMock, patch

import pytest
from google.genai.errors import ClientError, ServerError
from openai import APITimeoutError, BadRequestError

from src.providers.llm import generate_main, generate_planner


def _mock_completion(content: str) -> MagicMock:
    completion = MagicMock()
    completion.choices[0].message.content = content
    return completion


def _mock_gemini_response(text: str) -> MagicMock:
    response = MagicMock()
    response.text = text
    return response


def _retryable_openai_error() -> APITimeoutError:
    # src/providers/llm.py's RETRYABLE tuple deliberately only covers
    # transient/infrastructure failures (timeouts, rate limits, connection
    # errors, 5xx) — a generic openai.APIError (which also covers
    # non-transient errors like BadRequestError/AuthenticationError) is
    # NOT retried or failed over, by design. Use an actually-retryable
    # exception so these tests exercise the real failover path.
    return APITimeoutError(request=MagicMock())


def _retryable_gemini_error() -> ServerError:
    return ServerError(503, {"error": {"message": "server error"}})


@patch("src.providers.llm.gemini_client")
@patch("src.providers.llm.groq_client")
@patch("src.providers.llm.nim_client")
def test_uses_groq_as_primary_when_it_succeeds(mock_nim, mock_groq, mock_gemini):
    mock_groq.chat.completions.create.return_value = _mock_completion("groq answer")
    result = generate_main([{"role": "user", "content": "hi"}])
    assert result.provider == "groq"
    assert result.content == "groq answer"
    assert mock_nim.chat.completions.create.called is False
    assert mock_gemini.models.generate_content.called is False


@patch("src.providers.llm.gemini_client")
@patch("src.providers.llm.groq_client")
@patch("src.providers.llm.nim_client")
def test_falls_back_to_nim_when_groq_fails(mock_nim, mock_groq, mock_gemini):
    mock_groq.chat.completions.create.side_effect = _retryable_openai_error()
    mock_nim.chat.completions.create.return_value = _mock_completion("nim answer")
    result = generate_main([{"role": "user", "content": "hi"}])
    assert result.provider == "nim"
    assert result.content == "nim answer"
    assert mock_gemini.models.generate_content.called is False


@patch("src.providers.llm.gemini_client")
@patch("src.providers.llm.groq_client")
@patch("src.providers.llm.nim_client")
def test_falls_back_to_gemini_when_groq_and_nim_both_fail(mock_nim, mock_groq, mock_gemini):
    mock_groq.chat.completions.create.side_effect = _retryable_openai_error()
    mock_nim.chat.completions.create.side_effect = _retryable_openai_error()
    mock_gemini.models.generate_content.return_value = _mock_gemini_response("gemini answer")
    result = generate_main([{"role": "user", "content": "hi"}])
    assert result.provider == "gemini"
    assert result.content == "gemini answer"


@patch("src.providers.llm.gemini_client")
@patch("src.providers.llm.groq_client")
@patch("src.providers.llm.nim_client")
def test_gemini_transient_error_also_triggers_the_normal_retry(mock_nim, mock_groq, mock_gemini):
    mock_groq.chat.completions.create.side_effect = _retryable_openai_error()
    mock_nim.chat.completions.create.side_effect = _retryable_openai_error()
    # rate-limited (429) is Gemini's other transient case, distinct from a
    # 5xx ServerError, and must trigger the same fallback behavior.
    mock_gemini.models.generate_content.side_effect = [
        ClientError(429, {"error": {"message": "rate limited"}}),
        _mock_gemini_response("gemini answer after retry"),
    ]
    result = generate_main([{"role": "user", "content": "hi"}])
    assert result.provider == "gemini"
    assert result.content == "gemini answer after retry"


@patch("src.providers.llm.gemini_client")
@patch("src.providers.llm.groq_client")
@patch("src.providers.llm.nim_client")
def test_raises_clear_error_when_all_three_providers_fail(mock_nim, mock_groq, mock_gemini):
    mock_groq.chat.completions.create.side_effect = _retryable_openai_error()
    mock_nim.chat.completions.create.side_effect = _retryable_openai_error()
    mock_gemini.models.generate_content.side_effect = _retryable_gemini_error()
    with pytest.raises(RuntimeError, match="all providers in chain \\(groq -> nim -> gemini\\) failed"):
        generate_main([{"role": "user", "content": "hi"}])


@patch("src.providers.llm.gemini_client")
@patch("src.providers.llm.groq_client")
@patch("src.providers.llm.nim_client")
def test_non_transient_error_propagates_without_touching_the_rest_of_the_chain(mock_nim, mock_groq, mock_gemini):
    bad_request = BadRequestError("invalid request", response=MagicMock(status_code=400), body=None)
    mock_groq.chat.completions.create.side_effect = bad_request
    with pytest.raises(BadRequestError):
        generate_main([{"role": "user", "content": "hi"}])
    assert mock_nim.chat.completions.create.called is False
    assert mock_gemini.models.generate_content.called is False


@patch("src.providers.llm.gemini_client")
@patch("src.providers.llm.groq_client")
@patch("src.providers.llm.nim_client")
def test_gemini_non_transient_error_propagates_as_final_failure(mock_nim, mock_groq, mock_gemini):
    # a non-transient error from the LAST link in the chain should propagate
    # as itself, not get wrapped in the generic "all providers failed"
    # RuntimeError — there's nothing left to fall back to, and the specific
    # error type still carries useful information (e.g. this is genuinely a
    # bad request, not an exhausted chain).
    mock_groq.chat.completions.create.side_effect = _retryable_openai_error()
    mock_nim.chat.completions.create.side_effect = _retryable_openai_error()
    mock_gemini.models.generate_content.side_effect = ClientError(400, {"error": {"message": "bad request"}})
    with pytest.raises(ClientError):
        generate_main([{"role": "user", "content": "hi"}])


@patch("src.providers.llm.gemini_client")
@patch("src.providers.llm.groq_client")
@patch("src.providers.llm.nim_client")
def test_planner_uses_the_same_chain_order_with_planner_models(mock_nim, mock_groq, mock_gemini):
    mock_groq.chat.completions.create.return_value = _mock_completion("planner answer")
    generate_planner([{"role": "user", "content": "rewrite this"}])
    called_model = mock_groq.chat.completions.create.call_args.kwargs["model"]
    from src.config import settings

    assert called_model == settings.groq_planner_model
    assert mock_nim.chat.completions.create.called is False


@patch("src.providers.llm.gemini_client")
@patch("src.providers.llm.groq_client")
@patch("src.providers.llm.nim_client")
def test_gemini_call_translates_system_message_and_roles(mock_nim, mock_groq, mock_gemini):
    mock_groq.chat.completions.create.side_effect = _retryable_openai_error()
    mock_nim.chat.completions.create.side_effect = _retryable_openai_error()
    mock_gemini.models.generate_content.return_value = _mock_gemini_response("ok")

    generate_main(
        [
            {"role": "system", "content": "you are a helpful assistant"},
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": "second question"},
        ]
    )

    call_kwargs = mock_gemini.models.generate_content.call_args.kwargs
    assert call_kwargs["config"].system_instruction == "you are a helpful assistant"
    contents = call_kwargs["contents"]
    assert [c["role"] for c in contents] == ["user", "model", "user"]
    assert contents[0]["parts"] == [{"text": "first question"}]
