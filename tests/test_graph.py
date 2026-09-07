from unittest.mock import MagicMock, patch

from src.graph import canonicalize_node, exact_cache_node, late_exact_cache_node, rewrite_with_history_node


def _mock_result(content: str) -> MagicMock:
    result = MagicMock()
    result.content = content
    result.provider = "groq"
    result.model = "openai/gpt-oss-20b"
    return result


def test_canonicalize_node_is_deterministic_and_does_not_call_planner():
    with patch("src.graph.generate_planner") as planner:
        result = canonicalize_node({"standalone_question": "  what's   a Pod?  "})
    assert result["canonical_question"] == "what's a Pod?"
    planner.assert_not_called()


def test_rewrite_with_history_node_uses_planner_output_normally():
    with patch("src.graph.generate_planner", return_value=_mock_result("what is a StatefulSet")):
        result = rewrite_with_history_node(
            {
                "raw_message": "what about that?",
                "chat_history": [{"role": "user", "content": "what is a Deployment?"}],
            }
        )
    assert result["standalone_question"] == "what is a StatefulSet"


def test_exact_cache_skips_remote_calls_on_first_turn_hit():
    with patch("src.graph.exact_cache_get", return_value="cached answer") as cache_get:
        with patch("src.graph.safety_gate") as safety:
            result = exact_cache_node({"raw_message": "what is a Pod?", "chat_history": []})
    assert result["answer"] == "cached answer"
    assert result["cache_layer"] == "exact"
    cache_get.assert_called_once_with("what is a Pod?")
    safety.assert_not_called()


def test_exact_cache_is_not_used_before_rewrite_when_history_exists():
    with patch("src.graph.exact_cache_get") as cache_get:
        result = exact_cache_node({"raw_message": "what about that?", "chat_history": [{"role": "user", "content": "Deployment"}]})
    assert "cache_layer" not in result
    cache_get.assert_not_called()


def test_late_exact_cache_is_used_after_history_rewrite():
    with patch("src.graph.exact_cache_get", return_value="cached answer") as cache_get:
        result = late_exact_cache_node(
            {
                "standalone_question": "what is a Deployment?",
                "chat_history": [{"role": "user", "content": "Deployment"}],
            }
        )
    assert result["cache_layer"] == "exact"
    assert result["answer"] == "cached answer"
    cache_get.assert_called_once_with("what is a Deployment?")
