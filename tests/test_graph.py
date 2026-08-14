from unittest.mock import MagicMock, patch

from src.graph import canonicalize_node, rewrite_with_history_node


def _mock_result(content: str) -> MagicMock:
    result = MagicMock()
    result.content = content
    result.provider = "groq"
    result.model = "openai/gpt-oss-20b"
    return result


@patch("src.graph.generate_planner")
def test_canonicalize_node_uses_planner_output_normally(mock_planner):
    mock_planner.return_value = _mock_result("what is a kubernetes pod")
    state = {"standalone_question": "what's a pod?"}
    result = canonicalize_node(state)
    assert result["canonical_question"] == "what is a kubernetes pod"


@patch("src.graph.generate_planner")
def test_canonicalize_node_falls_back_to_standalone_question_on_empty_planner_output(mock_planner):
    # regression test: a live run hit exactly this — the planner (Groq
    # openai/gpt-oss-20b) returned an empty completion for a well-formed
    # question, canonical_question ended up as "", and that empty string
    # reached Gemini's embed_content in semantic_cache_node, which rejects
    # empty input with an opaque 400 ("empty Part") several frames deep in
    # a retry stack. canonicalize_node must never hand back an empty
    # canonical_question.
    mock_planner.return_value = _mock_result("")
    state = {"standalone_question": "what's a pod?"}
    result = canonicalize_node(state)
    assert result["canonical_question"] == "what's a pod?"


@patch("src.graph.generate_planner")
def test_canonicalize_node_falls_back_on_whitespace_only_planner_output(mock_planner):
    mock_planner.return_value = _mock_result("   \n  ")
    state = {"standalone_question": "what's a pod?"}
    result = canonicalize_node(state)
    assert result["canonical_question"] == "what's a pod?"


def test_rewrite_with_history_node_skips_planner_entirely_when_no_history():
    state = {"raw_message": "what's a pod?", "chat_history": []}
    result = rewrite_with_history_node(state)
    assert result["standalone_question"] == "what's a pod?"


@patch("src.graph.generate_planner")
def test_rewrite_with_history_node_falls_back_to_raw_message_on_empty_planner_output(mock_planner):
    mock_planner.return_value = _mock_result("")
    state = {
        "raw_message": "what about StatefulSets?",
        "chat_history": [{"role": "user", "content": "what is a Deployment?"}],
    }
    result = rewrite_with_history_node(state)
    assert result["standalone_question"] == "what about StatefulSets?"


@patch("src.graph.generate_planner")
def test_rewrite_with_history_node_uses_planner_output_normally(mock_planner):
    mock_planner.return_value = _mock_result("what is a StatefulSet")
    state = {
        "raw_message": "what about that?",
        "chat_history": [{"role": "user", "content": "what is a Deployment?"}],
    }
    result = rewrite_with_history_node(state)
    assert result["standalone_question"] == "what is a StatefulSet"
