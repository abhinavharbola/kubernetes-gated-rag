from unittest.mock import patch

import pytest

from src.retrieval.rerank import RerankUnavailableError, rerank_and_gate


def _candidate(text, manifest_kind=None, manifest_name=None):
    return {"text": text, "metadata": {"manifest_kind": manifest_kind, "manifest_name": manifest_name}}


@patch("src.retrieval.rerank.settings")
@patch("src.retrieval.rerank._ranker")
def test_empty_candidates_returns_empty(mock_ranker, mock_settings):
    assert rerank_and_gate("how do I create a resource?", []) == []


@patch("src.retrieval.rerank.settings")
@patch("src.retrieval.rerank._ranker")
def test_below_threshold_candidates_are_dropped(mock_ranker, mock_settings):
    mock_settings.rerank_score_threshold = 0.5
    mock_settings.rerank_fail_closed = True
    mock_ranker.rerank.return_value = [{"id": 0, "score": 0.9}, {"id": 1, "score": 0.2}]
    survivors = rerank_and_gate("question", [_candidate("relevant chunk"), _candidate("noisy chunk")])
    assert [c["text"] for c in survivors] == ["relevant chunk"]


@patch("src.retrieval.rerank.settings")
@patch("src.retrieval.rerank._ranker")
def test_ranker_crash_raises_unavailable_by_default(mock_ranker, mock_settings):
    # A FlashRank crash is an infrastructure failure, not a real "nothing is
    # relevant" verdict. It must not be silently swallowed into [] and then
    # cached by graph.py as a genuine "no grounded documentation" answer.
    mock_settings.rerank_score_threshold = 0.5
    mock_settings.rerank_fail_closed = True
    mock_ranker.rerank.side_effect = RuntimeError("ONNX load failed")
    with pytest.raises(RerankUnavailableError):
        rerank_and_gate("question", [_candidate("chunk")])


@patch("src.retrieval.rerank.settings")
@patch("src.retrieval.rerank._ranker")
def test_ranker_failure_fallback_still_applies_a_threshold(mock_ranker, mock_settings):
    # The fallback path (FlashRank crashed, rerank_fail_closed explicitly
    # disabled) must still gate on something - it can't silently return
    # every candidate unfiltered, which would defeat the relevance gate
    # entirely. It falls back to filtering on raw retrieval_score instead
    # of a rerank_score, since no rerank score exists in this path.
    mock_settings.rerank_fail_closed = False
    mock_settings.rerank_fallback_score_threshold = 0.5
    mock_ranker.rerank.side_effect = RuntimeError("ONNX load failed")
    candidates = [{**_candidate("low"), "retrieval_score": 0.3}, {**_candidate("high"), "retrieval_score": 0.7}]
    assert [c["text"] for c in rerank_and_gate("question", candidates)] == ["high"]
