from unittest.mock import MagicMock, patch

import pytest

import src.retrieval.cache as cache_module
from src.retrieval.cache import (
    embed_canonical_question,
    ensure_semantic_cache_indexes,
    exact_cache_clear,
    exact_cache_count,
    exact_cache_get,
    exact_cache_set,
    normalize_exact,
    normalize_semantic,
    semantic_cache_get,
    semantic_cache_set,
)


@pytest.fixture(autouse=True)
def reset_index_flag():
    cache_module._indexes_ensured = False
    cache_module._next_index_attempt = 0.0
    yield
    cache_module._indexes_ensured = False
    cache_module._next_index_attempt = 0.0


@pytest.fixture(autouse=True)
def clear_exact_cache():
    from src.retrieval.cache import _exact_cache
    _exact_cache.clear()
    yield
    _exact_cache.clear()


@pytest.fixture(autouse=True)
def isolate_corpus_version_marker(tmp_path, monkeypatch):
    monkeypatch.setattr(cache_module, "_CORPUS_VERSION_MARKER", tmp_path / "corpus_version")


def test_normalize_collapses_whitespace_and_strips_trailing_punctuation():
    assert normalize_exact("  How Do   I Create a Resource?  ") == "How Do I Create a Resource"


def test_normalize_preserves_case_significant_flags():
    assert normalize_exact("kubectl get pods -l app=web") != normalize_exact("kubectl get pods -L app=web")
    assert normalize_exact("what does -A do") != normalize_exact("what does -a do")


def test_normalize_preserves_brackets_and_braces():
    first = "kubectl get pods -o jsonpath='{.items[0].metadata.name}'"
    second = "kubectl get pods -o jsonpath='.items0.metadata.name'"
    assert normalize_exact(first) != normalize_exact(second)


def test_normalize_does_not_collapse_different_questions():
    assert normalize_exact("how do I create a resource") != normalize_exact("how do I destroy a resource")


def test_normalize_preserves_hyphens_in_kubernetes_identifiers():
    assert normalize_exact("What does the kube-system namespace do?") != normalize_exact("What does the kubesystem namespace do?")


def test_normalize_preserves_slashes_in_api_groups():
    assert normalize_exact("what apiVersion is apps/v1?") != normalize_exact("what apiVersion is apps v1?")


def test_normalize_preserves_dots_in_field_paths():
    assert normalize_exact("what is pod.spec.containers?") != normalize_exact("what is pod spec containers?")


def test_normalize_still_strips_cosmetic_punctuation():
    assert normalize_exact('What is a Pod?') == normalize_exact("What is a Pod")


def test_normalize_semantic_only_collapses_whitespace():
    assert normalize_semantic("  What is a Pod?  ") == "What is a Pod?"


def test_exact_cache_roundtrip():
    exact_cache_set("How do I create a resource?", "answer text")
    assert exact_cache_get("How do I create a resource") == "answer text"


def test_exact_cache_miss_returns_none():
    assert exact_cache_get("nothing stored for this question") is None


def test_exact_cache_is_invalidated_by_cache_versions(monkeypatch):
    exact_cache_set("question", "answer")
    monkeypatch.setattr("src.retrieval.cache.settings.cache_policy_version", "some-other-policy-version")
    assert exact_cache_get("question") is None


@patch("src.retrieval.cache.embed_for_cache", return_value=[0.1] * 768)
def test_embed_canonical_question_delegates_to_embed_for_cache(mock_embed):
    assert embed_canonical_question("how a Deployment differs from a StatefulSet") == [0.1] * 768
    mock_embed.assert_called_once_with("how a Deployment differs from a StatefulSet")


@patch("src.retrieval.cache.qdrant_client")
def test_semantic_cache_hit_above_threshold(mock_qdrant):
    point = MagicMock()
    point.payload = {
        "answer": "cached semantic answer",
        "cache_schema_version": "3",
        "policy_version": "1",
        "corpus_version": "1",
    }
    mock_qdrant.query_points.return_value.points = [point]
    assert semantic_cache_get([0.1] * 768) == "cached semantic answer"
    query = mock_qdrant.query_points.call_args.kwargs
    assert query["query_filter"].must


@patch("src.retrieval.cache.qdrant_client")
def test_semantic_cache_miss_below_threshold(mock_qdrant):
    mock_qdrant.query_points.return_value.points = []
    assert semantic_cache_get([0.1] * 768) is None


@patch("src.retrieval.cache.qdrant_client")
def test_semantic_cache_set_stores_cache_version_metadata(mock_qdrant):
    semantic_cache_set("canonical question", [0.2] * 768, "an answer")
    payload = mock_qdrant.upsert.call_args.kwargs["points"][0].payload
    assert payload == {
        "question": "canonical question",
        "answer": "an answer",
        "cache_schema_version": cache_module.settings.cache_schema_version,
        "policy_version": cache_module.settings.cache_policy_version,
        "corpus_version": "1",
    }


@patch("src.retrieval.cache.qdrant_client")
def test_ensure_semantic_cache_indexes_creates_index_per_version_field(mock_qdrant):
    ensure_semantic_cache_indexes()
    fields = {call.kwargs["field_name"] for call in mock_qdrant.create_payload_index.call_args_list}
    assert fields == {"cache_schema_version", "policy_version", "corpus_version"}


@patch("src.retrieval.cache.qdrant_client")
def test_ensure_semantic_cache_indexes_is_idempotent_per_process(mock_qdrant):
    ensure_semantic_cache_indexes()
    ensure_semantic_cache_indexes()
    assert mock_qdrant.create_payload_index.call_count == 3


@patch("src.retrieval.cache.qdrant_client")
def test_ensure_semantic_cache_indexes_swallows_already_exists_error(mock_qdrant):
    mock_qdrant.create_payload_index.side_effect = Exception("already exists")
    ensure_semantic_cache_indexes()


@patch("src.retrieval.cache.qdrant_client")
def test_semantic_cache_get_ensures_indexes_before_querying(mock_qdrant):
    mock_qdrant.query_points.return_value.points = []
    semantic_cache_get([0.1] * 768)
    assert mock_qdrant.create_payload_index.call_count == 3


@patch("src.retrieval.cache.qdrant_client")
def test_semantic_cache_get_treats_qdrant_failure_as_a_miss(mock_qdrant):
    mock_qdrant.query_points.side_effect = Exception("read timeout")
    assert semantic_cache_get([0.1] * 768) is None


def test_current_corpus_version_falls_back_to_settings_when_marker_absent():
    assert cache_module._current_corpus_version() == cache_module.settings.corpus_version


def test_current_corpus_version_prefers_marker_file_when_present():
    cache_module._CORPUS_VERSION_MARKER.write_text("abc123fingerprint")
    assert cache_module._current_corpus_version() == "abc123fingerprint"


def test_exact_key_changes_when_corpus_fingerprint_changes():
    exact_cache_set("question", "answer from old corpus")
    cache_module._CORPUS_VERSION_MARKER.write_text("new-fingerprint")
    assert exact_cache_get("question") is None


@patch("src.retrieval.cache.qdrant_client")
def test_semantic_cache_set_uses_current_corpus_fingerprint(mock_qdrant):
    cache_module._CORPUS_VERSION_MARKER.write_text("fingerprint-xyz")
    semantic_cache_set("canonical question", [0.2] * 768, "an answer")
    payload = mock_qdrant.upsert.call_args.kwargs["points"][0].payload
    assert payload["corpus_version"] == "fingerprint-xyz"


@patch("src.retrieval.cache.qdrant_client")
def test_ensure_semantic_cache_indexes_does_not_latch_on_real_failure(mock_qdrant):
    mock_qdrant.create_payload_index.side_effect = Exception("collection not found")
    ensure_semantic_cache_indexes()
    assert cache_module._indexes_ensured is False
    assert cache_module._next_index_attempt > 0

    mock_qdrant.create_payload_index.side_effect = None
    cache_module._next_index_attempt = 0.0
    ensure_semantic_cache_indexes()
    assert cache_module._indexes_ensured is True
    assert mock_qdrant.create_payload_index.call_count == 4


@patch("src.retrieval.cache.qdrant_client")
def test_ensure_semantic_cache_indexes_backs_off_after_a_failure(mock_qdrant):
    mock_qdrant.create_payload_index.side_effect = Exception("collection not found")
    ensure_semantic_cache_indexes()
    calls_after_first = mock_qdrant.create_payload_index.call_count
    ensure_semantic_cache_indexes()
    ensure_semantic_cache_indexes()
    assert mock_qdrant.create_payload_index.call_count == calls_after_first


def test_exact_cache_clear_returns_removed_count():
    exact_cache_set("what is a pod", "a")
    exact_cache_set("what is a node", "b")
    assert exact_cache_clear() == 2
    assert exact_cache_count() == 0


def test_exact_cache_count_reflects_diskcache_size():
    assert exact_cache_count() == 0
    exact_cache_set("what is a pod", "a pod is...")
    exact_cache_set("what is a service", "a service is...")
    assert exact_cache_count() == 2


def test_exact_cache_lookup_failure_is_a_miss():
    with patch.object(cache_module._exact_cache, "get", side_effect=OSError("disk error")):
        assert exact_cache_get("what is a pod") is None
