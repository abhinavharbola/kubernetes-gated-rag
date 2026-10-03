import sys
from unittest.mock import MagicMock, patch

import pytest

import ingest as ingest_module
from ingest import (
    _point_id,
    _write_corpus_version,
    compute_corpus_fingerprint,
    ensure_collection,
    ingest_directory,
)
from src.config import settings


@pytest.fixture(autouse=True)
def isolate_ingest(monkeypatch):
    monkeypatch.setattr(ingest_module, "INGEST_EMBED_DELAY_SECONDS", 0.0)
    with patch("ingest.ensure_semantic_cache_indexes"):
        yield


def _mock_collections(*names: str) -> MagicMock:
    response = MagicMock()
    response.collections = [MagicMock(name=n) for n in names]
    for mock_collection, n in zip(response.collections, names):
        mock_collection.name = n
    return response


@patch("ingest.qdrant_client")
def test_ensure_collection_creates_both_collections_when_missing(mock_qdrant):
    mock_qdrant.get_collections.return_value = _mock_collections()
    ensure_collection(wipe=False)
    created = {call.kwargs["collection_name"] for call in mock_qdrant.create_collection.call_args_list}
    assert created == {settings.qdrant_docs_collection, settings.qdrant_cache_collection}
    assert mock_qdrant.delete_collection.called is False


@patch("ingest.qdrant_client")
def test_ensure_collection_does_not_touch_existing_collections_without_wipe(mock_qdrant):
    mock_qdrant.get_collections.return_value = _mock_collections(
        settings.qdrant_docs_collection, settings.qdrant_cache_collection
    )
    ensure_collection(wipe=False)
    assert mock_qdrant.delete_collection.called is False
    assert mock_qdrant.create_collection.called is False


@patch("ingest._wipe_exact_cache")
@patch("ingest.qdrant_client")
def test_wipe_deletes_both_docs_and_semantic_cache_collections(mock_qdrant, mock_wipe_exact_cache):
    mock_qdrant.get_collections.return_value = _mock_collections(
        settings.qdrant_docs_collection, settings.qdrant_cache_collection
    )
    ensure_collection(wipe=True)
    deleted = {call.args[0] for call in mock_qdrant.delete_collection.call_args_list}
    assert deleted == {settings.qdrant_docs_collection, settings.qdrant_cache_collection}
    created = {call.kwargs["collection_name"] for call in mock_qdrant.create_collection.call_args_list}
    assert created == {settings.qdrant_docs_collection, settings.qdrant_cache_collection}


@patch("ingest._wipe_exact_cache")
@patch("ingest.qdrant_client")
def test_wipe_also_clears_the_local_exact_cache(mock_qdrant, mock_wipe_exact_cache):
    mock_qdrant.get_collections.return_value = _mock_collections()
    ensure_collection(wipe=True)
    assert mock_wipe_exact_cache.called is True


@patch("ingest.qdrant_client")
def test_wipe_is_a_noop_for_the_exact_cache_when_wipe_is_false(mock_qdrant):
    mock_qdrant.get_collections.return_value = _mock_collections(
        settings.qdrant_docs_collection, settings.qdrant_cache_collection
    )
    with patch("ingest._wipe_exact_cache") as mock_wipe_exact_cache:
        ensure_collection(wipe=False)
        assert mock_wipe_exact_cache.called is False


def test_write_corpus_version_writes_the_marker_file(tmp_path):
    marker = tmp_path / "nested" / "corpus_version"
    with patch("ingest.CORPUS_VERSION_MARKER", marker):
        _write_corpus_version("abc123")
    assert marker.read_text() == "abc123"



def test_ensure_collection_creates_the_source_path_index():
    with patch("ingest.qdrant_client") as mock_qdrant:
        mock_qdrant.get_collections.return_value = _mock_collections()
        ensure_collection(wipe=False)
    fields = {call.kwargs["field_name"] for call in mock_qdrant.create_payload_index.call_args_list}
    assert fields == {"metadata.source_path"}


def test_ensure_collection_tolerates_an_existing_source_path_index():
    with patch("ingest.qdrant_client") as mock_qdrant:
        mock_qdrant.get_collections.return_value = _mock_collections()
        mock_qdrant.create_payload_index.side_effect = Exception("index already exists")
        ensure_collection(wipe=False)


def test_point_ids_are_deterministic_per_source_and_index():
    assert _point_id("true_data/a.md", 0) == _point_id("true_data/a.md", 0)
    assert _point_id("true_data/a.md", 0) != _point_id("true_data/a.md", 1)
    assert _point_id("true_data/a.md", 0) != _point_id("true_data/b.md", 0)


def _run_ingest(tmp_path, chunks, relevant=True, root=None, **kwargs):
    with (
        patch("ingest._upsert_batch") as upsert,
        patch("ingest._delete_source_points") as delete,
        patch("ingest.embed_texts", side_effect=lambda texts, task_type: [[0.1] * 4 for _ in texts]),
        patch("ingest.is_relevant", return_value=relevant) as is_relevant,
        patch("ingest.chunk_document", return_value=chunks) as chunk_document,
        patch("ingest.parse_file", return_value="content"),
    ):
        result = ingest_directory(tmp_path, root=root, **kwargs)
    return result, upsert, delete, is_relevant, chunk_document


def test_reingesting_a_file_replaces_its_points_with_stable_ids(tmp_path):
    (tmp_path / "a.md").write_text("x")
    chunks = [{"text": "one", "metadata": {}}, {"text": "two", "metadata": {}}]
    _, first_upsert, first_delete, _, _ = _run_ingest(tmp_path, chunks)
    _, second_upsert, _, _, _ = _run_ingest(tmp_path, chunks)
    first_ids = [p.id for p in first_upsert.call_args.args[0]]
    second_ids = [p.id for p in second_upsert.call_args.args[0]]
    assert first_ids == second_ids
    assert len(set(first_ids)) == 2
    first_delete.assert_called_once_with("a.md")


def test_source_path_is_relative_to_the_root_and_posix(tmp_path):
    nested = tmp_path / "true_data" / "sub"
    nested.mkdir(parents=True)
    (nested / "a.md").write_text("x")
    _, upsert, delete, _, chunk_document = _run_ingest(
        tmp_path / "true_data", [{"text": "one", "metadata": {}}], root=tmp_path
    )
    delete.assert_called_once_with("true_data/sub/a.md")
    assert chunk_document.call_args.kwargs["base_metadata"] == {"source_path": "true_data/sub/a.md"}


def test_irrelevant_file_removes_stale_points_and_is_not_embedded(tmp_path):
    (tmp_path / "a.md").write_text("x")
    result, upsert, delete, _, _ = _run_ingest(tmp_path, [{"text": "one", "metadata": {}}], relevant=False)
    assert result["skipped_irrelevant"] == 1
    assert result["ingested"] == 0
    upsert.assert_not_called()
    delete.assert_called_once_with("a.md")


def test_relevance_fail_open_flag_is_forwarded(tmp_path):
    (tmp_path / "a.md").write_text("x")
    _, _, _, is_relevant, _ = _run_ingest(
        tmp_path, [{"text": "one", "metadata": {}}], relevance_fail_open=False
    )
    assert is_relevant.call_args.kwargs["fail_open"] is False


def test_yaml_files_skip_markdown_header_splitting(tmp_path):
    (tmp_path / "a.yaml").write_text("x")
    (tmp_path / "b.md").write_text("x")
    _, _, _, _, chunk_document = _run_ingest(tmp_path, [{"text": "one", "metadata": {}}])
    flags = {call.args[0] if call.args else None: call.kwargs["markdown"] for call in chunk_document.call_args_list}
    assert sorted(call.kwargs["markdown"] for call in chunk_document.call_args_list) == [False, True]
    assert flags


def _scroll_pages(*pages):
    responses = []
    for index, page in enumerate(pages):
        points = []
        for source, text in page:
            point = MagicMock()
            point.payload = {"text": text, "metadata": {"source_path": source}}
            points.append(point)
        next_offset = f"page-{index + 1}" if index + 1 < len(pages) else None
        responses.append((points, next_offset))
    return responses


def test_fingerprint_reflects_the_corpus_not_the_run_or_point_order():
    with patch("ingest.qdrant_client") as mock_qdrant:
        mock_qdrant.scroll.side_effect = _scroll_pages([("a.md", "one"), ("b.md", "two")])
        first = compute_corpus_fingerprint()
    with patch("ingest.qdrant_client") as mock_qdrant:
        mock_qdrant.scroll.side_effect = _scroll_pages([("b.md", "two")], [("a.md", "one")])
        second = compute_corpus_fingerprint()
    assert first == second


def test_fingerprint_changes_when_content_changes():
    with patch("ingest.qdrant_client") as mock_qdrant:
        mock_qdrant.scroll.side_effect = _scroll_pages([("a.md", "one")])
        before = compute_corpus_fingerprint()
    with patch("ingest.qdrant_client") as mock_qdrant:
        mock_qdrant.scroll.side_effect = _scroll_pages([("a.md", "one edited")])
        after = compute_corpus_fingerprint()
    assert before != after


def test_main_refuses_to_write_a_fingerprint_for_an_empty_corpus(tmp_path):
    (tmp_path / "true_data").mkdir()
    marker = tmp_path / "marker"
    with (
        patch.object(sys, "argv", ["ingest.py", str(tmp_path)]),
        patch("ingest.ensure_collection"),
        patch("ingest.ingest_directory", return_value={"ingested": 0, "skipped_irrelevant": 0, "failed": 0}),
        patch("ingest.qdrant_client") as mock_qdrant,
        patch("ingest.CORPUS_VERSION_MARKER", marker),
    ):
        mock_qdrant.count.return_value.count = 0
        with pytest.raises(SystemExit) as exit_info:
            ingest_module.main()
    assert exit_info.value.code == 1
    assert not marker.exists()


def test_main_writes_the_fingerprint_and_ingests_noisy_data_fail_closed(tmp_path):
    (tmp_path / "true_data").mkdir()
    (tmp_path / "noisy_data").mkdir()
    marker = tmp_path / "marker"
    calls = []

    def fake_ingest(directory, root=None, relevance_fail_open=True):
        calls.append((directory.name, relevance_fail_open))
        return {"ingested": 1, "skipped_irrelevant": 0, "failed": 0}

    with (
        patch.object(sys, "argv", ["ingest.py", str(tmp_path)]),
        patch("ingest.ensure_collection"),
        patch("ingest.ingest_directory", side_effect=fake_ingest),
        patch("ingest.qdrant_client") as mock_qdrant,
        patch("ingest.compute_corpus_fingerprint", return_value="fp123"),
        patch("ingest.CORPUS_VERSION_MARKER", marker),
    ):
        mock_qdrant.count.return_value.count = 5
        ingest_module.main()
    assert calls == [("true_data", True), ("noisy_data", False)]
    assert marker.read_text() == "fp123"
