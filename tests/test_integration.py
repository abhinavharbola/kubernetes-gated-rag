import hashlib
import sys

import pytest
from qdrant_client import QdrantClient
from unittest.mock import patch

import ingest as ingest_module
import src.graph as graph
import src.retrieval.cache as cache_module
import src.retrieval.search as search_module
from src.providers.llm import CompletionResult
from src.retrieval.cache import exact_cache_clear

DIM = 768


def fake_vector(text: str) -> list[float]:
    key = "pod-topic" if "pod" in text.lower() else text
    digest = hashlib.sha256(key.encode()).digest()
    vector = [(digest[i % 32] / 255.0) - 0.5 for i in range(DIM)]
    norm = sum(x * x for x in vector) ** 0.5
    return [x / norm for x in vector]


def fake_embed(texts, task_type, interactive=False):
    return [fake_vector(text) for text in texts]


@pytest.fixture
def memory_qdrant(tmp_path, monkeypatch):
    client = QdrantClient(":memory:")
    marker = tmp_path / "corpus_version"
    for module in (ingest_module, cache_module, search_module):
        monkeypatch.setattr(module, "qdrant_client", client)
    monkeypatch.setattr(ingest_module, "CORPUS_VERSION_MARKER", marker)
    monkeypatch.setattr(cache_module, "_CORPUS_VERSION_MARKER", marker)
    monkeypatch.setattr(ingest_module, "INGEST_EMBED_DELAY_SECONDS", 0.0)
    monkeypatch.setattr(ingest_module, "embed_texts", fake_embed)
    monkeypatch.setattr(ingest_module, "is_relevant", lambda text, fail_open=True: "sorting" not in text.lower())
    monkeypatch.setattr(search_module, "embed_query", fake_vector)
    monkeypatch.setattr(cache_module, "embed_for_cache", fake_vector)
    monkeypatch.setattr(cache_module, "_indexes_ensured", False)
    monkeypatch.setattr(cache_module, "_next_index_attempt", 0.0)
    exact_cache_clear()
    yield client, marker
    exact_cache_clear()


@pytest.fixture
def data_dir(tmp_path):
    root = tmp_path / "data"
    (root / "true_data").mkdir(parents=True)
    (root / "noisy_data").mkdir()
    (root / "true_data" / "pods.md").write_text(
        "# Pods\n\nA Pod is the smallest deployable unit.\n\n```yaml\napiVersion: v1\nkind: Pod\n"
        "metadata:\n  name: web\nspec:\n  containers:\n  - name: web\n    image: nginx\n```\n"
    )
    (root / "true_data" / "svc.yaml").write_text(
        "# a comment\napiVersion: v1\nkind: Service\nmetadata:\n  name: svc\nspec:\n  ports:\n  - port: 80\n"
    )
    (root / "noisy_data" / "sort.md").write_text("# Quicksort\nA sorting algorithm.\n")
    return root


def run_ingest(root, *extra):
    with patch.object(sys, "argv", ["ingest.py", str(root), *extra]):
        ingest_module.main()


def docs_count(client):
    return client.count(collection_name="kubernetes_docs", exact=True).count


def all_points(client):
    points, _ = client.scroll("kubernetes_docs", limit=500, with_payload=True)
    return points


def test_reingest_is_idempotent_and_fingerprint_is_stable(memory_qdrant, data_dir):
    client, marker = memory_qdrant
    run_ingest(data_dir)
    first_count, first_fingerprint = docs_count(client), marker.read_text()
    run_ingest(data_dir)
    assert docs_count(client) == first_count
    assert marker.read_text() == first_fingerprint


def test_noisy_data_is_rejected_and_paths_are_relative(memory_qdrant, data_dir):
    client, _ = memory_qdrant
    run_ingest(data_dir)
    sources = {p.payload["metadata"]["source_path"] for p in all_points(client)}
    assert sources == {"true_data/pods.md", "true_data/svc.yaml"}


def test_editing_a_file_replaces_its_chunks_and_changes_the_fingerprint(memory_qdrant, data_dir):
    client, marker = memory_qdrant
    run_ingest(data_dir)
    before = marker.read_text()
    assert any(p.payload["metadata"].get("manifest_kind") == "Pod" for p in all_points(client))
    (data_dir / "true_data" / "pods.md").write_text("# Pods\n\nShort now.\n")
    run_ingest(data_dir)
    assert not any(p.payload["metadata"].get("manifest_kind") == "Pod" for p in all_points(client))
    assert marker.read_text() != before
    assert marker.read_text() == ingest_module.compute_corpus_fingerprint()


def test_wipe_rebuilds_the_collections(memory_qdrant, data_dir):
    client, _ = memory_qdrant
    run_ingest(data_dir)
    (data_dir / "true_data" / "svc.yaml").unlink()
    run_ingest(data_dir)
    assert any(p.payload["metadata"]["source_path"] == "true_data/svc.yaml" for p in all_points(client))
    run_ingest(data_dir, "--wipe")
    assert {p.payload["metadata"]["source_path"] for p in all_points(client)} == {"true_data/pods.md"}


def test_empty_corpus_exits_non_zero_and_writes_no_fingerprint(memory_qdrant, tmp_path):
    _, marker = memory_qdrant
    root = tmp_path / "empty"
    (root / "true_data").mkdir(parents=True)
    with pytest.raises(SystemExit) as exit_info:
        run_ingest(root)
    assert exit_info.value.code == 1
    assert not marker.exists()


def test_retrieve_returns_ingested_chunks(memory_qdrant, data_dir):
    run_ingest(data_dir)
    results = search_module.retrieve("what is a Pod")
    assert results
    assert all({"text", "metadata", "retrieval_score"} <= set(r) for r in results)


def test_semantic_cache_round_trip_and_fingerprint_invalidation(memory_qdrant):
    _, marker = memory_qdrant
    ingest_module.ensure_collection()
    vector = fake_vector("what is a pod")
    cache_module.semantic_cache_set("what is a pod", vector, "A Pod is a unit.")
    assert cache_module.semantic_cache_get(vector) == "A Pod is a unit."
    assert cache_module.semantic_cache_get(fake_vector("unrelated question")) is None
    marker.write_text("a-different-corpus")
    assert cache_module.semantic_cache_get(vector) is None


@pytest.fixture
def stubbed_pipeline(memory_qdrant, data_dir):
    run_ingest(data_dir)
    llm_calls = []

    def fake_generate(messages, **kwargs):
        llm_calls.append(messages)
        return CompletionResult(content="A Pod is the smallest deployable unit.", provider="groq", model="m")

    def fake_rerank(question, candidates):
        return [{**c, "rerank_score": 0.9} for c in candidates[:3]]

    patches = [
        patch("src.graph.safety_gate", return_value=(True, None)),
        patch("src.graph.topic_gate", return_value=(True, None)),
        patch("src.graph.response_safety_gate", return_value=(True, None)),
        patch("src.graph.generate_main", side_effect=fake_generate),
        patch("src.graph.rerank_and_gate", side_effect=fake_rerank),
        patch("src.graph._submit_cache_write", side_effect=lambda fn, *a, **k: fn(*a, **k)),
    ]
    for p in patches:
        p.start()
    yield llm_calls
    for p in patches:
        p.stop()


def test_full_pipeline_answers_then_serves_exact_then_semantic_cache(stubbed_pipeline):
    llm_calls = stubbed_pipeline

    first = graph.run_turn("what is a Pod", [])
    assert first["answer"].startswith("A Pod")
    assert first.get("cache_layer") is None
    assert first["reranked"]
    assert len(llm_calls) == 1

    exact = graph.run_turn("what is a Pod?", [])
    assert exact["cache_layer"] == "exact"
    assert len(llm_calls) == 1

    semantic = graph.run_turn("explain the pod concept please", [])
    assert semantic["cache_layer"] == "semantic"
    assert semantic["answer"] == first["answer"]
    assert len(llm_calls) == 1


def test_full_pipeline_with_history_rewrites_then_caches_under_the_standalone_question(stubbed_pipeline):
    history = [{"role": "user", "content": "what is a Deployment?"}, {"role": "assistant", "content": "..."}]
    with patch("src.graph.generate_planner", return_value=CompletionResult("how do I scale a Deployment?", "nim", "m")):
        result = graph.run_turn("how do I scale it?", history)
    assert result["standalone_question"] == "how do I scale a Deployment?"
    assert result.get("rewrite_degraded") is None
    assert graph.run_turn("how do I scale a Deployment?", [])["cache_layer"] == "exact"


def test_full_pipeline_degraded_rewrite_leaves_no_cache_entries(stubbed_pipeline):
    history = [{"role": "user", "content": "what is a Deployment?"}, {"role": "assistant", "content": "..."}]
    with patch("src.graph.generate_planner", side_effect=RuntimeError("planner down")):
        result = graph.run_turn("how do I scale it?", history)
    assert result["rewrite_degraded"] is True
    assert result["answer"].startswith("A Pod")
    assert graph.run_turn("how do I scale it?", []).get("cache_layer") is None
