from src.ingestion.chunking import (
    FALLBACK_WINDOW_WORDS,
    _fallback_window,
    _parse_manifest_kind_and_name,
    chunk_document,
    split_by_manifest_blocks,
    split_by_markdown_headers,
)


def test_split_by_markdown_headers_basic():
    text = "# Title\nintro text\n## Section A\nbody a\n## Section B\nbody b\n"
    sections = split_by_markdown_headers(text)
    assert [s["header"] for s in sections] == ["Title", "Section A", "Section B"]
    assert "body a" in sections[1]["text"]
    assert "body b" in sections[2]["text"]


def test_split_by_markdown_headers_no_headers_returns_single_section():
    text = "just plain text, no headers at all"
    sections = split_by_markdown_headers(text)
    assert len(sections) == 1
    assert sections[0]["header"] is None
    assert sections[0]["text"] == text


def test_split_by_markdown_headers_preserves_text_before_first_header():
    text = "preamble\n# First Header\nbody"
    sections = split_by_markdown_headers(text)
    assert sections[0]["header"] is None
    assert "preamble" in sections[0]["text"]
    assert sections[1]["header"] == "First Header"


def test_parse_manifest_kind_and_name_extracts_both():
    block = "apiVersion: v1\nkind: Pod\nmetadata:\n  name: web\nspec:\n  containers: []\n"
    kind, name = _parse_manifest_kind_and_name(block)
    assert kind == "Pod"
    assert name == "web"


def test_parse_manifest_kind_and_name_handles_quoted_scalars():
    block = 'apiVersion: v1\nkind: "Service"\nmetadata:\n  name: \'my-svc\'\n'
    kind, name = _parse_manifest_kind_and_name(block)
    assert kind == "Service"
    assert name == "my-svc"


def test_parse_manifest_kind_and_name_returns_none_when_absent():
    block = "just prose, no manifest fields here"
    kind, name = _parse_manifest_kind_and_name(block)
    assert kind is None
    assert name is None


def test_parse_manifest_kind_and_name_does_not_read_past_metadata_block():
    # name lives inside metadata:, a name: key appearing under a different
    # top-level section (e.g. spec:) must not be picked up
    block = (
        "apiVersion: v1\nkind: Pod\nmetadata:\n  name: real-name\n"
        "  labels:\n    app: demo\nspec:\n  containers:\n    - name: decoy-name\n"
    )
    kind, name = _parse_manifest_kind_and_name(block)
    assert kind == "Pod"
    assert name == "real-name"


def test_split_by_manifest_blocks_single_document():
    section = "apiVersion: v1\nkind: Pod\nmetadata:\n  name: web\nspec:\n  containers: []\n"
    blocks = split_by_manifest_blocks(section)
    assert len(blocks) == 1
    assert blocks[0]["kind"] == "Pod"
    assert blocks[0]["name"] == "web"
    assert blocks[0]["text"].startswith("apiVersion: v1")


def test_split_by_manifest_blocks_multi_document_file():
    section = (
        "apiVersion: v1\nkind: Pod\nmetadata:\n  name: web\nspec:\n  containers: []\n"
        "---\n"
        "apiVersion: v1\nkind: Service\nmetadata:\n  name: web-svc\nspec:\n  ports: []\n"
    )
    blocks = split_by_manifest_blocks(section)
    assert [b["kind"] for b in blocks] == ["Pod", "Service"]
    assert [b["name"] for b in blocks] == ["web", "web-svc"]
    # the separator line itself must not leak into either block's text
    assert "---" not in blocks[0]["text"]
    assert "---" not in blocks[1]["text"]


def test_split_by_manifest_blocks_no_trailing_separator():
    # a file with no final "---" after the last document is the common
    # case, not an edge case: the last block must still be captured whole.
    section = "apiVersion: v1\nkind: ConfigMap\nmetadata:\n  name: cfg\ndata:\n  key: value\n"
    blocks = split_by_manifest_blocks(section)
    assert len(blocks) == 1
    assert blocks[0]["kind"] == "ConfigMap"
    assert blocks[0]["text"].rstrip().endswith("value")


def test_split_by_manifest_blocks_prose_with_no_manifest_falls_back_to_window():
    section = "This is a prose explanation of Kubernetes concepts with no manifest in it at all."
    blocks = split_by_manifest_blocks(section)
    assert len(blocks) == 1
    assert blocks[0]["kind"] is None
    assert blocks[0]["name"] is None
    assert blocks[0]["text"] == section


def test_split_by_manifest_blocks_prose_preamble_before_first_manifest():
    section = (
        "Here is an example Pod manifest:\n\n"
        "apiVersion: v1\nkind: Pod\nmetadata:\n  name: web\nspec:\n  containers: []\n"
    )
    blocks = split_by_manifest_blocks(section)
    # the leading prose becomes its own fallback-window block, the manifest
    # becomes its own structured block, in that order
    assert len(blocks) == 2
    assert blocks[0]["kind"] is None
    assert "example Pod manifest" in blocks[0]["text"]
    assert blocks[1]["kind"] == "Pod"


def test_fallback_window_short_text_stays_one_chunk():
    text = "a short paragraph that is well under the window size"
    chunks = _fallback_window(text)
    assert len(chunks) == 1
    assert chunks[0]["text"] == text


def test_fallback_window_empty_text_returns_no_chunks():
    assert _fallback_window("   ") == []
    assert _fallback_window("") == []


def test_fallback_window_long_text_splits_with_overlap():
    words = [f"word{i}" for i in range(700)]
    text = " ".join(words)
    chunks = _fallback_window(text)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk["text"].split()) <= FALLBACK_WINDOW_WORDS
    first_words = chunks[0]["text"].split()
    second_words = chunks[1]["text"].split()
    assert set(first_words) & set(second_words)


def test_chunk_document_combines_headers_and_manifest_blocks_with_metadata():
    text = (
        "# Pod basics\n\n"
        "A Pod is the smallest deployable unit.\n\n"
        "apiVersion: v1\nkind: Pod\nmetadata:\n  name: web\nspec:\n  containers: []\n\n"
        "## Notes\n\nAdditional prose here about usage."
    )
    chunks = chunk_document(text, base_metadata={"source_path": "pod.md"})

    assert all(c["metadata"]["source_path"] == "pod.md" for c in chunks)

    manifest_chunks = [c for c in chunks if c["metadata"]["manifest_kind"] == "Pod"]
    assert len(manifest_chunks) == 1
    assert manifest_chunks[0]["metadata"]["manifest_name"] == "web"
    assert manifest_chunks[0]["metadata"]["section_header"] == "Pod basics"

    notes_chunks = [c for c in chunks if c["metadata"]["section_header"] == "Notes"]
    assert len(notes_chunks) == 1
    assert notes_chunks[0]["metadata"]["manifest_kind"] is None


def test_chunk_document_skips_blank_chunks():
    text = "# Header\n\n\n\n## Next\n\nreal content"
    chunks = chunk_document(text, base_metadata={"source_path": "x"})
    assert all(c["text"].strip() for c in chunks)
