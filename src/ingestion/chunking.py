import re

HEADER_LINE_RE = re.compile(r"^(#{1,6})\s+(.*)$")
MANIFEST_START_RE = re.compile(r"^apiVersion:\s*\S")
SEPARATOR_RE = re.compile(r"^---\s*$")
FENCE_RE = re.compile(r"^\s*(?:```|~~~)")
YAML_CONTINUATION_RE = re.compile(r"^(?:[A-Za-z_][\w.\-/]*\s*:|#|-\s|-$)")
WORD_SPAN_RE = re.compile(r"\S+")

FALLBACK_WINDOW_WORDS = 300
FALLBACK_OVERLAP_WORDS = 50
MANIFEST_MAX_WORDS = FALLBACK_WINDOW_WORDS


def split_by_markdown_headers(text: str) -> list[dict]:
    sections = []
    header = None
    buffer: list[str] = []
    fence_marker = None

    def flush() -> None:
        if buffer:
            sections.append({"header": header, "text": "".join(buffer)})

    for line in text.splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            marker = stripped[:3]
            if fence_marker is None:
                fence_marker = marker
            elif marker == fence_marker:
                fence_marker = None
        match = None if fence_marker is not None else HEADER_LINE_RE.match(line.rstrip("\r\n"))
        if match:
            flush()
            header = match.group(2).strip()
            buffer = [line]
        else:
            buffer.append(line)
    flush()

    if not sections:
        return [{"header": None, "text": text}]
    return sections


def _clean_scalar(value: str | None) -> str | None:
    if value is None:
        return None
    return value.strip().strip("'\"")


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def _parse_manifest_kind_and_name(block_text: str) -> tuple[str | None, str | None]:
    kind_match = re.search(r"^kind:\s*(\S+)", block_text, re.MULTILINE)
    kind = _clean_scalar(kind_match.group(1)) if kind_match else None

    name = None
    metadata_match = re.search(r"^metadata:\s*$", block_text, re.MULTILINE)
    if metadata_match:
        rest = block_text[metadata_match.end() :]
        next_top_level = re.search(r"^\S", rest, re.MULTILINE)
        metadata_body = rest[: next_top_level.start()] if next_top_level else rest
        lines = [line for line in metadata_body.splitlines() if line.strip()]
        if lines:
            base_indent = _indent(lines[0])
            for line in lines:
                if _indent(line) != base_indent:
                    continue
                name_match = re.match(r"name:\s*(\S+)", line.strip())
                if name_match:
                    name = _clean_scalar(name_match.group(1))
                    break

    return kind, name


def _fallback_window(text: str) -> list[dict]:
    spans = [match.span() for match in WORD_SPAN_RE.finditer(text)]
    if not spans:
        return []
    if len(spans) <= FALLBACK_WINDOW_WORDS:
        return [{"text": text.strip(), "kind": None, "name": None}]

    chunks = []
    step = FALLBACK_WINDOW_WORDS - FALLBACK_OVERLAP_WORDS
    for start in range(0, len(spans), step):
        window = spans[start : start + FALLBACK_WINDOW_WORDS]
        if window:
            chunks.append({"text": text[window[0][0] : window[-1][1]], "kind": None, "name": None})
        if start + FALLBACK_WINDOW_WORDS >= len(spans):
            break
    return chunks


def split_by_manifest_blocks(section_text: str) -> list[dict]:
    blocks: list[dict] = []
    prose: list[str] = []
    manifest: list[str] | None = None

    def flush_prose() -> None:
        if prose:
            blocks.extend(_fallback_window("\n".join(prose)))
            prose.clear()

    def flush_manifest() -> None:
        nonlocal manifest
        if manifest is None:
            return
        block_text = "\n".join(manifest).strip()
        manifest = None
        if not block_text:
            return
        kind, name = _parse_manifest_kind_and_name(block_text)
        if len(block_text.split()) > MANIFEST_MAX_WORDS:
            for sub_chunk in _fallback_window(block_text):
                blocks.append({"text": sub_chunk["text"], "kind": kind, "name": name})
        else:
            blocks.append({"text": block_text, "kind": kind, "name": name})

    for line in section_text.splitlines():
        if manifest is not None:
            if SEPARATOR_RE.match(line) or FENCE_RE.match(line):
                flush_manifest()
                continue
            if MANIFEST_START_RE.match(line):
                flush_manifest()
                manifest = [line]
                continue
            if line.strip() and not line[0].isspace() and not YAML_CONTINUATION_RE.match(line):
                flush_manifest()
                prose.append(line)
                continue
            manifest.append(line)
            continue
        if SEPARATOR_RE.match(line) or FENCE_RE.match(line):
            continue
        if MANIFEST_START_RE.match(line):
            flush_prose()
            manifest = [line]
            continue
        prose.append(line)

    flush_manifest()
    flush_prose()
    return blocks


def chunk_document(text: str, base_metadata: dict, markdown: bool = True) -> list[dict]:
    sections = split_by_markdown_headers(text) if markdown else [{"header": None, "text": text}]
    chunks = []
    for section in sections:
        for block in split_by_manifest_blocks(section["text"]):
            if not block["text"].strip():
                continue
            metadata = {
                **base_metadata,
                "section_header": section["header"],
                "manifest_kind": block["kind"],
                "manifest_name": block["name"],
            }
            chunks.append({"text": block["text"], "metadata": metadata})
    return chunks
