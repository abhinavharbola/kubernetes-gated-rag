import logging

from src.llm import generate_planner

logger = logging.getLogger(__name__)

RELEVANCE_SYSTEM_PROMPT = (
    "You are a document relevance classifier for a Kubernetes documentation corpus. "
    "Given an excerpt from a document, classify whether its subject matter is about "
    "Kubernetes, container orchestration, or closely related cluster/infrastructure "
    "operations. Off-topic technical content (general CS theory, unrelated hardware, "
    "algorithms, compilers, data structures, etc.) is NOT relevant, even if it's "
    "technical and well-written. Respond with exactly one word: 'relevant' or "
    "'irrelevant'."
)

# enough for the classifier to judge subject matter (title, abstract, intro)
# without paying embedding-scale token cost on the full document
EXCERPT_CHARS = 2000


def is_relevant(document_text: str) -> bool:
    """Classifies a parsed document's excerpt for corpus relevance before it's
    chunked and embedded. Fails OPEN (treated as relevant) on any classifier
    error or unparseable response, deliberately the opposite of the
    query-time guardrails' fail-closed behavior: a missed rejection here just
    leaves one extra document that the rerank gate will very likely still
    filter out per-query, whereas a false rejection here silently shrinks a
    batch ingestion job's corpus with nobody watching to notice."""
    excerpt = document_text[:EXCERPT_CHARS]
    try:
        result = generate_planner(
            [
                {"role": "system", "content": RELEVANCE_SYSTEM_PROMPT},
                {"role": "user", "content": excerpt},
            ]
        )
        verdict = result.content.strip().lower()
    except Exception as error:
        logger.warning("relevance classifier failed, ingesting anyway: %s", error)
        return True

    # checked in this order deliberately: "irrelevant" contains "relevant"
    # as a substring, so checking for "relevant" first would misread every
    # "irrelevant" verdict as an affirmative match.
    if "irrelevant" in verdict:
        return False
    if "relevant" in verdict:
        return True

    logger.warning("relevance classifier gave unparseable verdict %r, ingesting anyway", verdict)
    return True
