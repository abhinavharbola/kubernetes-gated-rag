import logging

from langchain_groq import ChatGroq
from nemoguardrails import LLMRails, RailsConfig

from src.colang_rules import COLANG_CONTENT, JAILBREAK_INDICATORS, YAML_CONTENT
from src.config import settings
from src.llm import generate_planner

logger = logging.getLogger(__name__)

OFF_TOPIC_REFUSAL = (
    "I'm built to help with Kubernetes questions specifically. "
    "Ask me about Pods, Deployments, Services, or manifest syntax and I'll do my best."
)
UNSAFE_REFUSAL = (
    "I can't help with that request. I'm here to answer Kubernetes and container-"
    "orchestration questions, happy to help if you'd like to ask one."
)

TOPIC_SYSTEM_PROMPT = (
    "You are a topic-control classifier for a Kubernetes documentation assistant. Do "
    "not classify a message as on-topic unless it concerns Kubernetes objects, manifests, "
    "controllers (Deployments, StatefulSets, Services, etc.), cluster operations, or "
    "container orchestration workflows. General programming requests (write code, solve "
    "an algorithm, explain a language feature) that are not specifically about Kubernetes "
    "are off-topic, even though they're technical. Only classify small talk (greetings, "
    "thanks) as on-topic in addition to that. Respond with exactly one word: 'on-topic' "
    "or 'off-topic'."
)

_rails: LLMRails | None = None


def _get_rails() -> LLMRails:
    # lazily built once per process rather than at import time: Colang
    # parsing has real startup cost, worth paying once, not per request, but
    # also not worth paying at all for code paths that never call the gate.
    global _rails
    if _rails is None:
        guard_llm = ChatGroq(api_key=settings.groq_api_key, model=settings.groq_planner_model, temperature=0)
        config = RailsConfig.from_content(colang_content=COLANG_CONTENT, yaml_content=YAML_CONTENT)
        _rails = LLMRails(config, llm=guard_llm)
        logger.info("guardrails: NeMo Colang rails initialized")
    return _rails


def check_safety(raw_message: str) -> bool:
    # Colang's few-shot flow matching is a good fit for jailbreak detection
    # specifically: jailbreak attempts share distinctive, recognizable
    # phrasing regardless of surrounding topic, which is exactly what
    # few-shot similarity matching is good at.
    rails = _get_rails()
    result = rails.generate(messages=[{"role": "user", "content": raw_message}])
    content = result.get("content", "") if isinstance(result, dict) else str(result)
    return not any(indicator in content for indicator in JAILBREAK_INDICATORS)


def check_topic(standalone_question: str) -> bool:
    # deliberately NOT the Colang few-shot flow: "is this on-topic for
    # Kubernetes" is an open-ended classification over an unbounded space of
    # possible off-topic requests, not a small set of recognizable patterns.
    # Few-shot matching against a fixed example list generalizes poorly to
    # categories the examples don't resemble — e.g. "give me python code for
    # two sum" didn't match any "off topic" example closely enough, fell
    # through Colang's general-response flow, and got answered directly
    # instead of refused. A direct classifier judges the actual question
    # asked, not its distance from a handful of memorized examples.
    result = generate_planner(
        [
            {"role": "system", "content": TOPIC_SYSTEM_PROMPT},
            {"role": "user", "content": standalone_question},
        ]
    )
    verdict = result.content.strip().lower()
    if "on-topic" in verdict:
        return True
    if "off-topic" in verdict:
        return False
    logger.warning("topic gate got unparseable verdict %r, failing closed", verdict)
    return False


def safety_gate(raw_message: str) -> tuple[bool, str | None]:
    """Runs on the raw, unmodified user message, before any other pipeline step
    (including the history-rewrite planner call), so a jailbreak attempt is
    rejected before it costs a planner call. Fails closed on any error."""
    try:
        if not check_safety(raw_message):
            return False, UNSAFE_REFUSAL
        return True, None
    except Exception as error:
        logger.error("safety gate failed, failing closed: %s", error)
        return False, UNSAFE_REFUSAL


def topic_gate(standalone_question: str) -> tuple[bool, str | None]:
    """Runs on the history-rewritten standalone question (after safety_gate
    and rewrite_with_history), so context-dependent follow-ups aren't misjudged as
    off-topic. Fails closed on any error."""
    try:
        if not check_topic(standalone_question):
            return False, OFF_TOPIC_REFUSAL
        return True, None
    except Exception as error:
        logger.error("topic gate failed, failing closed: %s", error)
        return False, OFF_TOPIC_REFUSAL
