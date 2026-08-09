import logging

from langchain_groq import ChatGroq
from nemoguardrails import LLMRails, RailsConfig

from src.colang_rules import COLANG_CONTENT, JAILBREAK_INDICATORS, OFF_TOPIC_INDICATORS, YAML_CONTENT
from src.config import settings

logger = logging.getLogger(__name__)

OFF_TOPIC_REFUSAL = (
    "I'm built to help with Kubernetes questions specifically. "
    "Ask me about Pods, Deployments, Services, or manifest syntax and I'll do my best."
)
UNSAFE_REFUSAL = (
    "I can't help with that request. I'm here to answer Kubernetes and container-"
    "orchestration questions, happy to help if you'd like to ask one."
)

_rails: LLMRails | None = None


def _get_rails() -> LLMRails:
    # lazily built once per process rather than at import time: Colang
    # parsing has real startup cost, worth paying once, not per request, but
    # also not worth paying at all for code paths (e.g. a plain unit test
    # importing this module) that never actually call the gate.
    global _rails
    if _rails is None:
        guard_llm = ChatGroq(api_key=settings.groq_api_key, model=settings.groq_planner_model, temperature=0)
        config = RailsConfig.from_content(colang_content=COLANG_CONTENT, yaml_content=YAML_CONTENT)
        _rails = LLMRails(config, llm=guard_llm)
        logger.info("guardrails: NeMo Colang rails initialized")
    return _rails


def _flow_fired(message: str, indicators: list[str]) -> bool:
    rails = _get_rails()
    result = rails.generate(messages=[{"role": "user", "content": message}])
    # LLMRails.generate returns {'role': 'assistant', 'content': '...'}
    content = result.get("content", "") if isinstance(result, dict) else str(result)
    return any(indicator in content for indicator in indicators)


def check_safety(raw_message: str) -> bool:
    # checks only for the jailbreak flow specifically. If a different flow
    # fires instead (off-topic, greeting, ...), that's not this gate's job:
    # topic_gate handles off-topic later, on the rewritten question, and
    # small talk is allowed through by design.
    return not _flow_fired(raw_message, JAILBREAK_INDICATORS)


def check_topic(standalone_question: str) -> bool:
    return not _flow_fired(standalone_question, OFF_TOPIC_INDICATORS)


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
