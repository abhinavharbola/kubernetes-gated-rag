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
    "thanks) as on-topic in addition to that. Respond with exactly one word and nothing "
    "else, no explanation, no punctuation: on-topic or off-topic."
)

SAFETY_SYSTEM_PROMPT = (
    "Classify whether the user message contains clearly unsafe content: explicit "
    "violence, illegal activity, harassment, hate speech, sexual content, or self-harm. "
    "This is independent of topic, a message can be unsafe regardless of whether it "
    "mentions Kubernetes at all. Ordinary questions are safe even if they are off-topic, "
    "mundane, or unrelated to Kubernetes, being off-topic is not itself a safety issue. "
    "When in doubt, classify as safe, only flag content that is unambiguously harmful. "
    "Respond with exactly one word and nothing else, no explanation, no punctuation: "
    "safe or unsafe."
)

_rails: LLMRails | None = None


def _parse_binary_verdict(raw: str, true_word: str, false_word: str) -> bool | None:
    # checks only the FIRST token of the response, not "does this word
    # appear anywhere" — a substring-anywhere check misfires if the model
    # adds any preamble at all, e.g. "this is safe, not unsafe" contains
    # "unsafe" as a literal substring despite the actual verdict being
    # safe. We asked for exactly one word; treat only that first word as
    # the verdict, ignore whatever the model says around it.
    stripped = raw.strip()
    if not stripped:
        return None
    first_token = stripped.split()[0].strip(".,!?\"'").lower()
    if first_token == false_word:
        return False
    if first_token == true_word:
        return True
    return None


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
    # two independent checks, either firing is enough to block. Colang's
    # few-shot flow catches jailbreak-pattern attempts specifically (its
    # strength: jailbreaks share recognizable phrasing regardless of topic).
    # It does NOT catch general unsafe content that isn't phrased like a
    # jailbreak — violence, harassment, and similar categories were passing
    # this gate entirely and only getting caught downstream by topic_gate
    # rejecting them as off-topic, for the wrong reason, and not at all if
    # the content happened to be phrased in an on-topic-sounding way. The
    # direct classifier below closes that gap the same way it closed
    # check_topic's few-shot generalization gap.
    rails = _get_rails()
    result = rails.generate(messages=[{"role": "user", "content": raw_message}])
    content = result.get("content", "") if isinstance(result, dict) else str(result)
    if any(indicator in content for indicator in JAILBREAK_INDICATORS):
        return False

    classifier_result = generate_planner(
        [
            {"role": "system", "content": SAFETY_SYSTEM_PROMPT},
            {"role": "user", "content": raw_message},
        ]
    )
    verdict = _parse_binary_verdict(classifier_result.content, true_word="safe", false_word="unsafe")
    if verdict is None:
        logger.warning("safety classifier gave unparseable verdict %r, failing closed", classifier_result.content)
        return False
    return verdict


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
    verdict = _parse_binary_verdict(result.content, true_word="on-topic", false_word="off-topic")
    if verdict is None:
        logger.warning("topic gate got unparseable verdict %r, failing closed", result.content)
        return False
    return verdict


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
