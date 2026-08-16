import json
import logging
from concurrent.futures import ThreadPoolExecutor

from langchain_groq import ChatGroq
from nemoguardrails import LLMRails, RailsConfig

from src.guardrails.colang_rules import COLANG_CONTENT, JAILBREAK_INDICATORS, YAML_CONTENT
from src.config import settings
from src.providers.clients import nim_client
from src.tracing import provider_call_span

logger = logging.getLogger(__name__)

OFF_TOPIC_REFUSAL = (
    "I'm built to help with Kubernetes questions specifically. "
    "Ask me about Pods, Deployments, Services, or manifest syntax and I'll do my best."
)
UNSAFE_REFUSAL = (
    "I can't help with that request. I'm here to answer Kubernetes and container-"
    "orchestration questions, happy to help if you'd like to ask one."
)

# --- NeMoGuard topic-control (nemoguard_topic_model, called directly
# against nim_client — this model only exists on NIM, so there's no
# cross-provider chain to fail over through; an error here is caught by
# topic_gate() below and fails closed like everything else). ---
#
# Prompt shape follows NVIDIA's documented format for this model: a system
# prompt stating the topical policy, an output-restriction instruction
# appended to it, and the message to classify as the final user turn.
# https://docs.nvidia.com/nim/llama-3-1-nemoguard-8b-topiccontrol/latest/getting-started.html
TOPIC_POLICY_PROMPT = (
    "You are the Kubernetes documentation assistant's topic gate. Only allow messages "
    "that concern Kubernetes objects, manifests, controllers (Deployments, StatefulSets, "
    "Services, etc.), cluster operations, or container orchestration workflows, plus "
    "ordinary small talk (greetings, thanks). Do not allow general programming requests "
    "(writing code, solving an algorithm, explaining a language feature) that are not "
    "specifically about Kubernetes, even though they are technical, and do not allow any "
    "other subject unrelated to Kubernetes.\n\n"
    "If the user's message violates the policy above, answer only with \"off-topic\". "
    "If it does not violate the policy, answer only with \"on-topic\". Your entire reply "
    "must be exactly \"on-topic\" or \"off-topic\" and nothing else."
)

# --- NeMoGuard content-safety (nemoguard_safety_model). Prompt shape and
# category taxonomy (S1-S23) follow NVIDIA's documented template exactly —
# the model was instruction-tuned against this specific taxonomy and JSON
# output contract, deviating from it would cost calibration, not just
# style. https://docs.api.nvidia.com/nim/reference/nvidia-llama-3_1-nemoguard-8b-content-safety
_SAFETY_CATEGORIES = """S1: Violence.
S2: Sexual.
S3: Criminal Planning/Confessions.
S4: Guns and Illegal Weapons.
S5: Controlled/Regulated Substances.
S6: Suicide and Self Harm.
S7: Sexual (minor).
S8: Hate/Identity Hate.
S9: PII/Privacy.
S10: Harassment.
S11: Threat.
S12: Profanity.
S13: Needs Caution.
S14: Other.
S15: Manipulation.
S16: Fraud/Deception.
S17: Malware.
S18: High Risk Gov Decision Making.
S19: Political/Misinformation/Conspiracy.
S20: Copyright/Trademark/Plagiarism.
S21: Unauthorized Advice.
S22: Illegal Activity.
S23: Immoral/Unethical."""

_SAFETY_TASK = (
    "Task: Check if there is unsafe content in the user or agent messages in the below "
    "conversation in accordance with our safety policy with the below categories. Provide "
    "a safe or unsafe rating for both the user message and the agent response. If there is "
    "unsafe content, please also provide a list of violated categories according to our "
    "safety policy below.\n"
    f"<BEGIN UNSAFE CONTENT CATEGORIES>\n{_SAFETY_CATEGORIES}\n<END UNSAFE CONTENT CATEGORIES>\n"
)

_SAFETY_RESPONSE_FORMAT = (
    'Provide your safety assessment for the above conversation in the below output JSON '
    'format:\n{"User Safety": safe or unsafe rating for the user message, "Response Safety": '
    'safe or unsafe rating for the agent response. Omit if no agent response present. '
    '"Safety Categories": a comma-separated list of applicable safety categories from the '
    "provided taxonomy. Omit if all safe.}\n"
    "Do not include anything other than the output JSON in your response.\nOutput JSON:"
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


def preload() -> None:
    """Force the Colang rails (and the embedding model it downloads on
    first use for flow matching) to build now, rather than lazily on the
    first real check_safety() call. Called once at UI startup so that cost
    is paid during app load with its own loading indicator, not silently
    folded into the first user message's latency."""
    _get_rails()


def _colang_jailbreak_check(raw_message: str) -> bool:
    rails = _get_rails()
    result = rails.generate(messages=[{"role": "user", "content": raw_message}])
    content = result.get("content", "") if isinstance(result, dict) else str(result)
    return any(indicator in content for indicator in JAILBREAK_INDICATORS)


def _direct_safety_check(raw_message: str) -> bool | None:
    # NeMoGuard content-safety, called directly against nim_client — no
    # failover chain, this model only exists on NIM. Returns None (not an
    # exception) on an unparseable response so check_safety can log and
    # fail closed the same way it does for a jailbreak-check failure;
    # NVIDIA's own reference integration treats a JSON parse failure as
    # "unsafe" outright, we defer that call to check_safety's fail-closed
    # path instead so there's one place that decision is made.
    prompt = _SAFETY_TASK + f"<BEGIN CONVERSATION>\nuser: {raw_message}\n<END CONVERSATION>\n" + _SAFETY_RESPONSE_FORMAT
    with provider_call_span(provider="nim", model=settings.nemoguard_safety_model, role="safety_gate"):
        response = nim_client.chat.completions.create(
            model=settings.nemoguard_safety_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
        )
    raw = response.choices[0].message.content or ""
    try:
        parsed = json.loads(raw.strip())
        verdict = str(parsed["User Safety"]).strip().lower()
    except Exception:
        logger.warning("NeMoGuard content-safety returned unparseable output: %r", raw)
        return None
    if verdict == "safe":
        return True
    if verdict == "unsafe":
        return False
    return None


def check_safety(raw_message: str) -> bool:
    # two independent checks, either firing is enough to block. Colang's
    # few-shot flow catches jailbreak-pattern attempts specifically (its
    # strength: jailbreaks share recognizable phrasing regardless of topic).
    # The direct classifier catches general unsafe content that isn't
    # phrased like a jailbreak (violence, harassment, and similar
    # categories), AND now also carries jailbreak/prompt-injection as one of
    # its own categories, so a jailbreak attempt that doesn't closely match
    # any of Colang's few-shot examples still has a real chance of being
    # caught here instead of falling through both layers.
    #
    # The two checks don't depend on each other, so they're dispatched
    # concurrently rather than sequentially: check_safety's latency is
    # bounded by whichever of the two calls is slower, not their sum.
    with ThreadPoolExecutor(max_workers=2) as executor:
        colang_future = executor.submit(_colang_jailbreak_check, raw_message)
        classifier_future = executor.submit(_direct_safety_check, raw_message)
        jailbreak_detected = colang_future.result()
        verdict = classifier_future.result()

    if jailbreak_detected:
        return False
    if verdict is None:
        logger.warning("safety classifier gave unparseable verdict, failing closed")
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
    # instead of refused. NeMoGuard's topic-control model is purpose-tuned
    # for exactly this open-ended judgment, called directly against
    # nim_client (no failover chain — see module docstring above).
    with provider_call_span(provider="nim", model=settings.nemoguard_topic_model, role="topic_gate"):
        response = nim_client.chat.completions.create(
            model=settings.nemoguard_topic_model,
            messages=[
                {"role": "system", "content": TOPIC_POLICY_PROMPT},
                {"role": "user", "content": standalone_question},
            ],
            temperature=0.0,
            max_tokens=20,
        )
    raw = response.choices[0].message.content or ""
    verdict = _parse_binary_verdict(raw, true_word="on-topic", false_word="off-topic")
    if verdict is None:
        logger.warning("topic gate got unparseable verdict %r, failing closed", raw)
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
