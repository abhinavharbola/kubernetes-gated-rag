import json
import logging
import re

from src.guardrails.colang_rules import deterministic_jailbreak_check
from src.config import settings
from src.providers.circuit_breaker import CircuitBreaker
from src.providers.clients import nim_client
from src.providers.llm import generate_planner
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

TOPIC_POLICY_PROMPT = (
    "You are a Kubernetes documentation assistant, providing users with factual, "
    "technical information about Kubernetes and container orchestration. Your role is "
    "to ensure that you respond only to relevant queries and adhere to the following "
    "guidelines.\n\n"
    "Guidelines for the user messages:\n"
    "- Allow questions about Kubernetes objects, manifests, controllers (Deployments, "
    "StatefulSets, Services, etc.), cluster operations, and container orchestration "
    "workflows, including basic definitional questions like \"what is Kubernetes\".\n"
    "- Allow ordinary small talk such as greetings and thanks.\n"
    "- Do not allow general programming requests (writing code, solving an algorithm, "
    "explaining a language feature) that are not specifically about Kubernetes, even "
    "though they are technical.\n"
    "- Do not allow any other subject unrelated to Kubernetes.\n\n"
    'If any of the above guidelines are violated, please respond with "off-topic". '
    'Otherwise, respond with "on-topic".'
)

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


_breakers = {
    "safety": CircuitBreaker(settings.guardrail_circuit_failure_threshold, settings.guardrail_circuit_recovery_seconds),
    "topic": CircuitBreaker(settings.guardrail_circuit_failure_threshold, settings.guardrail_circuit_recovery_seconds),
}


def reset_circuit_breakers() -> None:
    for breaker in _breakers.values():
        breaker.record_success()


def _parse_binary_verdict(raw: str, true_word: str, false_word: str) -> bool | None:
    stripped = raw.strip()
    if not stripped:
        return None
    first_token = stripped.split()[0].strip(".,!?\"'").lower()
    if first_token == false_word:
        return False
    if first_token == true_word:
        return True
    last_line = stripped.splitlines()[-1].strip().strip(".,!?\"'").lower()
    if last_line == false_word:
        return False
    if last_line == true_word:
        return True
    return None


def _parse_safety_json(raw: str) -> bool | None:
    stripped = raw.strip()
    try:
        parsed = json.loads(stripped)
    except Exception:
        match = re.search(r"\{.*\}", stripped, re.DOTALL)
        if not match:
            logger.warning("safety classifier returned unparseable output: %r", raw)
            return None
        try:
            parsed = json.loads(match.group(0))
        except Exception:
            logger.warning("safety classifier returned unparseable output: %r", raw)
            return None
    verdict = str(parsed.get("User Safety", "")).strip().lower()
    if verdict == "safe":
        return True
    if verdict == "unsafe":
        return False
    return None


def _call_nemoguard(model: str, messages: list[dict], max_tokens: int, role: str):
    with provider_call_span(provider="nim", model=model, role=role):
        return nim_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=max_tokens,
            timeout=settings.guardrail_timeout_seconds,
        )


def _direct_safety_check(raw_message: str) -> bool | None:
    prompt = _SAFETY_TASK + f"<BEGIN CONVERSATION>\nuser: {raw_message}\n<END CONVERSATION>\n" + _SAFETY_RESPONSE_FORMAT
    response = _call_nemoguard(
        settings.nemoguard_safety_model,
        [{"role": "user", "content": prompt}],
        200,
        "safety_gate",
    )
    return _parse_safety_json(response.choices[0].message.content or "")


def _fallback_safety_check(raw_message: str) -> bool | None:
    prompt = _SAFETY_TASK + f"<BEGIN CONVERSATION>\nuser: {raw_message}\n<END CONVERSATION>\n" + _SAFETY_RESPONSE_FORMAT
    try:
        result = generate_planner(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=250,
            timeout_seconds=settings.planner_timeout_seconds,
        )
    except Exception as error:
        logger.warning("safety fallback classifier failed: %s", error)
        return None
    return _parse_safety_json(result.content)


def _safety_classifier_verdict(raw_message: str) -> bool | None:
    if settings.guardrail_skip_nemoguard_safety:
        return _fallback_safety_check(raw_message)
    breaker = _breakers["safety"]
    if not breaker.allow():
        logger.warning("safety NeMoGuard circuit open, using fallback classifier")
        return _fallback_safety_check(raw_message)
    try:
        verdict = _direct_safety_check(raw_message)
    except Exception as error:
        breaker.record_failure()
        logger.warning("NeMoGuard safety call failed: %s", error)
        return _fallback_safety_check(raw_message)
    if verdict is None:
        breaker.record_failure()
        return _fallback_safety_check(raw_message)
    breaker.record_success()
    return verdict


def check_safety(raw_message: str) -> bool:
    if deterministic_jailbreak_check(raw_message):
        return False
    verdict = _safety_classifier_verdict(raw_message)
    if verdict is None:
        logger.warning("safety classifier unavailable or unparseable, failing closed")
        return False
    return verdict


def _fallback_topic_check(standalone_question: str) -> bool | None:
    try:
        result = generate_planner(
            [
                {
                    "role": "system",
                    "content": TOPIC_POLICY_PROMPT + '\n\nRespond with only the single word "on-topic" or "off-topic".',
                },
                {"role": "user", "content": standalone_question},
            ],
            temperature=0.0,
            max_tokens=100,
            timeout_seconds=settings.planner_timeout_seconds,
        )
    except Exception as error:
        logger.warning("topic fallback classifier failed: %s", error)
        return None
    return _parse_binary_verdict(result.content, true_word="on-topic", false_word="off-topic")


def _is_small_talk(standalone_question: str) -> bool:
    normalized = re.sub(r"\s+", " ", standalone_question.strip().lower())
    return normalized in {
        "hi",
        "hello",
        "hey",
        "good morning",
        "good afternoon",
        "thanks",
        "thank you",
        "bye",
        "goodbye",
    }


def check_topic(standalone_question: str) -> bool:
    if _is_small_talk(standalone_question):
        return True
    if settings.guardrail_skip_nemoguard_topic:
        verdict = _fallback_topic_check(standalone_question)
        return verdict if verdict is not None else False
    breaker = _breakers["topic"]
    if not breaker.allow():
        logger.warning("topic NeMoGuard circuit open, using fallback classifier")
        verdict = _fallback_topic_check(standalone_question)
        return verdict if verdict is not None else False
    try:
        response = _call_nemoguard(
            settings.nemoguard_topic_model,
            [
                {"role": "system", "content": TOPIC_POLICY_PROMPT},
                {"role": "user", "content": standalone_question},
            ],
            20,
            "topic_gate",
        )
        verdict = _parse_binary_verdict(
            response.choices[0].message.content or "",
            true_word="on-topic",
            false_word="off-topic",
        )
    except Exception as error:
        breaker.record_failure()
        logger.warning("NeMoGuard topic call failed: %s", error)
        verdict = None
    if verdict is None:
        fallback_verdict = _fallback_topic_check(standalone_question)
        return fallback_verdict if fallback_verdict is not None else False
    breaker.record_success()
    return verdict


def safety_gate(raw_message: str) -> tuple[bool, str | None]:
    try:
        if not check_safety(raw_message):
            return False, UNSAFE_REFUSAL
        return True, None
    except Exception as error:
        logger.error("safety gate failed, failing closed: %s", error)
        return False, UNSAFE_REFUSAL


def topic_gate(standalone_question: str) -> tuple[bool, str | None]:
    try:
        if not check_topic(standalone_question):
            return False, OFF_TOPIC_REFUSAL
        return True, None
    except Exception as error:
        logger.error("topic gate failed, failing closed: %s", error)
        return False, OFF_TOPIC_REFUSAL
