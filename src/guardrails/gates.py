import json
import logging
import re

from src.config import settings
from src.guardrails.jailbreak_patterns import deterministic_jailbreak_check
from src.providers.circuit_breaker import CircuitBreaker
from src.providers.clients import nim_client
from src.providers.llm import generate_planner
from src.tracing import provider_call_span

logger = logging.getLogger(__name__)


class GateUnavailableError(RuntimeError):
    pass


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


def _new_breaker() -> CircuitBreaker:
    return CircuitBreaker(settings.guardrail_circuit_failure_threshold, settings.guardrail_circuit_recovery_seconds)


_breakers = {
    "safety": _new_breaker(),
    "response_safety": _new_breaker(),
    "topic": _new_breaker(),
}


def reset_circuit_breakers() -> None:
    for breaker in _breakers.values():
        breaker.record_success()


def _require_verdict(verdict: bool | None, gate: str) -> bool:
    if verdict is None:
        raise GateUnavailableError(f"{gate} classifier unavailable or unparseable")
    return verdict


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


def _extract_json_object(raw: str) -> dict | None:
    stripped = raw.strip()
    try:
        parsed = json.loads(stripped)
    except Exception:
        parsed = None
    if isinstance(parsed, dict):
        return parsed
    match = re.search(r"\{.*\}", stripped, re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _parse_safety_field(raw: str, field: str) -> bool | None:
    parsed = _extract_json_object(raw)
    if parsed is None:
        logger.warning("safety classifier returned unparseable output: %r", raw)
        return None
    verdict = str(parsed.get(field, "")).strip().lower()
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


def _build_conversation(user_message: str, response_message: str | None) -> str:
    lines = [f"user: {user_message}"]
    if response_message is not None:
        lines.append(f"response: agent: {response_message}")
    return "\n".join(lines) + "\n"


def _safety_prompt(user_message: str, response_message: str | None) -> str:
    conversation = _build_conversation(user_message, response_message)
    return _SAFETY_TASK + f"<BEGIN CONVERSATION>\n{conversation}<END CONVERSATION>\n" + _SAFETY_RESPONSE_FORMAT


def _safety_field_for(response_message: str | None) -> str:
    return "User Safety" if response_message is None else "Response Safety"


def _safety_breaker_for(response_message: str | None) -> CircuitBreaker:
    return _breakers["safety"] if response_message is None else _breakers["response_safety"]


def _direct_safety_check(user_message: str, response_message: str | None = None) -> bool | None:
    prompt = _safety_prompt(user_message, response_message)
    role = "safety_gate" if response_message is None else "response_safety_gate"
    response = _call_nemoguard(settings.nemoguard_safety_model, [{"role": "user", "content": prompt}], 200, role)
    return _parse_safety_field(response.choices[0].message.content or "", _safety_field_for(response_message))


def _fallback_safety_check(user_message: str, response_message: str | None = None) -> bool | None:
    prompt = _safety_prompt(user_message, response_message)
    try:
        result = generate_planner(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=settings.classifier_max_tokens,
            timeout_seconds=settings.planner_timeout_seconds,
        )
    except Exception as error:
        logger.warning("safety fallback classifier failed: %s", error)
        return None
    return _parse_safety_field(result.content, _safety_field_for(response_message))


def _safety_classifier_verdict(user_message: str, response_message: str | None = None) -> bool | None:
    if settings.guardrail_skip_nemoguard_safety:
        return _fallback_safety_check(user_message, response_message)
    breaker = _safety_breaker_for(response_message)
    if not breaker.allow():
        logger.warning("%s NeMoGuard circuit open, using fallback classifier", _safety_field_for(response_message))
        return _fallback_safety_check(user_message, response_message)
    try:
        verdict = _direct_safety_check(user_message, response_message)
    except Exception as error:
        breaker.record_failure()
        logger.warning("NeMoGuard safety call failed: %s", error)
        return _fallback_safety_check(user_message, response_message)
    if verdict is None:
        breaker.record_failure()
        return _fallback_safety_check(user_message, response_message)
    breaker.record_success()
    return verdict


def check_safety(raw_message: str) -> bool:
    if deterministic_jailbreak_check(raw_message):
        return False
    return _require_verdict(_safety_classifier_verdict(raw_message), "safety")


def check_response_safety(user_message: str, response_message: str) -> bool:
    return _require_verdict(_safety_classifier_verdict(user_message, response_message), "response safety")


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
            max_tokens=settings.classifier_max_tokens,
            timeout_seconds=settings.planner_timeout_seconds,
        )
    except Exception as error:
        logger.warning("topic fallback classifier failed: %s", error)
        return None
    return _parse_binary_verdict(result.content, true_word="on-topic", false_word="off-topic")


_SMALL_TALK_PHRASES = {
    "hi",
    "hii",
    "hiya",
    "hello",
    "hey",
    "hey there",
    "yo",
    "sup",
    "good morning",
    "good afternoon",
    "good evening",
    "thanks",
    "thank you",
    "thanks a lot",
    "thank you very much",
    "ok thanks",
    "okay thanks",
    "bye",
    "goodbye",
    "see you",
    "cheers",
}


def is_small_talk(message: str) -> bool:
    normalized = re.sub(r"\s+", " ", message.strip().lower()).rstrip("!.,")
    return normalized in _SMALL_TALK_PHRASES


def check_topic(standalone_question: str) -> bool:
    if is_small_talk(standalone_question):
        return True
    if settings.guardrail_skip_nemoguard_topic:
        return _require_verdict(_fallback_topic_check(standalone_question), "topic")
    breaker = _breakers["topic"]
    if not breaker.allow():
        logger.warning("topic NeMoGuard circuit open, using fallback classifier")
        return _require_verdict(_fallback_topic_check(standalone_question), "topic")
    verdict = None
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
        logger.warning("NeMoGuard topic call failed: %s", error)
    if verdict is None:
        breaker.record_failure()
        return _require_verdict(_fallback_topic_check(standalone_question), "topic")
    breaker.record_success()
    return verdict


def _run_gate(check, refusal: str, name: str, *args) -> tuple[bool, str | None]:
    try:
        allowed = check(*args)
    except GateUnavailableError:
        raise
    except Exception as error:
        logger.error("%s gate failed: %s", name, error)
        raise GateUnavailableError(f"{name} gate failed: {error}") from error
    if not allowed:
        return False, refusal
    return True, None


def safety_gate(raw_message: str) -> tuple[bool, str | None]:
    return _run_gate(check_safety, UNSAFE_REFUSAL, "safety", raw_message)


def topic_gate(standalone_question: str) -> tuple[bool, str | None]:
    return _run_gate(check_topic, OFF_TOPIC_REFUSAL, "topic", standalone_question)


def response_safety_gate(user_message: str, response_message: str) -> tuple[bool, str | None]:
    return _run_gate(check_response_safety, UNSAFE_REFUSAL, "response safety", user_message, response_message)
