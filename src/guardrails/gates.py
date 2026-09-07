import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor

from langchain_groq import ChatGroq
from nemoguardrails import LLMRails, RailsConfig
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.guardrails.colang_rules import COLANG_CONTENT, JAILBREAK_INDICATORS, YAML_CONTENT
from src.config import settings
from src.providers.clients import nim_client
from src.providers.llm import RETRYABLE, generate_planner
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
# This is a narrowly LoRA-tuned classifier (not a general instruction-
# following model), fine-tuned on a specific system-prompt shape: a
# persona assignment ("you are a ___ assistant, providing users with
# ___") plus a bulleted guidelines list, not a meta-instruction telling
# the model it IS a topic classifier. Deviating from that shape is what
# caused this gate to misclassify obviously on-topic questions as
# off-topic — the model was answering "does this fit the assigned
# persona's domain" against a persona it never got a clean read on.
# Structure and the output-restriction sentence follow NVIDIA's
# documented format as closely as sensible; the bulleted guidelines
# content itself is ours.
# https://docs.nvidia.com/nim/llama-3-1-nemoguard-8b-topiccontrol/latest/getting-started.html
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


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=4),
    retry=retry_if_exception_type(RETRYABLE),
    reraise=True,
)
def _call_nemoguard(model: str, messages: list[dict], max_tokens: int, role: str):
    # NVIDIA's hosted NeMoGuard instances occasionally crash server-side
    # with a TensorRT-LLM CUDA error (illegal memory access in the batch
    # manager) rather than returning a clean error, this is infra flakiness
    # on their end, not a property of the specific request. The openai SDK
    # maps a 500 response to InternalServerError, which RETRYABLE already
    # covers (see src/providers/llm.py), so this reuses that same tuple
    # rather than defining a second one that could drift out of sync.
    #
    # 3 attempts, not the 2 every other chain in this app uses: those
    # chains have a next provider to fail over to after their 2 attempts,
    # this gate doesn't — NeMoGuard only exists on NIM, there's nowhere
    # else to send the request. Observed failures on this specific model
    # have repeated across consecutive requests in the same session rather
    # than clearing on the very next call (consistent with a load balancer
    # repeatedly routing back to the same unhealthy worker, or a sustained
    # bad stretch on NVIDIA's end rather than one-off flakiness), so a
    # single retry undersells what's needed here. This only costs latency
    # on the failure path, a healthy call still returns on the first
    # attempt, and topic_gate()/safety_gate() below still fail closed if
    # all 3 attempts fail, this just gives a real outage more room to
    # clear before a genuinely on-topic/safe question gets refused.
    with provider_call_span(provider="nim", model=model, role=role):
        return nim_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=max_tokens,
        )


def _parse_binary_verdict(raw: str, true_word: str, false_word: str) -> bool | None:
    # Checks the first token and, if that alone isn't the bare verdict, the
    # last non-empty line, never an "does this word appear anywhere"
    # substring search, since that misfires on any preamble at all, e.g.
    # "this is safe, not unsafe" contains "unsafe" as a literal substring
    # despite the actual verdict being safe. NeMoGuard's purpose-tuned classifiers
    # answer with just the verdict word, so the first token alone always
    # resolves it for them, and for a terse response the first token and
    # last line are the same thing anyway, so this doesn't change their
    # behavior. It matters for the fallback classifier (see
    # _fallback_topic_check): a general instruction-following model is
    # more likely to reason before landing on an answer, so the verdict is
    # more likely to be the last thing it says than the first, this is
    # still a whole-line match, not a substring search, just checked in
    # one more place.
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


def _get_rails() -> LLMRails:
    # lazily built once per process rather than at import time: Colang
    # parsing has real startup cost, worth paying once, not per request, but
    # also not worth paying at all for code paths that never call the gate.
    global _rails
    if _rails is None:
        # request_timeout/max_retries match every other client in this app
        # (see src/providers/clients.py) — ChatGroq's own defaults are
        # request_timeout=None (falls back to the SDK's ~60s default) and
        # max_retries=2, which with backoff can sum to 90+ seconds for a
        # single slow/erroring call. That mismatch was the actual cause of
        # multi-minute safety-gate latency: this call was silently exempt
        # from the tight timeout budget every other provider call honors.
        guard_llm = ChatGroq(
            api_key=settings.groq_api_key,
            model=settings.groq_planner_model,
            temperature=0,
            request_timeout=15.0,
            max_retries=0,
        )
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


def _parse_safety_json(raw: str) -> bool | None:
    # Tries the response as-is first (NeMoGuard returns pure JSON with
    # nothing else around it, so this succeeds immediately on the primary
    # path). Falls back to pulling out the first {...} block only if that
    # fails, since a general instruction-following model (the fallback
    # classifier, see _fallback_safety_check) is more likely to reason
    # before or around the JSON than NeMoGuard's tuned output is.
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
    try:
        verdict = str(parsed["User Safety"]).strip().lower()
    except Exception:
        logger.warning("safety classifier JSON missing 'User Safety' key: %r", raw)
        return None
    if verdict == "safe":
        return True
    if verdict == "unsafe":
        return False
    return None


def _direct_safety_check(raw_message: str) -> bool | None:
    # NeMoGuard content-safety, called directly against nim_client — no
    # failover chain, this model only exists on NIM. Returns None (not an
    # exception) on an unparseable response so check_safety can log and
    # fail closed the same way it does for a jailbreak-check failure;
    # NVIDIA's own reference integration treats a JSON parse failure as
    # "unsafe" outright, we defer that call to check_safety's fail-closed
    # path instead so there's one place that decision is made.
    prompt = _SAFETY_TASK + f"<BEGIN CONVERSATION>\nuser: {raw_message}\n<END CONVERSATION>\n" + _SAFETY_RESPONSE_FORMAT
    response = _call_nemoguard(
        model=settings.nemoguard_safety_model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        role="safety_gate",
    )
    raw = response.choices[0].message.content or ""
    return _parse_safety_json(raw)


def _fallback_safety_check(raw_message: str) -> bool | None:
    # Used only when the dedicated NeMoGuard content-safety model itself
    # errors after _call_nemoguard's retries are exhausted (an NVIDIA-side
    # outage, not a verdict) — see _fallback_topic_check's docstring below
    # for the full reasoning, this is the same pattern applied to the
    # safety check. generate_planner's chain (NIM gpt-oss -> Groq gpt-oss)
    # doesn't touch the NeMoGuard deployment at all, so an outage there
    # doesn't take this down too. A general instruction-following model
    # wasn't calibrated against NeMoGuard's specific S1-S23 taxonomy the
    # way NeMoGuard itself was, so this is a genuinely lower-confidence
    # fallback, not a drop-in equivalent, but a lower-confidence safety
    # check is still better than refusing every message during an outage
    # that has nothing to do with whether the message is actually unsafe.
    prompt = _SAFETY_TASK + f"<BEGIN CONVERSATION>\nuser: {raw_message}\n<END CONVERSATION>\n" + _SAFETY_RESPONSE_FORMAT
    try:
        # max_tokens well above the primary check's 200: NeMoGuard was
        # tuned to answer with just the JSON object, a general model is
        # more likely to reason before or around it, and a response
        # truncated before it ever reaches the JSON is unparseable no
        # matter how good _parse_safety_json's extraction gets.
        result = generate_planner([{"role": "user", "content": prompt}], temperature=0.0, max_tokens=400)
    except Exception as error:
        logger.warning("safety gate fallback classifier also failed: %s", error)
        return None
    verdict = _parse_safety_json(result.content)
    if verdict is None:
        logger.warning("safety gate fallback classifier raw response: %r", result.content)
    return verdict


def _safety_classifier_verdict(raw_message: str) -> bool | None:
    # Tries the dedicated NeMoGuard content-safety model first, it's
    # calibrated against the specific S1-S23 taxonomy this prompt uses, the
    # fallback below isn't. Falls back to _fallback_safety_check whenever
    # the primary attempt didn't produce a usable verdict, whether that's
    # because _call_nemoguard's retries were exhausted (an NVIDIA-side
    # outage) or because it returned something unparseable, both are cases
    # where treating "no usable answer" as "fail closed" would refuse the
    # message for a reason unrelated to whether it's actually unsafe.
    if settings.guardrail_skip_nemoguard:
        logger.info("guardrail_skip_nemoguard is set, going straight to fallback classifier")
        return _fallback_safety_check(raw_message)
    try:
        verdict = _direct_safety_check(raw_message)
    except Exception as error:
        logger.warning("NeMoGuard content-safety call failed after retries (%s), trying fallback classifier", error)
        return _fallback_safety_check(raw_message)
    if verdict is not None:
        return verdict
    logger.warning("safety classifier gave unparseable verdict, trying fallback classifier")
    return _fallback_safety_check(raw_message)


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
        classifier_future = executor.submit(_safety_classifier_verdict, raw_message)
        jailbreak_detected = colang_future.result()
        verdict = classifier_future.result()

    if jailbreak_detected:
        return False
    if verdict is None:
        logger.warning("both safety classifiers gave no usable verdict, failing closed")
        return False
    return verdict


def _fallback_topic_check(standalone_question: str) -> bool | None:
    # Used only when the dedicated NeMoGuard topic-control model itself
    # errors after _call_nemoguard's retries are exhausted — an NVIDIA-side
    # outage on that specific hosted model, not this classifier having
    # judged the question off-topic. Failing every question closed for the
    # duration of an unrelated infra outage makes the whole app unusable,
    # so this falls back to the same generate_planner chain (NIM gpt-oss ->
    # Groq gpt-oss) that fronted this exact judgment before NeMoGuard
    # existed (see the module docstring above check_topic). It's not
    # NeMoGuard's purpose-tuned calibration, a general instruction-
    # following model reading the same policy prompt is a genuinely
    # lower-confidence topic judgment, but it's a real judgment rather than
    # a blanket refusal, and it only ever runs when the primary classifier
    # is already down.
    # Extra instruction appended only here, not in TOPIC_POLICY_PROMPT
    # itself: NeMoGuard was tuned to answer with just the verdict word
    # without needing to be told twice, a general model benefits from the
    # explicit reminder not to reason out loud first.
    fallback_instruction = (
        "\n\nRespond with only the single word \"on-topic\" or \"off-topic\" and nothing else."
    )
    try:
        # max_tokens well above the primary check's 20: a response that
        # gets truncated mid-reasoning never reaches a verdict at all, no
        # matter how the parser looks for one.
        result = generate_planner(
            [
                {"role": "system", "content": TOPIC_POLICY_PROMPT + fallback_instruction},
                {"role": "user", "content": standalone_question},
            ],
            temperature=0.0,
            max_tokens=300,
        )
    except Exception as error:
        logger.warning("topic gate fallback classifier also failed: %s", error)
        return None
    verdict = _parse_binary_verdict(result.content, true_word="on-topic", false_word="off-topic")
    if verdict is None:
        logger.warning("topic gate fallback classifier raw response: %r", result.content)
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
    # nim_client; if that call itself fails (not a verdict, an actual
    # outage), check_topic falls back to _fallback_topic_check rather than
    # treating an NVIDIA infra crash as a genuine off-topic classification.
    if settings.guardrail_skip_nemoguard:
        logger.info("guardrail_skip_nemoguard is set, going straight to fallback classifier")
        fallback_verdict = _fallback_topic_check(standalone_question)
        if fallback_verdict is None:
            logger.warning("topic gate fallback classifier gave no usable verdict, failing closed")
            return False
        return fallback_verdict
    try:
        response = _call_nemoguard(
            model=settings.nemoguard_topic_model,
            messages=[
                {"role": "system", "content": TOPIC_POLICY_PROMPT},
                {"role": "user", "content": standalone_question},
            ],
            max_tokens=20,
            role="topic_gate",
        )
        raw = response.choices[0].message.content or ""
        logger.info("topic gate raw response: %r", raw)
        verdict = _parse_binary_verdict(raw, true_word="on-topic", false_word="off-topic")
        if verdict is not None:
            return verdict
        logger.warning("topic gate got unparseable verdict %r, trying fallback classifier", raw)
    except Exception as error:
        logger.warning("topic gate's NeMoGuard call failed after retries (%s), trying fallback classifier", error)

    fallback_verdict = _fallback_topic_check(standalone_question)
    if fallback_verdict is None:
        logger.warning("topic gate fallback classifier also gave no usable verdict, failing closed")
        return False
    return fallback_verdict


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