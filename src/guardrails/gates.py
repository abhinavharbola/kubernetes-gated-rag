import json
import logging
from concurrent.futures import ThreadPoolExecutor

from langchain_groq import ChatGroq
from nemoguardrails import LLMRails, RailsConfig
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.guardrails.colang_rules import COLANG_CONTENT, JAILBREAK_INDICATORS, YAML_CONTENT
from src.config import settings
from src.providers.clients import nim_client
from src.providers.llm import RETRYABLE
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