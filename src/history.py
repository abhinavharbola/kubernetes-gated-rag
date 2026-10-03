from src.config import settings

_NOT_ANSWERED_FLAGS = ("error", "blocked_stage", "service_unavailable", "small_talk")


def _is_answered(details: dict) -> bool:
    return not any(details.get(flag) for flag in _NOT_ANSWERED_FLAGS)


def build_context_history(turns: list[dict]) -> list[dict]:
    usable: list[dict] = []
    for index in range(len(turns) - 1):
        user_turn, assistant_turn = turns[index], turns[index + 1]
        if user_turn["role"] != "user" or assistant_turn["role"] != "assistant":
            continue
        details = assistant_turn.get("details") or {}
        if not _is_answered(details):
            continue
        usable.append({"role": "user", "content": user_turn["content"]})
        usable.append({"role": "assistant", "content": assistant_turn["content"]})
    return usable[-settings.history_max_messages :]


def compute_session_stats(turns: list[dict]) -> tuple[int, str, str]:
    assistant_turns = [turn for turn in turns if turn["role"] == "assistant"]
    total = len(assistant_turns)
    if total == 0:
        return 0, "-", "-"

    all_details = [turn.get("details") or {} for turn in assistant_turns]
    answered = [details for details in all_details if _is_answered(details)]
    if answered:
        hits = sum(1 for details in answered if details.get("cache_layer"))
        hit_rate = f"{round(hits / len(answered) * 100)}%"
    else:
        hit_rate = "-"

    latencies = [d["latency_seconds"] for d in all_details if d.get("latency_seconds") is not None]
    avg_latency = f"{sum(latencies) / len(latencies):.2f}s" if latencies else "-"
    return total, hit_rate, avg_latency
