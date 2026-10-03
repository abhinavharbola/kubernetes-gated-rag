from src.history import build_context_history, compute_session_stats


def _turn(question, answer, **details):
    return [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer, "details": details},
    ]


def test_normal_turns_are_kept_in_order():
    turns = _turn("q1", "a1") + _turn("q2", "a2")
    history = build_context_history(turns)
    assert [m["content"] for m in history] == ["q1", "a1", "q2", "a2"]
    assert all(set(m) == {"role", "content"} for m in history)


def test_refused_errored_unavailable_and_small_talk_turns_are_dropped():
    turns = (
        _turn("bad", "refusal", blocked_stage="safety")
        + _turn("boom", "error", error="x")
        + _turn("down", "unavailable", service_unavailable=True)
        + _turn("hi", "hello", small_talk=True)
        + _turn("real question", "real answer")
    )
    history = build_context_history(turns)
    assert [m["content"] for m in history] == ["real question", "real answer"]


def test_history_is_capped_to_the_most_recent_messages():
    turns = []
    for i in range(10):
        turns += _turn(f"q{i}", f"a{i}")
    history = build_context_history(turns)
    assert len(history) == 6
    assert history[-1]["content"] == "a9"
    assert history[0]["content"] == "q7"


def test_empty_history_stays_empty():
    assert build_context_history([]) == []


def test_unanswered_user_turn_does_not_misalign_pairing():
    turns = [
        {"role": "user", "content": "interrupted question"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2", "details": {}},
        {"role": "user", "content": "q3"},
        {"role": "assistant", "content": "a3", "details": {}},
    ]
    history = build_context_history(turns)
    assert [m["content"] for m in history] == ["q2", "a2", "q3", "a3"]
    assert [m["role"] for m in history] == ["user", "assistant", "user", "assistant"]


def test_session_stats_empty():
    assert compute_session_stats([]) == (0, "-", "-")


def test_session_stats_hit_rate_only_counts_answered_questions():
    turns = (
        _turn("q1", "a1", cache_layer="exact", latency_seconds=0.1)
        + _turn("q2", "a2", cache_layer=None, latency_seconds=3.0)
        + _turn("hi", "hello", small_talk=True, latency_seconds=0.0)
        + _turn("bad", "refused", blocked_stage="safety", latency_seconds=0.5)
        + _turn("boom", "error", error="x")
    )
    total, hit_rate, avg_latency = compute_session_stats(turns)
    assert total == 5
    assert hit_rate == "50%"
    assert avg_latency == "0.90s"


def test_session_stats_hit_rate_is_dash_when_nothing_was_answered():
    turns = _turn("hi", "hello", small_talk=True) + _turn("bad", "refused", blocked_stage="safety")
    assert compute_session_stats(turns)[1] == "-"
