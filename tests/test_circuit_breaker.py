import time

from src.providers.circuit_breaker import CircuitBreaker


def test_breaker_starts_closed_and_opens_at_the_threshold():
    breaker = CircuitBreaker(failure_threshold=2, recovery_seconds=60)
    assert breaker.allow() is True
    breaker.record_failure()
    assert breaker.allow() is True
    breaker.record_failure()
    assert breaker.allow() is False


def test_success_resets_the_failure_count():
    breaker = CircuitBreaker(failure_threshold=2, recovery_seconds=60)
    breaker.record_failure()
    breaker.record_success()
    breaker.record_failure()
    assert breaker.allow() is True


def test_after_recovery_only_a_single_probe_is_admitted():
    breaker = CircuitBreaker(failure_threshold=1, recovery_seconds=0.05)
    breaker.record_failure()
    assert breaker.allow() is False
    time.sleep(0.06)
    assert breaker.allow() is True
    assert breaker.allow() is False


def test_a_failed_probe_reopens_immediately():
    breaker = CircuitBreaker(failure_threshold=3, recovery_seconds=0.05)
    for _ in range(3):
        breaker.record_failure()
    time.sleep(0.06)
    assert breaker.allow() is True
    breaker.record_failure()
    assert breaker.allow() is False


def test_a_successful_probe_closes_the_breaker():
    breaker = CircuitBreaker(failure_threshold=1, recovery_seconds=0.05)
    breaker.record_failure()
    time.sleep(0.06)
    assert breaker.allow() is True
    breaker.record_success()
    assert breaker.allow() is True
    assert breaker.allow() is True


def test_late_failures_do_not_extend_the_open_window():
    breaker = CircuitBreaker(failure_threshold=1, recovery_seconds=0.05)
    breaker.record_failure()
    opened_at = breaker.opened_at
    time.sleep(0.02)
    breaker.record_failure()
    assert breaker.opened_at == opened_at
