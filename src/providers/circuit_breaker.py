import threading
import time


class CircuitBreaker:
    """Opens after failure_threshold consecutive failures, refuses calls
    until recovery_seconds has passed, then fully closes again (not a
    gradual half-open trial — the next call after recovery just tries
    normally and reopens fast if the provider is still down)."""

    def __init__(self, failure_threshold: int, recovery_seconds: float):
        self.failure_threshold = failure_threshold
        self.recovery_seconds = recovery_seconds
        self.failures = 0
        self.opened_at = 0.0
        self._lock = threading.Lock()

    def allow(self) -> bool:
        with self._lock:
            if self.opened_at == 0.0:
                return True
            if time.monotonic() - self.opened_at >= self.recovery_seconds:
                self.opened_at = 0.0
                self.failures = 0
                return True
            return False

    def record_success(self) -> None:
        with self._lock:
            self.failures = 0
            self.opened_at = 0.0

    def record_failure(self) -> None:
        with self._lock:
            self.failures += 1
            if self.failures >= self.failure_threshold:
                self.opened_at = time.monotonic()
