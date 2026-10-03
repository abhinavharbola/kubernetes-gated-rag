import threading
import time


class CircuitBreaker:
    def __init__(self, failure_threshold: int, recovery_seconds: float):
        self.failure_threshold = failure_threshold
        self.recovery_seconds = recovery_seconds
        self.failures = 0
        self.opened_at = 0.0
        self._probing = False
        self._lock = threading.Lock()

    def allow(self) -> bool:
        with self._lock:
            if self.opened_at == 0.0:
                return True
            if time.monotonic() - self.opened_at >= self.recovery_seconds:
                if self._probing:
                    return False
                self._probing = True
                return True
            return False

    def record_success(self) -> None:
        with self._lock:
            self.failures = 0
            self.opened_at = 0.0
            self._probing = False

    def record_failure(self) -> None:
        with self._lock:
            self.failures += 1
            if self._probing:
                self.opened_at = time.monotonic()
                self._probing = False
            elif self.failures >= self.failure_threshold and self.opened_at == 0.0:
                self.opened_at = time.monotonic()
