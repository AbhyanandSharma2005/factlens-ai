import threading
import time


class RateLimiter:
    """
    Process-wide throttle shared by every Groq call (extraction + reconciliation).
    Ensures that no matter how many documents are being ingested concurrently,
    actual API calls are serialized and spaced apart.

    Groq free tier: 30 requests/minute, 14,400 requests/day (org-wide, shared
    across all models). A 2.2s floor keeps you safely under 30 RPM (27/min)
    with headroom for retries.
    """

    def __init__(self, min_interval_seconds: float = 2.2, max_concurrent: int = 1):
        self._slot = threading.Semaphore(max_concurrent)
        self._min_interval = min_interval_seconds
        self._timing_lock = threading.Lock()
        self._last_call_time = 0.0

    def acquire(self):
        self._slot.acquire()
        with self._timing_lock:
            elapsed = time.monotonic() - self._last_call_time
            wait = self._min_interval - elapsed
            if wait > 0:
                time.sleep(wait)
            self._last_call_time = time.monotonic()

    def release(self):
        self._slot.release()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


# Single shared instance imported by both extraction.py and reconciler.py.
groq_rate_limiter = RateLimiter(min_interval_seconds=15.0, max_concurrent=1)


def is_rate_limit_error(exc: Exception) -> bool:
    """Heuristic check across providers' differing error message formats."""
    msg = str(exc).upper()
    return "RATE_LIMIT" in msg or "429" in msg or "QUOTA" in msg or "TOO MANY REQUESTS" in msg


def is_retryable_error(exc: Exception) -> bool:
    """
    Only retry errors that have a real chance of succeeding on a second attempt.

    Rate limits (429) and transient server-side/network issues are worth waiting
    out. A 400 (e.g. Groq's json_validate_failed) is a deterministic failure for
    that exact prompt+page combination -- retrying it just replays the same
    failure 5 times at exponential backoff cost, for nothing.
    """
    msg = str(exc).upper()

    if is_rate_limit_error(exc):
        return True

    # Transient / server-side -- worth a retry
    if any(code in msg for code in ("500", "502", "503", "504", "TIMEOUT", "CONNECTION")):
        return True

    # Deterministic client-side failures (bad request, schema/validation failures,
    # auth, not-found, etc.) -- retrying changes nothing, fail fast instead.
    if any(code in msg for code in ("400", "401", "403", "404", "JSON_VALIDATE_FAILED", "INVALID_REQUEST")):
        return False

    # Unknown error shape: default to retryable so we don't silently swallow
    # something that genuinely might be transient (e.g. a new SDK exception type).
    return True


def backoff_delay(attempt: int, is_rate_limit: bool) -> float:
    """
    Exponential backoff with jitter. Rate-limit errors get a longer base delay
    than transient/network errors.
    """
    import random
    base = 15.0 if is_rate_limit else 3.0
    delay = base * (2 ** attempt)
    jitter = random.uniform(0, base * 0.5)
    return min(delay + jitter, 75.0)