import threading
import time


class RateLimiter:
    """
    Process-wide throttle shared by every Gemini call (extraction + reconciliation).
    Ensures that no matter how many documents are being ingested concurrently,
    actual API calls are serialized and spaced apart — this is what actually
    prevents 429/RESOURCE_EXHAUSTED errors, rather than retrying after the fact.
    """

    def __init__(self, min_interval_seconds: float = 6.0, max_concurrent: int = 1):
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
# Tune min_interval_seconds up if you're still seeing 429s — this is a floor,
# not a target: e.g. 6.0 = max ~10 requests/minute across the whole app.
gemini_rate_limiter = RateLimiter(min_interval_seconds=6.0, max_concurrent=1)


def is_rate_limit_error(exc: Exception) -> bool:
    """Heuristic check since the SDK doesn't always expose a clean status code."""
    msg = str(exc).upper()
    return "RESOURCE_EXHAUSTED" in msg or "429" in msg or "QUOTA" in msg or "RATE" in msg


def backoff_delay(attempt: int, is_rate_limit: bool) -> float:
    """
    Exponential backoff with jitter. Rate-limit errors get a much longer base
    delay than transient/network errors, since retrying quickly just burns
    more of the same quota window.
    """
    import random
    base = 20.0 if is_rate_limit else 3.0
    delay = base * (2 ** attempt)
    jitter = random.uniform(0, base * 0.5)
    return min(delay + jitter, 90.0)  # cap so a single page can't stall forever