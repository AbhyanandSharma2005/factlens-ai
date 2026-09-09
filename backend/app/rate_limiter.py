import time
from threading import Lock

class TokenBucketRateLimiter:
    """
    Proactive rate limiter that enforces a strict maximum call frequency 
    to prevent ever hitting provider-side rate limits.
    """
    def __init__(self, max_calls_per_minute: int = 12):
        # 12 calls per minute = 1 call every 5 seconds securely within free tier
        self.interval = 60.0 / max_calls_per_minute
        self.last_call_time = 0
        self.lock = Lock()

    def acquire(self):
        with self.lock:
            now = time.time()
            elapsed = now - self.last_call_time
            if elapsed < self.interval:
                sleep_time = self.interval - elapsed
                time.sleep(sleep_time)
            self.last_call_time = time.time()

# Global singleton rate limiter for the entire app
strict_pacer = TokenBucketRateLimiter(max_calls_per_minute=12)