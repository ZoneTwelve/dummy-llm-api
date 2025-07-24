import time
import threading
from collections import defaultdict

class RPMRateLimiter:
    def __init__(self, max_rpm: int):
        self.max_rpm = max_rpm
        self._lock = threading.Lock()
        self._requests = 0
        self._reset_at = time.time() + 60

    def check(self):
        now = time.time()
        with self._lock:
            if now >= self._reset_at:
                self._requests = 0
                self._reset_at = now + 60

            if self._requests >= self.max_rpm:
                raise RuntimeError("RPM limit exceeded.")
            self._requests += 1


class SessionTokenThrottle:
    def __init__(self, tokens_per_sec: int):
        self.tokens_per_sec = tokens_per_sec
        self.session_last_emit: dict[str, float] = defaultdict(float)
        self.session_token_cursor: dict[str, float] = defaultdict(float)

    def sleep_until_next_token(self, session_id: str, tokens_to_emit: int = 1):
        now = time.time()
        interval = 1.0 / self.tokens_per_sec

        # Get the last emission time
        last_emit = self.session_token_cursor[session_id]

        # Compute when we’re allowed to emit next
        next_time = max(now, last_emit + tokens_to_emit * interval)
        sleep_time = next_time - now

        if sleep_time > 0:
            time.sleep(sleep_time)

        self.session_token_cursor[session_id] = next_time
