"""API key check and a per-key rate limit for the local API.

The API binds to 127.0.0.1 only; the key (API_KEY in .env) keeps other local programs and browser
pages from using it. Keys are compared in constant time. A missing server-side key refuses every
protected call (503) instead of letting everything in.
"""

import hashlib
import secrets
import threading
import time
from collections import deque

from fastapi import HTTPException, Request

from rflens.config import ApiSettings

WINDOW_S = 60.0


class RateLimiter:
    """At most `limit` requests per key in any sliding 60 s window."""

    def __init__(self, limit: int, clock=time.monotonic):
        self.limit = limit
        self.clock = clock
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> float | None:
        """None if the request may go on, else the seconds until it could."""
        digest = hashlib.sha256(key.encode()).hexdigest()
        now = self.clock()
        with self._lock:
            hits = self._hits.setdefault(digest, deque())
            while hits and now - hits[0] >= WINDOW_S:
                hits.popleft()
            if len(hits) >= self.limit:
                return max(WINDOW_S - (now - hits[0]), 0.0)
            hits.append(now)
        return None


def make_guard(settings: ApiSettings, limiter: RateLimiter):
    """The FastAPI dependency for every protected route."""

    def guard(request: Request) -> None:
        expected = settings.api_key.get_secret_value() if settings.api_key else ""
        if not expected:
            raise HTTPException(503, "API_KEY is not configured on the server")
        given = request.headers.get("x-api-key", "")
        if not given or not secrets.compare_digest(given.encode(), expected.encode()):
            raise HTTPException(
                401, "missing or wrong X-API-Key", headers={"WWW-Authenticate": "API-Key"}
            )
        wait = limiter.check(given)
        if wait is not None:
            raise HTTPException(
                429, "rate limit exceeded", headers={"Retry-After": str(int(wait) + 1)}
            )

    return guard
