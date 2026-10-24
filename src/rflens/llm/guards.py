"""Loop protection: per-run and per-question call caps, repeat detectors, and a rate limiter.

A "call" for the caps is one attempt that reaches a provider, so retries and the fallback attempt
count. The repeat detectors work on logical calls (one `generate()`), not on attempts, because a
retry necessarily repeats its prompt and its error.
"""

import threading
import time
from collections import Counter, deque
from collections.abc import Callable

from rflens.llm.base import CallCapExceeded, LoopDetected

REPEAT_LIMIT = 3


class RunGuard:
    """Counters for one run (one command). Not shared between runs."""

    def __init__(
        self,
        run_id: str,
        max_calls_per_run: int,
        max_calls_per_question: int,
    ):
        self.run_id = run_id
        self.max_calls_per_run = max_calls_per_run
        self.max_calls_per_question = max_calls_per_question
        self.attempts = 0
        self.per_question: Counter[str] = Counter()
        self.prompt_counts: Counter[str] = Counter()
        self._last_error: str | None = None
        self._error_streak = 0

    def count_attempt(self, question_id: str | None) -> None:
        """Count one attempt against both caps, or raise before it is made."""
        if self.attempts >= self.max_calls_per_run:
            raise CallCapExceeded(
                "run",
                self.run_id,
                f"run {self.run_id} stopped: it already made {self.attempts} LLM calls "
                f"(LLM_MAX_CALLS_PER_RUN={self.max_calls_per_run}). Retries and fallbacks count.",
            )
        if question_id is not None and (
            self.per_question[question_id] >= self.max_calls_per_question
        ):
            raise CallCapExceeded(
                "question",
                self.run_id,
                f"run {self.run_id} stopped: question {question_id!r} already used "
                f"{self.per_question[question_id]} LLM calls "
                f"(LLM_MAX_CALLS_PER_QUESTION={self.max_calls_per_question}). "
                "Retries and corrections count.",
            )
        self.attempts += 1
        if question_id is not None:
            self.per_question[question_id] += 1

    def note_prompt(self, prompt_hash: str) -> None:
        """Stop the run when the same prompt is sent for the third time."""
        self.prompt_counts[prompt_hash] += 1
        if self.prompt_counts[prompt_hash] >= REPEAT_LIMIT:
            raise LoopDetected(
                "repeat_prompt",
                self.run_id,
                f"run {self.run_id} stopped: the same prompt (hash {prompt_hash[:12]}) was sent "
                f"{REPEAT_LIMIT} times",
            )

    def note_success(self) -> None:
        self._last_error = None
        self._error_streak = 0

    def note_failure(self, signature: str) -> None:
        """Stop the run when a call fails with the same error three calls in a row."""
        if signature == self._last_error:
            self._error_streak += 1
        else:
            self._last_error, self._error_streak = signature, 1
        if self._error_streak >= REPEAT_LIMIT:
            raise LoopDetected(
                "repeat_error",
                self.run_id,
                f"run {self.run_id} stopped: the same error occurred {REPEAT_LIMIT} calls in a "
                f"row ({signature})",
            )


class RateLimiter:
    """At most `max_per_minute` attempts in any 60 seconds, shared by every run in the process.
    `acquire` sleeps until a slot is free."""

    def __init__(
        self,
        max_per_minute: int,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.max_per_minute = max_per_minute
        self._clock = clock
        self._sleep = sleep
        self._stamps: deque[float] = deque()
        self._lock = threading.Lock()
        self.waited_s = 0.0

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = self._clock()
                while self._stamps and now - self._stamps[0] >= 60:
                    self._stamps.popleft()
                if len(self._stamps) < self.max_per_minute:
                    self._stamps.append(now)
                    return
                wait = 60 - (now - self._stamps[0])
            self.waited_s += wait
            self._sleep(max(wait, 0.001))
