"""Import-time guards of the orchestrator: no tracing leaves the machine."""

import os
from collections.abc import Mapping

TRACING_VARS = (
    "LANGSMITH_TRACING",
    "LANGSMITH_TRACING_V2",
    "LANGCHAIN_TRACING_V2",
    "LANGCHAIN_TRACING",
)
TRUTHY = {"1", "true", "yes", "on"}


class TracingEnabledError(RuntimeError):
    pass


def assert_tracing_off(env: Mapping[str, str] | None = None) -> None:
    """Raise if any LangSmith / LangChain tracing switch is on. Questions, SQL results and
    specification text must not be sent to a tracing service."""
    env = os.environ if env is None else env
    on = [name for name in TRACING_VARS if env.get(name, "").strip().lower() in TRUTHY]
    if on:
        raise TracingEnabledError(
            f"tracing is switched on ({', '.join(on)}); the orchestrator refuses to run: "
            "set it to false or unset it"
        )
