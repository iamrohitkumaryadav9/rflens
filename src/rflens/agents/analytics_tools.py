"""The stats agent's only door to the analyses: a client for analytics-mcp over MCP.

`target` is analytics-mcp's URL or, in tests, an in-process MCPServer; both are spoken to with the
official MCP client. Same shape as `SqlMcpTools` / `DocsMcpTools`: synchronous, one connection per
call (analytics-mcp is stateless).
"""

from dataclasses import dataclass, field
from typing import Any

import anyio
from mcp import Client


class AnalyticsMcpUnavailable(Exception):
    pass


@dataclass(frozen=True)
class AnalysisOutcome:
    ok: bool
    response: dict[str, Any] = field(default_factory=dict)  # the server's AnalysisResponse
    error: str | None = None  # the server's message, without the SDK's "Error executing tool"
    error_kind: str | None = None  # 'rejected' | 'failed' | 'other'


def _strip_sdk_prefix(text: str) -> str:
    prefix = "Error executing tool "
    if text.startswith(prefix) and ": " in text:
        return text.split(": ", 1)[1]
    return text


class AnalyticsMcpTools:
    def __init__(self, target: Any):
        self.target = target
        self._menu: dict[str, Any] | None = None

    def _call(self, tool: str, arguments: dict) -> Any:
        async def run():
            async with Client(self.target) as client:
                return await client.call_tool(tool, arguments)

        try:
            return anyio.run(run)
        except Exception as e:
            raise AnalyticsMcpUnavailable(
                f"cannot reach analytics-mcp at {self.target!r} ({type(e).__name__}: {e}); "
                "start it with `make analytics-mcp`"
            ) from e

    def menu(self) -> dict[str, Any]:
        """The dataset and the analyses with their parameter schemas, cached for this object."""
        if self._menu is None:
            result = self._call("list_analyses", {"question": "stats-agent menu"})
            if result.is_error:
                raise AnalyticsMcpUnavailable(_strip_sdk_prefix(result.content[0].text))
            self._menu = result.structured_content
        return self._menu

    def run_analysis(self, analysis: str, params: dict[str, Any], question: str) -> AnalysisOutcome:
        """Run one analysis; `question` lands in analytics_call_log next to the call."""
        result = self._call(analysis, {"params": params, "question": question})
        if result.is_error:
            message = _strip_sdk_prefix(result.content[0].text)
            kind = (
                "rejected"
                if message.startswith("rejected (")
                else "failed"
                if message.startswith("analysis failed (")
                else "other"
            )
            return AnalysisOutcome(ok=False, error=message, error_kind=kind)
        return AnalysisOutcome(ok=True, response=result.structured_content)
