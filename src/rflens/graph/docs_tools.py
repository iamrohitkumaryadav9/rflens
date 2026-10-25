"""The doc agent's only door to the specifications: a client for docs-mcp over MCP.

`target` is docs-mcp's URL or, in tests, an in-process server. Same shape as `SqlMcpTools`.
"""

from typing import Any

import anyio
from mcp import Client


class DocsMcpUnavailable(Exception):
    pass


class DocsMcpError(Exception):
    """docs-mcp answered with a tool error (for example: the embedding service is down)."""


class DocsMcpTools:
    def __init__(self, target: Any):
        self.target = target
        self.calls = 0

    def _call(self, tool: str, arguments: dict) -> dict:
        async def run():
            async with Client(self.target) as client:
                return await client.call_tool(tool, arguments)

        try:
            result = anyio.run(run)
        except Exception as e:
            raise DocsMcpUnavailable(
                f"cannot reach docs-mcp at {self.target!r} ({type(e).__name__}: {e}); "
                "start it with `make docs-mcp`"
            ) from e
        self.calls += 1
        if result.is_error:
            raise DocsMcpError(result.content[0].text)
        return result.structured_content

    def search(
        self, query: str, spec: str | None, mode: str, k: int, context: str | None = None
    ) -> list[dict]:
        arguments: dict[str, Any] = {"query": query, "mode": mode, "k": k}
        if spec:
            arguments["spec"] = spec
        if context:
            arguments["context"] = context
        return self._call("search_docs", arguments)["results"]

    def get_chunk(self, chunk_id: str) -> dict:
        return self._call("get_chunk", {"chunk_id": chunk_id, "include_related": False})
