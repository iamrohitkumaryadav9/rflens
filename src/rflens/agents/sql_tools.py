"""The SQL agent's only door to data: a client for sql-mcp over MCP.

Architecture rule: agents never connect to Postgres. `target` is the URL of a running sql-mcp
(Streamable HTTP) or, in tests, an in-process MCPServer; both are spoken to with the official MCP
client. Calls are synchronous (anyio.run per call; sql-mcp is stateless, so there is no session to
keep). The schema context is built once per `SqlMcpTools` object, i.e. once per run.
"""

from dataclasses import dataclass, field
from typing import Any

import anyio
from mcp import Client

SCHEMA_QUESTION = "sql-agent schema context"


class SqlMcpUnavailable(Exception):
    pass


@dataclass(frozen=True)
class QueryOutcome:
    ok: bool
    query_id: int | None = None  # the call's row in sql_query_log
    columns: list[str] = field(default_factory=list)
    rows: list[list[Any]] = field(default_factory=list)
    row_count: int = 0
    truncated: bool = False
    executed_sql: str | None = None
    plan_cost: float | None = None
    error: str | None = None  # the server's message, without the SDK's "Error executing tool"
    error_kind: str | None = None  # 'rejected' | 'failed' | 'other'


def _strip_sdk_prefix(text: str) -> str:
    prefix = "Error executing tool "
    if text.startswith(prefix) and ": " in text:
        return text.split(": ", 1)[1]
    return text


class SqlMcpTools:
    def __init__(self, target: Any):
        self.target = target
        self._schema: str | None = None
        self._described: list[dict] = []

    def _call(self, tool: str, arguments: dict) -> Any:
        async def run():
            async with Client(self.target) as client:
                return await client.call_tool(tool, arguments)

        try:
            return anyio.run(run)
        except Exception as e:  # connection refused, protocol error, ...
            raise SqlMcpUnavailable(
                f"cannot reach sql-mcp at {self.target!r} ({type(e).__name__}: {e}); "
                "start it with `make sql-mcp`"
            ) from e

    def list_tables(self) -> list[dict]:
        result = self._call("list_tables", {"question": SCHEMA_QUESTION})
        if result.is_error:
            raise SqlMcpUnavailable(_strip_sdk_prefix(result.content[0].text))
        return result.structured_content["tables"]

    def describe_table(self, table: str) -> dict:
        result = self._call("describe_table", {"table": table, "question": SCHEMA_QUESTION})
        if result.is_error:
            raise SqlMcpUnavailable(_strip_sdk_prefix(result.content[0].text))
        return result.structured_content

    def schema_context(self) -> str:
        """Tables, views, columns and their comments (units, caveats), cached for this object."""
        if self._schema is None:
            blocks = []
            for table in self.list_tables():
                d = self.describe_table(table["name"])
                self._described.append(d)
                lines = [f"== {d['name']} ({d['kind']}) ==", d["description"] or ""]
                lines += [
                    f"  {c['name']} {c['type']}"
                    + (f" -- {c['description']}" if c["description"] else "")
                    for c in d["columns"]
                ]
                blocks.append("\n".join(lines))
            self._schema = "\n\n".join(blocks)
        return self._schema

    def column_notes(self) -> dict[str, dict[str, bool]]:
        """Per column name: whether its comment says the unit is UNKNOWN or ASSUMED (the schema
        comments are the single source of unit caveats). Names shared by several relations merge."""
        self.schema_context()
        notes: dict[str, dict[str, bool]] = {}
        for relation in self._described:
            for column in relation["columns"]:
                text = column["description"] or ""
                entry = notes.setdefault(column["name"], {"unknown": False, "assumed": False})
                entry["unknown"] = entry["unknown"] or "UNKNOWN" in text
                entry["assumed"] = entry["assumed"] or "ASSUMED" in text
        return notes

    def column_comments(self) -> dict[str, str]:
        """Per column name, the schema comments joined (names shared by relations merge)."""
        self.schema_context()
        comments: dict[str, str] = {}
        for relation in self._described:
            for column in relation["columns"]:
                text = column["description"] or ""
                if text and text not in comments.get(column["name"], ""):
                    comments[column["name"]] = (
                        comments.get(column["name"], "") + " " + text
                    ).strip()
        return comments

    def run_query(self, sql: str, question: str) -> QueryOutcome:
        """Run one SELECT through sql-mcp; `question` lands in sql_query_log next to the SQL."""
        result = self._call("run_query", {"sql": sql, "question": question})
        if result.is_error:
            message = _strip_sdk_prefix(result.content[0].text)
            kind = (
                "rejected"
                if message.startswith("rejected (")
                else "failed"
                if message.startswith("query failed (")
                else "other"
            )
            return QueryOutcome(ok=False, error=message, error_kind=kind)
        d = result.structured_content
        return QueryOutcome(
            ok=True,
            query_id=d.get("query_id"),
            columns=d["columns"],
            rows=d["rows"],
            row_count=d["row_count"],
            truncated=d["truncated"],
            executed_sql=d["executed_sql"],
            plan_cost=d["plan_cost"],
        )
