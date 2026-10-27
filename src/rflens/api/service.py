"""The orchestrator behind the HTTP API.

Every call opens what it needs and closes it again: a gateway run (its own run id and ledger
summary), a checkpointer connection as rflens_checkpointer, and the compiled graph. Nothing stays
open between requests, so a request that dies leaves no half-open state, and a thread is always
resumable from its last checkpoint. Data is reached only through the MCP servers (inside the
graph) and through docs-mcp's `get_chunk` for the UI's source panels.
"""

import re
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, text

from rflens.agents.analytics_tools import AnalyticsMcpTools
from rflens.agents.sql_tools import SqlMcpTools
from rflens.analytics.sandbox import artifacts_dir
from rflens.config import (
    GraphSettings,
    get_agent_settings,
    get_artifact_settings,
    get_checkpointer_settings,
    get_graph_settings,
)
from rflens.db import get_ledger_engine
from rflens.graph.app import (
    build_graph,
    continue_thread,
    open_checkpointer,
    start_thread,
    thread_config,
)
from rflens.graph.docs_tools import DocsMcpError, DocsMcpTools
from rflens.graph.nodes import GraphDeps
from rflens.graph.view import thread_view
from rflens.llm.gateway import Gateway, build_gateway
from rflens.mcp.docs_server import CHUNK_ID_RE

THREAD_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
ARTIFACT_ID_RE = re.compile(r"^[0-9a-f]{64}$")  # an analysis_id: never a path
NODE_OF_PURPOSE = {
    "planner": "planner",
    "sql-agent": "sql_agent",
    "stats-agent": "stats_agent",
    "writer": "writer",
}


class ThreadNotFound(LookupError):
    pass


class ThreadExists(ValueError):
    pass


class NotWaiting(ValueError):
    """The thread is finished (or otherwise not waiting for a decision)."""


class BadChunkId(ValueError):
    pass


class ChunkNotFound(LookupError):
    pass


class BadArtifactId(ValueError):
    pass


class ArtifactNotFound(LookupError):
    pass


@dataclass
class Parts:
    """What one request needs besides the checkpointer."""

    gateway: Gateway
    sql_tools: SqlMcpTools
    docs_tools: DocsMcpTools
    settings: GraphSettings
    analytics_tools: AnalyticsMcpTools | None = None


_DEFAULT_TOOLS: dict[str, Any] = {}


def default_parts() -> Parts:
    """The real thing: the configured provider, and the MCP servers of AGENT_* settings. The MCP
    clients are shared between requests, so the SQL schema context is built once per process."""
    agent = get_agent_settings()
    if not _DEFAULT_TOOLS:
        _DEFAULT_TOOLS["sql"] = SqlMcpTools(agent.sql_mcp_url)
        _DEFAULT_TOOLS["docs"] = DocsMcpTools(agent.docs_mcp_url)
        _DEFAULT_TOOLS["analytics"] = AnalyticsMcpTools(agent.analytics_mcp_url)
    return Parts(
        build_gateway(),
        _DEFAULT_TOOLS["sql"],
        _DEFAULT_TOOLS["docs"],
        get_graph_settings(),
        _DEFAULT_TOOLS["analytics"],
    )


class OrchestratorService:
    def __init__(self, parts: Callable[[], Parts] = default_parts):
        self._parts = parts

    @contextmanager
    def _graph(self, run_name: str, *, ask: bool = False, track: bool = True) -> Iterator[Any]:
        parts = self._parts()
        settings = parts.settings.model_copy(update={"ask": True}) if ask else parts.settings
        gateway_run = parts.gateway.run(run_name)
        with open_checkpointer() as saver:
            deps = GraphDeps(
                gateway_run,
                parts.sql_tools,
                parts.docs_tools,
                settings,
                approval_mode="interrupt",
                analytics_tools=parts.analytics_tools,
            )
            graph = build_graph(deps, saver)
            if track:
                with gateway_run:  # prints the run summary at the end, like every command
                    yield graph
            else:
                yield graph  # read-only calls: no run summary

    # --- reading ------------------------------------------------------------------------------

    @staticmethod
    def _snapshot_view(graph: Any, thread_id: str, *, drafts: bool = True) -> dict[str, Any]:
        snapshot = graph.get_state(thread_config(thread_id))
        if not snapshot.values:
            raise ThreadNotFound(thread_id)
        pending = next((i.value for t in snapshot.tasks for i in t.interrupts), None)
        return thread_view(thread_id, dict(snapshot.values), pending, snapshot.next, drafts)

    def view(self, thread_id: str) -> dict[str, Any]:
        with self._graph("api-read", track=False) as graph:
            view = self._snapshot_view(graph, thread_id)
        view["runs"] = self.ledger_runs(thread_id)
        return view

    def recent_threads(self, limit: int = 20) -> list[dict[str, Any]]:
        """The latest threads, newest first, each as a short summary."""
        engine = create_engine(get_checkpointer_settings().sa_url())
        try:
            with engine.connect() as c:
                ids = [
                    r[0]
                    for r in c.execute(
                        text(
                            "SELECT thread_id FROM checkpoints GROUP BY thread_id "
                            "ORDER BY max(checkpoint_id) DESC LIMIT :n"
                        ),
                        {"n": limit},
                    )
                ]
        finally:
            engine.dispose()
        out = []
        with self._graph("api-read", track=False) as graph:
            for thread_id in ids:
                try:
                    v = self._snapshot_view(graph, thread_id, drafts=False)
                except ThreadNotFound:
                    continue
                out.append(
                    {
                        "thread_id": thread_id,
                        "question": v["question"],
                        "status": v["status"],
                        "badge": v["badge"],
                        "route": v["route"],
                        "cost_usd": v["cost"]["usd"],
                    }
                )
        return out

    @staticmethod
    def ledger_runs(thread_id: str) -> list[dict[str, Any]]:
        """Run ids that made calls for this question id, with calls and cost per node."""
        with get_ledger_engine().connect() as c:
            rows = c.execute(
                text(
                    "SELECT run_id, purpose, count(*) AS calls, "
                    "coalesce(sum(cost_usd), 0) AS cost FROM llm_calls "
                    "WHERE question_id = :q AND status IN ('pending', 'ok', 'error') "
                    "GROUP BY run_id, purpose ORDER BY min(created_at), purpose"
                ),
                {"q": thread_id},
            ).all()
        runs: dict[str, dict[str, Any]] = {}
        for r in rows:
            run = runs.setdefault(r.run_id, {"run_id": r.run_id, "nodes": {}})
            run["nodes"][NODE_OF_PURPOSE.get(r.purpose, r.purpose)] = {
                "calls": r.calls,
                "cost_usd": float(r.cost),
            }
        return list(runs.values())

    def chunk(self, chunk_id: str) -> dict[str, Any]:
        """One spec chunk through docs-mcp (for the source panel)."""
        if not CHUNK_ID_RE.fullmatch(chunk_id):
            raise BadChunkId(chunk_id)
        try:
            full = self._parts().docs_tools.get_chunk(chunk_id)
        except DocsMcpError as e:
            if "unknown chunk_id" in str(e):
                raise ChunkNotFound(chunk_id) from e
            raise
        return {
            k: full[k]
            for k in (
                "chunk_id",
                "spec",
                "version",
                "kind",
                "clause",
                "clause_title",
                "table_number",
                "table_part",
                "text",
            )
        }

    @staticmethod
    def artifact_path(artifact_id: str) -> Path:
        """The PNG of an analysis artifact, by its id (64 hex characters)."""
        if not ARTIFACT_ID_RE.fullmatch(artifact_id):
            raise BadArtifactId(artifact_id)
        path = artifacts_dir(get_artifact_settings()) / f"{artifact_id}.png"
        if not path.is_file():
            raise ArtifactNotFound(artifact_id)
        return path

    # --- running ------------------------------------------------------------------------------

    def ask(self, question: str, *, ask: bool = False, thread_id: str | None = None) -> dict:
        thread_id = thread_id or f"api-{uuid.uuid4().hex[:12]}"
        with self._graph("api-ask", ask=ask) as graph:
            if graph.get_state(thread_config(thread_id)).values:
                raise ThreadExists(thread_id)
            start_thread(graph, question, thread_id)
            view = self._snapshot_view(graph, thread_id)
        view["runs"] = self.ledger_runs(thread_id)
        return view

    def resume(self, thread_id: str, decision: dict[str, Any]) -> dict:
        with self._graph("api-resume") as graph:
            snapshot = graph.get_state(thread_config(thread_id))
            if not snapshot.values:
                raise ThreadNotFound(thread_id)
            if not snapshot.next:
                raise NotWaiting(thread_id)
            continue_thread(graph, thread_id, decision)
            view = self._snapshot_view(graph, thread_id)
        view["runs"] = self.ledger_runs(thread_id)
        return view
