"""Graph assembly, the checkpointer and the entry points `ask` / `resume`.

    with open_checkpointer() as saver:
        graph = build_graph(deps, saver)
        out = ask(graph, "question text", question_id="q-001")
        if out.interrupt:                      # waiting for a human
            out = resume(graph, "q-001", {"approve": True, "spec": "36.101"})

The thread id is the question id; checkpoints are written synchronously (durability="sync"), so a
crash loses at most the node that was running.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from rflens.config import get_checkpointer_settings
from rflens.graph import nodes
from rflens.graph.state import GraphState

DURABILITY = "sync"


class CheckpointSchemaError(RuntimeError):
    pass


def check_checkpoint_schema(saver: PostgresSaver) -> None:
    """The library's tables are created by migration 0012, not at runtime (the runtime role has
    DML rights only). Refuse to start when the schema is missing or older than the installed
    library expects."""
    expected = len(PostgresSaver.MIGRATIONS) - 1
    try:
        with saver._cursor() as cur:  # the library's own connection handling
            cur.execute("SELECT max(v) AS v FROM checkpoint_migrations")
            row = cur.fetchone()
    except Exception as e:
        raise CheckpointSchemaError(
            f"cannot read checkpoint_migrations ({type(e).__name__}: {e}); run `make migrate`"
        ) from e
    have = row["v"] if row and row["v"] is not None else -1
    if have != expected:
        raise CheckpointSchemaError(
            f"checkpoint schema is at version {have}, langgraph-checkpoint-postgres expects "
            f"{expected}; run `make migrate` (migration 0012 or a newer one)"
        )


@contextmanager
def open_checkpointer() -> Iterator[PostgresSaver]:
    """A PostgresSaver connected as rflens_checkpointer, schema verified, never `setup()`."""
    with PostgresSaver.from_conn_string(get_checkpointer_settings().conninfo()) as saver:
        check_checkpoint_schema(saver)
        yield saver


def _after_planner(state: GraphState) -> str:
    if state.get("halt") or state.get("route") == "refuse":
        return "finalize"
    return "approval"


def _after_approval(state: GraphState) -> str:
    if state.get("status") == "cancelled":
        return "finalize"
    return "sql_agent" if state["route"] == "sql" else "doc_agent"


def _after_doc(state: GraphState) -> str:
    if state.get("halt"):
        return "finalize"
    return "sql_agent" if state["route"] == "mixed" else "writer"


def _after_sql(state: GraphState) -> str:
    sql = state.get("sql") or {}
    if state.get("halt") or sql.get("status") != "answered":
        return "finalize"
    return "writer"


def _after_writer(state: GraphState) -> str:
    return "finalize" if state.get("halt") else "verifier"


def build_graph(deps: nodes.GraphDeps, checkpointer: Any):
    g = StateGraph(GraphState)
    g.add_node("planner", nodes.make_planner(deps))
    g.add_node("approval", nodes.make_approval(deps))
    g.add_node("doc_agent", nodes.make_doc_agent(deps))
    g.add_node("sql_agent", nodes.make_sql_agent(deps))
    g.add_node("writer", nodes.make_writer(deps))
    g.add_node("verifier", nodes.verifier)
    g.add_node("redact", nodes.redact)
    g.add_node("finalize", nodes.finalize)

    max_retries = deps.settings.orch_max_writer_retries

    def after_verifier(state: GraphState) -> str:
        if state["verification"]["ok"]:
            return "finalize"
        return "writer" if state.get("writer_attempts", 0) <= max_retries else "redact"

    g.add_edge(START, "planner")
    g.add_conditional_edges("planner", _after_planner, ["approval", "finalize"])
    g.add_conditional_edges("approval", _after_approval, ["doc_agent", "sql_agent", "finalize"])
    g.add_conditional_edges("doc_agent", _after_doc, ["sql_agent", "writer", "finalize"])
    g.add_conditional_edges("sql_agent", _after_sql, ["writer", "finalize"])
    g.add_conditional_edges("writer", _after_writer, ["verifier", "finalize"])
    g.add_conditional_edges("verifier", after_verifier, ["writer", "redact", "finalize"])
    g.add_edge("redact", "finalize")
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer)


@dataclass
class Outcome:
    question_id: str
    state: dict[str, Any]
    interrupt: dict[str, Any] | None = None  # the approval payload while the graph waits
    stats: dict[str, dict] = field(default_factory=dict)

    @property
    def status(self) -> str | None:
        return None if self.interrupt else self.state.get("status")


def _outcome(graph, question_id: str, result: dict) -> Outcome:
    pending = result.get("__interrupt__")
    snapshot = graph.get_state({"configurable": {"thread_id": question_id}})
    state = dict(snapshot.values)
    return Outcome(
        question_id=question_id,
        state=state,
        interrupt=pending[0].value if pending else None,
        stats=state.get("node_stats", {}),
    )


def ask(graph, question: str, question_id: str) -> Outcome:
    config = {"configurable": {"thread_id": question_id}}
    result = graph.invoke(
        {"question_id": question_id, "question": question}, config, durability=DURABILITY
    )
    return _outcome(graph, question_id, result)


def resume(graph, question_id: str, decision: Any) -> Outcome:
    config = {"configurable": {"thread_id": question_id}}
    result = graph.invoke(Command(resume=decision), config, durability=DURABILITY)
    return _outcome(graph, question_id, result)


def final_text(outcome: Outcome) -> str:
    """The answer with its deterministic Sources section."""
    state = outcome.state
    text = state.get("answer", "")
    if state.get("sources"):
        text += "\n\nSources:\n" + state["sources"]
    return text
