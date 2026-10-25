"""Graph state: JSON types only (dict, list, str, int, float, bool, None).

Checkpoints are serialized with msgpack in strict mode, which refuses user classes, and tuples
come back as lists, so nothing else may be stored.
"""

from typing import Annotated, Any, TypedDict

STATUSES = ("answered", "partial", "refused", "failed", "cancelled")


def merge_stats(old: dict | None, new: dict | None) -> dict:
    """node_stats reducer: calls and cost add up per node name (a retried writer adds to itself)."""
    merged = {k: dict(v) for k, v in (old or {}).items()}
    for node, stat in (new or {}).items():
        have = merged.setdefault(node, {"calls": 0, "cost_usd": 0.0, "runs": 0})
        have["calls"] += stat.get("calls", 0)
        have["cost_usd"] = round(have["cost_usd"] + stat.get("cost_usd", 0.0), 8)
        have["runs"] += 1
    return merged


class GraphState(TypedDict, total=False):
    question_id: str
    question: str
    # planner
    route: str  # sql | doc | mixed | refuse
    plan: dict[str, Any]
    spec: str | None
    spec_source: str  # rule | llm-hint | human | ambiguous | none
    # approval
    approval: dict[str, Any]  # {needed, reasons, mode, decision}
    # evidence
    chunks: list[dict[str, Any]]  # [{ref, chunk_id, spec, version, clause, ..., text}]
    sql: dict[str, Any] | None
    doc_hits: list[dict[str, Any]]  # every search result (or looked-up chunk): rank, id, clause
    band_lookup: dict[str, Any] | None  # {query_id, requested, found, missing} of the band lookup
    # writer / verifier
    draft: str
    sql_retries: int  # SQL-side verification failures answered by a second SQL attempt
    sql_feedback: list[dict[str, Any]]  # the failures the SQL agent is to fix (else empty)
    writer_raw: str  # the writer's last raw output (cut to WRITER_RAW_CHARS)
    writer_failure: str  # '' or the reason the last writer call gave no usable answer
    draft_history: list[dict[str, Any]]  # every writer draft with its verification failures
    writer_attempts: int
    verification: dict[str, Any]
    # outcome
    status: str
    answer: str
    sources: str
    halt: str | None
    node_stats: Annotated[dict[str, dict[str, Any]], merge_stats]
