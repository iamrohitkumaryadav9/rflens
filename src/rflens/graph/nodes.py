"""The graph's nodes. Each is a plain function of the state returning a partial state update.

`GraphDeps` carries everything a node may touch: the gateway run (all LLM calls), the two MCP
clients (all data) and the settings. A node that resumes after a crash starts from its top, so no
node does anything irreversible before it returns, and the approval node calls `interrupt` first.

Stops that belong to one question (its call cap, a failed model call, an unreachable MCP server)
end the question as `failed` through `halt`. Stops that belong to the whole command (run cap,
budget, loop detector, paid-call block, ledger failure) propagate out of the graph: the thread
keeps its last checkpoint and can be resumed.
"""

import json
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Literal

from langgraph.types import interrupt

from rflens.agents.analytics_tools import AnalyticsMcpTools, AnalyticsMcpUnavailable
from rflens.agents.sql_agent import MAX_CORRECTIONS, answer_question
from rflens.agents.sql_prompt import build_system_prompt
from rflens.agents.sql_tools import SqlMcpTools, SqlMcpUnavailable
from rflens.agents.stats_agent import MAX_CORRECTIONS as STATS_MAX_CORRECTIONS
from rflens.agents.stats_agent import MAX_OUTPUT_TOKENS as STATS_MAX_OUTPUT_TOKENS
from rflens.agents.stats_agent import answer_stats
from rflens.agents.stats_prompt import STATS_SYSTEM
from rflens.config import GraphSettings, get_agent_settings
from rflens.graph.band_lookup import BandLookupError, clean_bands, describe_row, lookup_bands
from rflens.graph.docs_tools import DocsMcpError, DocsMcpTools, DocsMcpUnavailable
from rflens.graph.prompts import (
    PLANNER_SCHEMA,
    PLANNER_SYSTEM,
    WRITER_SCHEMA,
    WRITER_SYSTEM,
    mixed_sql_context,
    writer_user_message,
)
from rflens.graph.spec_rules import SPECS, infer_spec, merge_with_hint
from rflens.graph.state import GraphState
from rflens.graph.verifier import CITE_RE, split_sentences, units_in, verify
from rflens.llm.base import (
    BadStructuredOutput,
    CallCapExceeded,
    LoopDetected,
    Message,
    ProviderError,
    estimate_tokens,
)
from rflens.llm.gateway import LLMRun
from rflens.retrieval.variants import extract_refs

ROUTES = ("sql", "doc", "mixed", "stats", "refuse")
WRITER_EST_OVERHEAD_TOKENS = 700
CAP_NOTE = "per-question call cap"
# failures that come from the SQL itself: the SQL result is withheld when one is left after retries
SQL_FAILURES = ("unsupported_sql_literal", "edge_side_mismatch", "frn_name_grouping")
MAX_SQL_RETRIES = 1  # a SQL-side failure sends the question back to the SQL agent once


@dataclass
class GraphDeps:
    run: LLMRun
    sql_tools: SqlMcpTools
    docs_tools: DocsMcpTools
    settings: GraphSettings
    approval_mode: Literal["interrupt", "auto"] = "interrupt"
    analytics_tools: AnalyticsMcpTools | None = None  # only the stats route needs it

    def __post_init__(self) -> None:
        # one graph run is one question: the orchestrated cap replaces the per-command default
        self.run.guard.max_calls_per_question = self.settings.orch_max_calls_per_question


@contextmanager
def _measured(deps: GraphDeps, question_id: str):
    """Yields a dict that ends up as {calls, cost_usd} of the block, from the gateway's counters."""
    stat = {"calls": 0, "cost_usd": 0.0}
    calls0 = deps.run.guard.per_question[question_id]
    cost0 = deps.run.cost_usd
    try:
        yield stat
    finally:
        stat["calls"] = deps.run.guard.per_question[question_id] - calls0
        stat["cost_usd"] = deps.run.cost_usd - cost0


def _halt_cap(e: CallCapExceeded) -> dict:
    if e.scope != "question":
        raise e
    return {"halt": f"stopped by the {CAP_NOTE}: {e}"}


# --- planner ------------------------------------------------------------------------------------


def planned_worst_case_usd(deps: GraphDeps, route: str, question: str) -> float:
    """Worst-case cost of the rest of the run for `route`: every SQL generation and writer call
    the retry rules allow, each priced at its input estimate and full output cap."""
    gw = deps.run.gateway
    prov = gw.providers[gw.settings.llm_provider]
    if not prov.is_paid or route == "refuse":
        return 0.0
    s = deps.settings

    def one(est_in: int, max_out: int) -> float:
        return gw.pricing.worst_case_usd(prov.name, prov.model, prov.service_tier, est_in, max_out)

    total = 0.0
    if route == "stats":
        est = estimate_tokens([Message.system(STATS_SYSTEM), Message.user(question)])
        total += (1 + STATS_MAX_CORRECTIONS) * one(est, STATS_MAX_OUTPUT_TOKENS)
    if route in ("sql", "mixed"):
        prompt = [
            Message.system(build_system_prompt(deps.sql_tools.schema_context())),
            Message.user(question),
        ]
        est = estimate_tokens(prompt) + (s.orch_evidence_chars // 3 if route == "mixed" else 0)
        total += (1 + MAX_CORRECTIONS) * one(est, get_agent_settings().agent_max_output_tokens)
    est_writer = s.orch_evidence_chars // 3 + WRITER_EST_OVERHEAD_TOKENS
    total += (1 + s.orch_max_writer_retries) * one(est_writer, s.orch_writer_max_tokens)
    return total


def _loop_halt(state: GraphState, e: LoopDetected, node: str) -> dict:
    """A repeated prompt or error within one question fails that question (failure type
    loop_detected) and nothing else: the run goes on with the next question."""
    failure = {"type": "loop_detected", "sentence": None, "detail": f"{node}: {e}"}
    history = state.get("draft_history") or []
    entry = {
        "attempt": len(history) + 1,
        "draft": "",
        "raw": "",
        "ok": False,
        "failures": [failure],
    }
    return {
        "halt": f"loop_detected in {node}: {e}",
        "verification": {
            "ok": False,
            "failures": [failure],
            "evidence_map": [],
            "band_edges": [],
            "band_joins": [],
        },
        "draft_history": [*history, entry],
    }


def make_planner(deps: GraphDeps):
    def planner(state: GraphState) -> dict:
        qid, question = state["question_id"], state["question"]
        deps.run.seed_question(qid)
        with _measured(deps, qid) as stat:
            try:
                generated = deps.run.generate(
                    [Message.system(PLANNER_SYSTEM), Message.user(question)],
                    purpose="planner",
                    json_schema=PLANNER_SCHEMA,
                    question_id=qid,
                    max_output_tokens=deps.settings.orch_planner_max_tokens,
                )
            except CallCapExceeded as e:
                return {"halt": _halt_cap(e)["halt"], "node_stats": {"planner": stat}}
            except LoopDetected as e:
                return {**_loop_halt(state, e, "planner"), "node_stats": {"planner": stat}}
            except (ProviderError, BadStructuredOutput) as e:
                return {"halt": f"the planner call failed: {e}", "node_stats": {"planner": stat}}
        plan = generated.parsed if isinstance(generated.parsed, dict) else {}
        route = plan.get("route")
        if route not in ROUTES:
            return {
                "halt": f"the planner returned an unusable route {route!r}",
                "node_stats": {"planner": stat},
            }
        hint = plan.get("spec") if plan.get("spec") in SPECS else None
        rules = infer_spec(question)
        spec, source, conflict = merge_with_hint(rules, hint)
        doc_query = str(plan.get("doc_query") or "").strip()
        sql_question = str(plan.get("sql_question") or "").strip()
        stats_question = str(plan.get("stats_question") or "").strip()
        reasons: list[str] = []
        if route != "refuse" and deps.settings.ask:
            reasons.append("ask: ASK is set")  # (a)
        planned = planned_worst_case_usd(deps, route, question)
        if planned > deps.settings.approval_cost_usd:  # (b)
            reasons.append(
                f"cost: planned worst case ${planned:.4f} exceeds "
                f"${deps.settings.approval_cost_usd}"
            )
        if planned > 0:  # (c) only a paid provider has a worst case
            remaining = deps.run.gateway.settings.llm_budget_usd - (
                deps.run.gateway.ledger.total_usd()
            )
            if remaining < 2 * planned:
                reasons.append(
                    f"budget: ${remaining:.4f} remains, under twice the planned worst case "
                    f"${planned:.4f}"
                )
        if route in ("doc", "mixed"):
            if conflict:  # (d)
                reasons.append(
                    f"spec_conflict: the wording says {spec}, the planner suggests {hint}"
                )
            elif rules.source == "ambiguous":  # (e)
                spec, source = None, "ambiguous"
                reasons.append("spec_ambiguous: the question mentions both LTE and NR wording")
        return {
            "route": route,
            "plan": {
                "route": route,
                "spec_hint": hint,
                "doc_query": doc_query,
                "sql_question": sql_question,
                "stats_question": stats_question,
                "bands": clean_bands(plan.get("bands")) if route == "mixed" else [],
                "refuse_reason": str(plan.get("refuse_reason") or ""),
                "rule_matches": list(rules.matched),
                "planned_worst_case_usd": round(planned, 6),
            },
            "spec": spec if route in ("doc", "mixed") else None,
            "spec_source": source,
            "approval": {"needed": bool(reasons), "reasons": reasons},
            "node_stats": {"planner": stat},
        }

    return planner


# --- approval -----------------------------------------------------------------------------------


def make_approval(deps: GraphDeps):
    def approval(state: GraphState) -> dict:
        ap = dict(state.get("approval") or {})
        if not ap.get("needed"):
            return {}
        payload = {
            "question_id": state["question_id"],
            "question": state["question"],
            "route": state["route"],
            "reasons": ap["reasons"],
            "proposed_spec": state.get("spec"),
            "spec_hint": (state.get("plan") or {}).get("spec_hint"),
            "spec_options": [*SPECS, None],
            "planned_worst_case_usd": (state.get("plan") or {}).get("planned_worst_case_usd"),
            "reply_with": {"approve": "bool", "spec": "36.101 | 38.101-1 | null (both)"},
        }
        if deps.approval_mode == "auto":
            decision: Any = {"approve": True, "spec": state.get("spec"), "by": "auto"}
        else:
            decision = interrupt(payload)  # first side effect of this node; re-run on resume
        if not isinstance(decision, dict):
            decision = {"approve": bool(decision)}
        ap["decision"] = {
            "approve": bool(decision.get("approve", False)),
            "by": decision.get("by", "human"),
        }
        if not ap["decision"]["approve"]:
            return {
                "approval": ap,
                "status": "cancelled",
                "answer": "Not run: the approval step was declined.",
            }
        update: dict = {"approval": ap}
        if "spec" in decision and state["route"] in ("doc", "mixed"):
            chosen = decision["spec"] if decision["spec"] in SPECS else None
            update["spec"] = chosen
            if decision.get("by") != "auto":
                update["spec_source"] = "human"
        return update

    return approval


# --- doc agent ----------------------------------------------------------------------------------


def _chunk_entry(full: dict, ref: str, room: int, band_lines: list[str] | None = None) -> dict:
    body = full["text"][:room]
    entry = {
        "ref": ref,
        "chunk_id": full["chunk_id"],
        "spec": full["spec"],
        "version": full["version"],
        "clause": full["clause"],
        "clause_title": full["clause_title"],
        "table_number": full["table_number"],
        "text": body,
        "cut": len(body) < len(full["text"]),
    }
    if band_lines:
        entry["band_lines"] = band_lines
    return entry


def _query_with_refs(query: str, question: str) -> str:
    """The search text. A clause or table number the question names must survive the planner's
    rewrite (docs-mcp fetches it directly), so any that the rewrite dropped is put back."""
    clauses, tables = extract_refs(question)
    lowered = query.lower()
    missing = [f"clause {c}" for c in clauses if c.lower() not in lowered]
    missing += [f"Table {t}" for t in tables if t.lower() not in lowered]
    return f"{query} {' '.join(missing)}" if missing else query


def _hit_summary(hit: dict, rank: int) -> dict:
    """A search result (or fetched chunk) reduced to what an eval needs to judge retrieval."""
    return {
        "rank": rank,
        "chunk_id": hit["chunk_id"],
        "spec": hit["spec"],
        "clause": hit["clause"],
        "table_number": hit.get("table_number"),
    }


def make_doc_agent(deps: GraphDeps):
    def doc_agent(state: GraphState) -> dict:
        s = deps.settings
        started = time.perf_counter()
        plan = state.get("plan") or {}
        extra: dict = {}
        doc_hits: list[dict] = []  # what the search (or band lookup) returned, for evals
        try:
            chunks: list[dict] = []
            if plan.get("bands"):
                # named bands: read their rows through sql-mcp, cite the chunks that hold them
                rows, info = lookup_bands(deps.sql_tools, plan["bands"], state["question"])
                extra["band_lookup"] = info
                groups: dict[str, list[str]] = {}
                for row in rows:
                    groups.setdefault(row["source_chunk_id"], []).append(describe_row(row))
                for chunk_id, lines in groups.items():
                    full = deps.docs_tools.get_chunk(chunk_id)
                    chunks.append(
                        _chunk_entry(full, f"D{len(chunks) + 1}", s.orch_chunk_chars, lines)
                    )
                doc_hits = [_hit_summary(c, i) for i, c in enumerate(chunks, start=1)]
                if not chunks:
                    stat = {"calls": 0, "cost_usd": 0.0}
                    return {
                        "halt": "none of the named bands ("
                        + ", ".join(info["requested"])
                        + ") is in the operating-band tables",
                        "band_lookup": info,
                        "node_stats": {"doc_agent": stat},
                    }
            else:
                hits = deps.docs_tools.search(
                    _query_with_refs(plan.get("doc_query") or state["question"], state["question"]),
                    state.get("spec"),
                    s.orch_retrieval_mode,
                    s.orch_doc_k,
                    state["question"],  # topic words for the variant-clause rule
                )
                doc_hits = [_hit_summary(h, i) for i, h in enumerate(hits, start=1)]
                used = 0
                for hit in hits[: s.orch_doc_full_chunks]:
                    full = deps.docs_tools.get_chunk(hit["chunk_id"])
                    room = min(s.orch_chunk_chars, max(s.orch_evidence_chars - used, 0))
                    if room <= 0:
                        break
                    entry = _chunk_entry(full, f"D{len(chunks) + 1}", room)
                    used += len(entry["text"])
                    chunks.append(entry)
        except (DocsMcpError, DocsMcpUnavailable, SqlMcpUnavailable, BandLookupError) as e:
            return {"halt": f"doc agent: {e}", "node_stats": {"doc_agent": {"calls": 0}}}
        stat = {"calls": 0, "cost_usd": 0.0, "ms": int((time.perf_counter() - started) * 1000)}
        if not chunks:
            return {
                "halt": "no specification passages were found for this question",
                "node_stats": {"doc_agent": stat},
            }
        return {"chunks": chunks, "doc_hits": doc_hits, "node_stats": {"doc_agent": stat}, **extra}

    return doc_agent


# --- SQL agent ----------------------------------------------------------------------------------


def _column_flags(deps: GraphDeps, sql_text: str, columns: list[str]) -> dict[str, dict]:
    comments = deps.sql_tools.column_comments()
    notes = deps.sql_tools.column_notes()
    flags: dict[str, dict] = {}
    for name, note in notes.items():
        if not (note["unknown"] or note["assumed"]):
            continue
        if name in columns or re.search(rf"\b{re.escape(name)}\b", sql_text, re.I):
            flags[name] = {
                "unknown": note["unknown"],
                "assumed": note["assumed"],
                "units": sorted(units_in(comments.get(name, ""))),
            }
    return flags


def _sql_retry_note(state: GraphState, failures: list[dict]) -> str:
    previous = state.get("sql") or {}
    lines = "\n".join(f"- {f['type']}: {f['detail']}" for f in failures)
    sql_text = " ".join((previous.get("executed_sql") or previous.get("sql") or "").split())
    return (
        "Your previous SQL ran, but the answer built on it failed verification.\n"
        f"Previous SQL: {sql_text}\n"
        f"Problems:\n{lines}\n"
        "Write a corrected SQL that fixes exactly these problems and still keeps every "
        "constraint of the question."
    )


def make_sql_agent(deps: GraphDeps):
    def sql_agent(state: GraphState) -> dict:
        qid = state["question_id"]
        deps.run.seed_question(qid)
        feedback = state.get("sql_feedback") or []
        context = None
        if state["route"] == "mixed":
            # the original question goes in as the question (and is what sql_query_log records);
            # the planner's restatement is context and never overrides the original's constraints
            context = mixed_sql_context(
                (state.get("plan") or {}).get("sql_question") or "",
                state.get("chunks") or [],
                deps.settings.orch_evidence_chars,
            )
        if feedback:
            context = "\n\n".join(p for p in (context, _sql_retry_note(state, feedback)) if p)
        try:
            result = answer_question(
                state["question"],
                run=deps.run,
                tools=deps.sql_tools,
                question_id=qid,
                context=context,
            )
        except LoopDetected as e:
            return {**_loop_halt(state, e, "sql_agent"), "node_stats": {"sql_agent": {"calls": 0}}}
        except SqlMcpUnavailable as e:
            return {"halt": str(e), "node_stats": {"sql_agent": {"calls": 0}}}
        stat = {"calls": result.calls, "cost_usd": result.cost_usd}
        keep = deps.settings.orch_max_state_rows
        sql = {
            "status": result.status,
            "sql": result.sql,
            "executed_sql": result.executed_sql,
            "query_id": result.query_id,
            "columns": result.columns,
            "rows": result.rows[:keep],
            "row_count": result.row_count,
            "truncated": result.truncated or len(result.rows) > keep,
            "tables_used": result.tables_used,
            "notes": result.notes,
            "corrections": result.corrections,
            "column_flags": {},
        }
        if result.status == "answered":
            sql["column_flags"] = _column_flags(
                deps, result.executed_sql or result.sql or "", result.columns
            )
        update: dict = {"sql": sql, "node_stats": {"sql_agent": stat}}
        if feedback:
            # the new SQL gets a fresh writer chain; the old draft and verdict no longer apply
            update |= {
                "sql_feedback": [],
                "sql_retries": state.get("sql_retries", 0) + 1,
                "writer_attempts": 0,
                "draft": "",
                "verification": {},
            }
        if result.status == "failed" and any(CAP_NOTE in n for n in result.notes):
            update["halt"] = next(n for n in result.notes if CAP_NOTE in n)
        return update

    return sql_agent


# --- stats agent --------------------------------------------------------------------------------


def make_stats_agent(deps: GraphDeps):
    def stats_agent(state: GraphState) -> dict:
        qid = state["question_id"]
        deps.run.seed_question(qid)
        if deps.analytics_tools is None:
            return {
                "halt": "analytics-mcp is not configured for this run",
                "node_stats": {"stats_agent": {"calls": 0}},
            }
        # the planner's restatement keeps the question's feature numbers and counts; the original
        # question is what analytics_call_log records
        asked = (state.get("plan") or {}).get("stats_question") or state["question"]
        try:
            result = answer_stats(
                asked if asked == state["question"] else f"{state['question']}\n({asked})",
                run=deps.run,
                tools=deps.analytics_tools,
                question_id=qid,
            )
        except LoopDetected as e:
            return {
                **_loop_halt(state, e, "stats_agent"),
                "node_stats": {"stats_agent": {"calls": 0}},
            }
        except AnalyticsMcpUnavailable as e:
            return {"halt": str(e), "node_stats": {"stats_agent": {"calls": 0}}}
        stat = {"calls": result.calls, "cost_usd": result.cost_usd}
        stats = {
            "status": result.status,
            "analysis": result.analysis,
            "params": result.params,
            "call_id": result.call_id,
            "analysis_id": result.analysis_id,
            "result": result.result,
            "artifacts": result.artifacts,
            "dataset": result.dataset,
            "n_runs_used": result.n_runs_used,
            "n_runs_in_extract": result.n_runs_in_extract,
            "missing_policy": result.missing_policy,
            "warnings": result.warnings,
            "notes": result.notes,
            "corrections": result.corrections,
        }
        update: dict = {"stats": stats, "node_stats": {"stats_agent": stat}}
        if result.status == "failed" and any(CAP_NOTE in n for n in result.notes):
            update["halt"] = next(n for n in result.notes if CAP_NOTE in n)
        return update

    return stats_agent


# --- writer -------------------------------------------------------------------------------------


WRITER_RAW_CHARS = 2000  # how much of a raw writer output draft_history keeps
SHORTEN = (
    "Shorten the answer: summarise long lists (name the first few items and give the count), "
    "leave out detail the question did not ask for, and keep the citations."
)


def _correction(failures: list[dict], retry: int = 1) -> str:
    """The feedback for a writer retry. It names the retry number and the failures, so no retry
    prompt can repeat an earlier one, even when the previous draft was empty."""
    lines = "\n".join(f"- {f['type']}: {f['detail']}" for f in failures) or "- no usable answer"
    shorten = f"\n{SHORTEN}" if any(f["type"] == "writer_truncated" for f in failures) else ""
    return (
        f"Retry {retry}. Your previous answer failed verification against the evidence:\n"
        f"{lines}{shorten}\n"
        "Rewrite it so that every claim is supported and cited. Remove any claim you cannot "
        'support. Answer with the JSON object {"answer": "..."} only.'
    )


def _truncation_failure(e: BadStructuredOutput) -> tuple[str, str]:
    """(raw output, failure detail) for a writer call whose output was not a complete JSON
    answer: cut off at the output limit, or not valid JSON at all."""
    result = e.result
    if result.truncated:
        why = (
            f"the answer was cut off at the output limit ({result.finish_reason}) before the "
            "JSON was complete"
        )
    else:
        why = "the output was not a valid JSON answer"
    return result.text or "", why


def make_writer(deps: GraphDeps):
    def writer(state: GraphState) -> dict:
        qid = state["question_id"]
        deps.run.seed_question(qid)
        attempts = state.get("writer_attempts", 0)
        messages = [
            Message.system(WRITER_SYSTEM),
            Message.user(
                writer_user_message(
                    state["question"],
                    state.get("sql"),
                    state.get("chunks") or [],
                    deps.settings.orch_writer_rows,
                    state.get("band_lookup"),
                    state.get("stats"),
                )
            ),
        ]
        if attempts:  # always say what went wrong, even if the previous draft was empty
            failures = (state.get("verification") or {}).get("failures") or []
            previous = state.get("draft") or (state.get("writer_raw") or "")[:800]
            messages += [
                Message.assistant(previous or "(no usable answer)"),
                Message.user(_correction(failures, attempts)),
            ]
        with _measured(deps, qid) as stat:
            try:
                generated = deps.run.generate(
                    messages,
                    purpose="writer",
                    json_schema=WRITER_SCHEMA,
                    question_id=qid,
                    max_output_tokens=deps.settings.orch_writer_max_tokens,
                )
                draft = str((generated.parsed or {}).get("answer") or "").strip()
                raw, failure = generated.text or "", ""
                if generated.truncated:  # valid JSON, but the model hit the output limit
                    failure = (
                        f"the answer was cut off at the output limit ({generated.finish_reason})"
                    )
            except CallCapExceeded as e:
                return {"halt": _halt_cap(e)["halt"], "node_stats": {"writer": stat}}
            except LoopDetected as e:
                return {**_loop_halt(state, e, "writer"), "node_stats": {"writer": stat}}
            except ProviderError as e:
                return {"halt": f"the writer call failed: {e}", "node_stats": {"writer": stat}}
            except BadStructuredOutput as e:
                draft = ""  # the verifier reports writer_truncated and the retry rule applies
                raw, failure = _truncation_failure(e)
        return {
            "draft": draft,
            "writer_raw": raw[:WRITER_RAW_CHARS],
            "writer_failure": failure,
            "writer_attempts": attempts + 1,
            "node_stats": {"writer": stat},
        }

    return writer


# --- verifier, redact, finalize -----------------------------------------------------------------


def verifier(state: GraphState) -> dict:
    result = verify(
        state.get("draft") or "",
        question=state["question"],
        route=state["route"],
        sql=state.get("sql"),
        chunks=state.get("chunks") or [],
        sql_question=(state.get("plan") or {}).get("sql_question") or "",
        stats=state.get("stats"),
    )
    if state.get("writer_failure"):  # the model gave no complete answer: not an "empty draft"
        failure = {"type": "writer_truncated", "sentence": None, "detail": state["writer_failure"]}
        result = {**result, "ok": False, "failures": [failure, *result["failures"]]}
        if not state.get("draft"):
            result["failures"] = [failure]
    # every draft and its failures stay in the checkpoint, so a redacted answer can be explained
    history = state.get("draft_history") or []
    entry = {
        "attempt": len(history) + 1,
        "draft": state.get("draft") or "",
        "raw": state.get("writer_raw") or "",
        "ok": result["ok"],
        "failures": result["failures"],
    }
    # only SQL-side failures: the writer cannot fix them, so send the question back to the SQL
    # agent (once); prose failures stay with the writer
    kinds = {f["type"] for f in result["failures"]}
    sql_only = bool(kinds) and kinds <= set(SQL_FAILURES)
    retry = sql_only and state.get("sql_retries", 0) < MAX_SQL_RETRIES
    return {
        "verification": result,
        "draft_history": [*history, entry],
        "sql_feedback": result["failures"] if retry else [],
        "node_stats": {"verifier": {"calls": 0}},
    }


def _fallback_text(state: GraphState, drop_sql: bool) -> str:
    sql = None if drop_sql else state.get("sql")
    lines: list[str] = []
    if sql and sql.get("status") == "answered":
        lines.append(
            f"The query returned {sql['row_count']} row(s) [S1]: columns "
            + ", ".join(sql["columns"])
            + "."
        )
        for row in sql["rows"][:10]:
            lines.append("- " + " | ".join(str(c) for c in row) + " [S1]")
        if any(f.get("assumed") for f in (sql.get("column_flags") or {}).values()):
            lines.append("Units of columns marked ASSUMED in the schema are assumed.")
    elif drop_sql:
        lines.append("The database result was withheld because the query could not be verified.")
    stats = state.get("stats")
    if stats and stats.get("status") == "answered":
        lines.append(
            f"The analysis {stats['analysis']} ran on {stats['n_runs_used']} run(s) [A1]; "
            "its result is in the Sources section."
        )
    for chunk in state.get("chunks") or []:
        lines.append(
            f"Relevant passage: TS {chunk['spec']} clause {chunk['clause']} [{chunk['ref']}]."
        )
    return "\n".join(lines)


def redact(state: GraphState) -> dict:
    """Retries are used up: keep only what verifies, mark the answer partial."""
    failures = (state.get("verification") or {}).get("failures") or []
    bad = {f["sentence"] for f in failures if f["sentence"] is not None}
    drop_sql = any(f["type"] in SQL_FAILURES for f in failures)
    sentences = split_sentences(state.get("draft") or "")
    kept = [s for i, s in enumerate(sentences) if i not in bad and not (drop_sql and "[S1]" in s)]
    sql = None if drop_sql else state.get("sql")
    recheck = (
        verify(
            "\n".join(kept),
            question=state["question"],
            route=state["route"],
            sql=sql,
            chunks=state.get("chunks") or [],
            sql_question=(state.get("plan") or {}).get("sql_question") or "",
            stats=state.get("stats"),
        )
        if kept
        else None
    )
    kinds = sorted({f["type"] for f in failures})
    if recheck and recheck["ok"]:
        text = "\n".join(kept)
        verification = {**recheck, "redacted": kinds}
    else:
        text = _fallback_text(state, drop_sql)
        verification = {"ok": False, "failures": failures, "evidence_map": [], "redacted": kinds}
    text += (
        "\n\nNote: this answer is partial. Parts of the draft could not be verified against "
        f"the evidence and were removed ({', '.join(kinds)})."
    )
    update: dict = {"answer": text, "status": "partial", "verification": verification}
    if drop_sql:
        update["sql"] = None
        update["verification"]["withheld_sql_query_id"] = (state.get("sql") or {}).get("query_id")
    return update


def sources_section(answer: str, state: GraphState) -> str:
    cited = {(m.group(1), int(m.group(2))) for m in CITE_RE.finditer(answer)}
    lines: list[str] = []
    sql = state.get("sql")
    stats = state.get("stats")
    if ("A", 1) in cited and stats and stats.get("status") == "answered":
        lines.append(
            f"[A1] analytics-mcp call #{stats['call_id']}: {stats['analysis']}("
            + json.dumps(stats["params"], separators=(",", ":"))
            + f") on SECOM, {stats['n_runs_used']} run(s), extract "
            + str((stats.get("dataset") or {}).get("extract_sha256", ""))[:12]
        )
        for art in stats.get("artifacts") or []:
            lines.append(f"    plot artifact {art['id']} (PNG, {art['bytes']} bytes)")
    if ("S", 1) in cited and sql and sql.get("status") == "answered":
        lines.append(
            f"[S1] sql-mcp query #{sql['query_id']} ({sql['row_count']} row(s)): "
            + " ".join((sql.get("executed_sql") or sql.get("sql") or "").split())
        )
    for i, chunk in enumerate(state.get("chunks") or [], start=1):
        if ("D", i) in cited:
            title = f", {chunk['clause_title']}" if chunk.get("clause_title") else ""
            table = f", table {chunk['table_number']}" if chunk.get("table_number") else ""
            lines.append(
                f"[D{i}] TS {chunk['spec']} V{chunk['version']}, clause {chunk['clause']}"
                f"{title}{table} (chunk {chunk['chunk_id']})"
            )
    return "\n".join(lines)


def finalize(state: GraphState) -> dict:
    status = state.get("status")
    sql = state.get("sql") or {}
    stats = state.get("stats") or {}
    if status in ("cancelled", "partial"):
        answer = state["answer"]
    elif state.get("halt"):
        status, answer = "failed", f"Stopped without an answer: {state['halt']}."
    elif state.get("route") == "refuse":
        status = "refused"
        answer = (state.get("plan") or {}).get("refuse_reason") or "This question is out of scope."
    elif sql.get("status") == "refused":
        status = "refused"
        answer = (sql.get("notes") or ["The question was refused."])[0]
    elif stats.get("status") == "refused":
        status = "refused"
        answer = (stats.get("notes") or ["The question was refused."])[0]
    elif stats and stats.get("status") != "answered":
        status = "failed"
        answer = "The stats agent could not answer: " + ((stats.get("notes") or ["no reason"])[-1])
    elif sql and sql.get("status") != "answered":
        status = "failed"
        answer = (
            "The SQL agent could not answer: " + ((sql.get("notes") or ["no reason given"])[-1])
        )
    elif (state.get("verification") or {}).get("ok"):
        status, answer = "answered", state["draft"]
    else:
        status, answer = "failed", "No verified answer was produced."
    sources = sources_section(answer, state) if status in ("answered", "partial") else ""
    return {"status": status, "answer": answer, "sources": sources}
