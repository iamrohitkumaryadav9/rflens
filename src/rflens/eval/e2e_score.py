"""Scoring of an end-to-end run (make eval-e2e): the final answer of the whole graph, per question.

Pure functions of a golden question and the final graph state (a JSON dict); no database, no model.

- sql: the answer cites the SQL result [S1], that result matches the stored reference under the
  relaxed `compare()` (scorer.py), and the answer text contains every reference value (a ranked
  question in reference order); `must_mention` words are in the answer text.
- doc: a cited chunk hits the expected clause or table (right spec, exact clause or listed table)
  and the answer is `answered` with verification passed. Evidence recall is reported separately:
  the expected target among the search results, and among the chunks the writer was given.
- mixed: the doc part as for doc, and the SQL part as for sql (incl. `must_mention`).
- stats: the final state holds an answered analysis of the reference menu entry whose result
  passes every `reference_stats.checks` entry (numbers at 1e-6 relative, strings and lists exactly),
  the answer cites it [A1] and its text contains `answer_rows` (a ranked list in order); a stats
  question that must be refused is scored like a refuse question and also needs that no analysis
  ran.
- refuse: status `refused` with a reason, and no SQL query and no document search ran.

Order (ranked questions only), scorer v2: the order of ROWS is checked, never the order of values
within one row, and rows tied on the ranking value form an unordered group: order is checked
between groups, not within them. The ranking column is the numeric, non-constant column along
which the reference rows are monotone (the one with the fewest distinct values if several are);
a reference with no such column is checked row by row.

How a reference value is found in the answer text: a number matches a number written in the text
at the text's own precision (12.35 -> 12.3, 12; 1,234 = 1234); zero also matches the words
zero / no / none; a string matches case-insensitively as a substring. NULL needs no text.
"""

import math
import re
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from rflens.eval.golden import GoldenQuestion
from rflens.eval.scorer import ResultSet, compare
from rflens.graph.verifier import CITE_RE, NUM_RE, clean_numbers, parse_number

SQL_TYPES = ("sql", "mixed")
DOC_TYPES = ("doc", "mixed")
CAUSES = (
    "retrieval miss",
    "wrong SQL",
    "writer/verifier redaction",
    "wrong route",
    "refusal error",
    "wrong analysis",
    "scorer issue",
    "other",
)


@dataclass
class Score:
    qid: str
    type: str
    author: str
    question: str
    correct: bool
    reason: str
    status: str
    verified: bool
    route: str | None
    cited: list[str]
    sql_ok: bool | None  # sql and mixed questions
    doc_hit: bool | None  # doc and mixed questions: a cited chunk hits the target
    recall_search: bool | None  # target among the search results (or looked-up chunks)
    recall_evidence: bool | None  # target among the chunks the writer saw
    writer_retries: int
    sql_retries: int
    calls: int
    cost_usd: float
    failure_types: list[str]
    cause: str = ""
    answer: str = ""
    sources: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


# --- values in the answer text --------------------------------------------------------------------


def _number_tokens(text: str) -> list[tuple[int, float, int]]:
    """(start, value, decimals) of the numbers written in `text`."""
    out = []
    for m in NUM_RE.finditer(clean_numbers(text)):
        value, decimals = parse_number(m.group(0))
        out.append((m.start(), value, decimals))
    return out


def _find(value: Any, lowered: str, tokens: list, start: int) -> int | None:
    """Position of the first occurrence of `value` at or after `start`, or None. `lowered` is the
    answer with single spaces and lower case; `tokens` are the numbers written in it."""
    if value is None:
        return start  # NULL: nothing to find
    if isinstance(value, bool):
        value = str(value)
    if isinstance(value, int | float | Decimal):
        v = float(value)
        for pos, x, decimals in tokens:
            if pos >= start and abs(x - v) <= 0.5 * 10**-decimals + 1e-9:
                return pos
        if v == 0:
            m = re.compile(r"\b(zero|no|none)\b").search(lowered, start)
            return m.start() if m else None
        return None
    at = lowered.find(" ".join(str(value).split()).lower(), start)
    return None if at < 0 else at


def ranking_groups(rows: list[list[Any]]) -> list[list[int]]:
    """Row indexes grouped by ties on the ranking column (see the module doc); one group per row
    when there is no ranking column."""

    def numeric(v: Any) -> bool:
        return isinstance(v, int | float | Decimal) and not isinstance(v, bool)

    best: tuple[int, int] | None = None  # (distinct values, column)
    for k in range(len(rows[0]) if rows else 0):
        values = [r[k] for r in rows]
        if not all(numeric(v) for v in values) or len(set(values)) < 2:
            continue
        down = all(a >= b for a, b in zip(values, values[1:], strict=False))
        up = all(a <= b for a, b in zip(values, values[1:], strict=False))
        if (down or up) and (best is None or (len(set(values)), k) < best):
            best = (len(set(values)), k)
    if best is None:
        return [[i] for i in range(len(rows))]
    k = best[1]
    groups: list[list[int]] = []
    for i, row in enumerate(rows):
        if groups and rows[groups[-1][0]][k] == row[k]:
            groups[-1].append(i)
        else:
            groups.append([i])
    return groups


def contains_values(answer: str, reference: ResultSet, ordered: bool) -> tuple[bool, str]:
    """Is every reference value in the answer text (a ranked result in reference order)?"""
    text = " ".join(clean_numbers(answer).split())
    lowered = text.lower()
    tokens = _number_tokens(text)
    missing: list[str] = []
    out_of_order: list[str] = []
    if not ordered:
        for row in reference.rows:
            for value in row:
                if _find(value, lowered, tokens, 0) is None:
                    missing.append(str(value))
    else:
        start = 0
        for group in ranking_groups(reference.rows):
            anchors = []
            for i in group:
                found = []
                for value in reference.rows[i]:
                    if value is None:
                        continue  # NULL: nothing to find
                    at = _find(value, lowered, tokens, start)
                    if at is None:
                        anywhere = _find(value, lowered, tokens, 0)
                        (out_of_order if anywhere is not None else missing).append(str(value))
                    else:
                        found.append(at)
                if found:
                    anchors.append(min(found))  # a row sits where its first value is written
            if anchors:
                start = max(anchors) + 1  # the next group must come after every row of this one
    if missing:
        return False, f"the answer text lacks reference value(s) {missing[:5]}"
    if out_of_order:
        return False, f"reference row(s) with {out_of_order[:5]} are not in reference order"
    return True, ""


# --- one question ---------------------------------------------------------------------------


def cited_tags(answer: str) -> list[str]:
    return sorted({f"{k}{n}" for k, n in CITE_RE.findall(answer)})


def chunk_hits(q: GoldenQuestion, chunk: dict[str, Any]) -> bool:
    """The hit rule of eval-docs: right spec and an exact clause, or a listed table number."""
    return chunk.get("spec") == q.spec and (
        chunk.get("clause") in q.expected_clauses
        or (chunk.get("table_number") is not None and chunk["table_number"] in q.expected_tables)
    )


def _sql_part(q: GoldenQuestion, state: dict, answer: str, cited: list[str]) -> tuple[bool, str]:
    sql = state.get("sql")
    if not sql or sql.get("status") != "answered":
        return False, "no SQL result in the final state"
    if "S1" not in cited:
        return False, "the answer does not cite the SQL result [S1]"
    verdict = compare(ResultSet(sql["columns"], sql["rows"]), q.reference_result, q.ordered)
    if not verdict.match:
        return False, f"SQL result differs from the reference: {verdict.reason}"
    ok, reason = contains_values(answer, q.reference_result, q.ordered)
    if not ok:
        return False, reason
    lowered = answer.lower()
    missing = [w for w in q.must_mention if w.lower() not in lowered]
    if missing:
        return False, f"the answer does not mention {missing}"
    return True, ""


def _doc_part(q: GoldenQuestion, state: dict, cited: list[str]) -> tuple[bool, str]:
    chunks = state.get("chunks") or []
    hit = any(chunk_hits(q, c) for c in chunks if c["ref"] in cited)
    if not hit:
        return False, "no cited chunk hits the expected clause or table"
    if state.get("status") != "answered" or not (state.get("verification") or {}).get("ok"):
        return False, f"hit, but the answer is {state.get('status')} / verification not passed"
    return True, ""


def _at(obj: Any, path: str) -> Any:
    """`top.0.feature`: dict keys and list indexes separated by dots."""
    for part in path.split("."):
        obj = obj[int(part)] if isinstance(obj, list) else obj[part]
    return obj


def _same(actual: Any, expected: Any) -> bool:
    if isinstance(expected, bool) or not isinstance(expected, int | float):
        if isinstance(expected, list):
            return (
                isinstance(actual, list)
                and len(actual) == len(expected)
                and all(_same(a, e) for a, e in zip(actual, expected, strict=True))
            )
        return actual == expected
    return (
        isinstance(actual, int | float)
        and not isinstance(actual, bool)
        and math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-12 if expected == 0 else 0.0)
    )


def _stats_part(q: GoldenQuestion, state: dict, answer: str, cited: list[str]) -> tuple[bool, str]:
    stats = state.get("stats")
    if not stats or stats.get("status") != "answered":
        return False, "no answered analysis in the final state"
    if "A1" not in cited:
        return False, "the answer does not cite the analysis [A1]"
    wanted = (q.reference_analysis or {}).get("analysis")
    if stats.get("analysis") != wanted:
        return False, f"the agent ran {stats.get('analysis')}, the reference is {wanted}"
    for check in (q.reference_stats or {}).get("checks") or []:
        try:
            actual = _at(stats["result"], check["path"])
        except (KeyError, IndexError, TypeError, ValueError):
            return False, f"the result has no {check['path']}"
        if not _same(actual, check["value"]):
            return False, f"{check['path']} is {actual!r}, the reference is {check['value']!r}"
    rows = (q.reference_stats or {}).get("answer_rows") or []
    ok, reason = (
        contains_values(
            answer,
            ResultSet(["v"] * len(rows[0]), rows),
            bool((q.reference_stats or {}).get("ordered")),
        )
        if rows
        else (True, "")
    )
    return ok, reason


def _refuse_part(state: dict) -> tuple[bool, str]:
    if state.get("status") != "refused":
        return False, f"expected a refusal, status is {state.get('status')}"
    if not (state.get("answer") or "").strip():
        return False, "refused without a reason"
    sql = state.get("sql") or {}
    ran = []
    if (state.get("stats") or {}).get("status") == "answered":
        ran.append("an analysis")
    if sql.get("query_id") is not None:
        ran.append("a SQL query")
    if state.get("chunks") or state.get("doc_hits") or state.get("band_lookup"):
        ran.append("a document search")
    if ran:
        return False, "refused, but " + " and ".join(ran) + " ran"
    return True, ""


def score_question(q: GoldenQuestion, state: dict) -> Score:
    answer = state.get("answer") or ""
    cited = cited_tags(answer)
    history = state.get("draft_history") or []
    sql_retries = state.get("sql_retries", 0)
    stats = state.get("node_stats") or {}
    verification = state.get("verification") or {}
    reasons: list[str] = []

    sql_ok = doc_hit = recall_search = recall_evidence = None
    if q.type == "refuse" or (q.type == "stats" and q.expected_refusal):
        ok, reason = _refuse_part(state)
        correct, reasons = ok, ([] if ok else [reason])
    elif q.type == "stats":
        ok, reason = _stats_part(q, state, answer, cited)
        correct, reasons = ok, ([] if ok else [reason])
    else:
        if q.type in SQL_TYPES:
            sql_ok, why = _sql_part(q, state, answer, cited)
            if not sql_ok:
                reasons.append(why)
        if q.type in DOC_TYPES:
            doc_hit, why = _doc_part(q, state, cited)
            if not doc_hit:
                reasons.append(why)
            recall_search = any(chunk_hits(q, h) for h in state.get("doc_hits") or [])
            recall_evidence = any(chunk_hits(q, c) for c in state.get("chunks") or [])
        correct = not reasons
    failure_types = sorted({f["type"] for h in history for f in h["failures"]})
    return Score(
        qid=q.id,
        type=q.type,
        author=q.author,
        question=q.question,
        correct=correct,
        reason="; ".join(reasons),
        status=state.get("status") or "unknown",
        verified=bool(verification.get("ok")) and state.get("status") == "answered",
        route=state.get("route"),
        cited=cited,
        sql_ok=sql_ok,
        doc_hit=doc_hit,
        recall_search=recall_search,
        recall_evidence=recall_evidence,
        writer_retries=max(len(history) - 1 - sql_retries, 0),
        sql_retries=sql_retries,
        calls=sum(s.get("calls", 0) for s in stats.values()),
        cost_usd=round(sum(s.get("cost_usd", 0.0) for s in stats.values()), 8),
        failure_types=failure_types,
        cause="" if correct else classify_miss(q, state, sql_ok, doc_hit, recall_evidence),
        answer=answer,
        sources=state.get("sources") or "",
        extra={
            "halt": state.get("halt"),
            "plan": state.get("plan"),
            "sql": state.get("sql"),
            "stats": state.get("stats"),
            "chunk_ids": [c["chunk_id"] for c in state.get("chunks") or []],
            "doc_hits": state.get("doc_hits"),
            "redacted": verification.get("redacted"),
            "failures": verification.get("failures"),
        },
    )


def classify_miss(
    q: GoldenQuestion, state: dict, sql_ok: bool | None, doc_hit: bool | None, recall_evidence
) -> str:
    """A first guess at the cause of a miss, from the state alone; a human reads each miss."""
    status = state.get("status")
    route = state.get("route")
    if q.type == "refuse" or (q.type == "stats" and q.expected_refusal):
        return "refusal error"
    if route != q.type:
        return "wrong route"
    if q.type == "stats":
        stats = state.get("stats") or {}
        if stats.get("status") != "answered":
            return "wrong analysis"
        if status in ("partial", "failed") or not (state.get("verification") or {}).get("ok"):
            return "writer/verifier redaction"
        return "wrong analysis"  # it ran, and the verified answer still misses the reference
    if state.get("halt"):
        return "other"
    if doc_hit is False and not recall_evidence:
        return "retrieval miss"
    redacted = status in ("partial", "failed") or not (state.get("verification") or {}).get("ok")
    if doc_hit is False and redacted:
        return "writer/verifier redaction"
    if sql_ok is False:
        sql = state.get("sql") or {}
        if sql.get("status") != "answered" or redacted:
            return "writer/verifier redaction"
        ref = q.reference_result
        verdict = compare(ResultSet(sql["columns"], sql["rows"]), ref, q.ordered)
        return "wrong SQL" if not verdict.match else "scorer issue"
    return "other"
