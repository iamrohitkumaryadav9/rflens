"""Execution-accuracy scorer for the SQL agent: does its result set contain the reference?

Rules
- Column names are ignored. Matching is relaxed on columns: the result matches if, for every
  reference column, the agent's result has a column with the same values. Extra agent columns are
  allowed; a missing reference column fails. Agent columns may come in any order.
- The row count must match exactly.
- Row order is ignored unless `ordered` (the question asks for a ranking); then the values of each
  column must also be in the reference order.
- The matched columns are compared row by row, so values that match column-wise but come from
  different rows do not pass.
- Numbers (int, float, Decimal) compare with relative tolerance 1e-6 (absolute 1e-12, so float
  noise near zero is not a mismatch). A bool is not a number; a number is not its string form.
- NULL equals NULL and nothing else. Strings, dates (ISO strings) and the rest compare exactly.
- Empty results have no values to compare: they match if the agent has at least as many columns
  as the reference.
"""

import math
from dataclasses import dataclass
from decimal import Decimal
from itertools import product
from typing import Any

REL_TOL = 1e-6
ABS_TOL = 1e-12


@dataclass(frozen=True)
class ResultSet:
    columns: list[str]
    rows: list[list[Any]]


@dataclass(frozen=True)
class Verdict:
    match: bool
    reason: str = ""


def _is_number(v: Any) -> bool:
    return isinstance(v, int | float | Decimal) and not isinstance(v, bool)


def values_equal(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if _is_number(a) and _is_number(b):
        return math.isclose(float(a), float(b), rel_tol=REL_TOL, abs_tol=ABS_TOL)
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    return type(a) is type(b) and a == b


def rows_equal(a: list[Any], b: list[Any]) -> bool:
    return len(a) == len(b) and all(values_equal(x, y) for x, y in zip(a, b, strict=True))


def _column(rows: list[list[Any]], index: int) -> list[list[Any]]:
    return [[row[index]] for row in rows]


def _rows_match(agent: list[list[Any]], reference: list[list[Any]], ordered: bool) -> str:
    """'' if the two row lists match (same length assumed), else why not."""
    if ordered:
        for i, (a, r) in enumerate(zip(agent, reference, strict=True), 1):
            if not rows_equal(a, r):
                return f"row {i} differs: {a} vs reference {r}"
        return ""
    unmatched = list(agent)
    for r in reference:
        for i, a in enumerate(unmatched):
            if rows_equal(a, r):
                del unmatched[i]
                break
        else:
            return f"reference row {r} not in the agent's result"
    return ""


def compare(agent: ResultSet, reference: ResultSet, ordered: bool = False) -> Verdict:
    if len(agent.rows) != len(reference.rows):
        return Verdict(False, f"{len(agent.rows)} row(s), reference has {len(reference.rows)}")
    n_agent, n_ref = len(agent.columns), len(reference.columns)
    if n_agent < n_ref:
        return Verdict(False, f"{n_agent} column(s), reference needs {n_ref}")
    if not reference.rows:
        return Verdict(True)

    # Candidate agent columns for each reference column: those with the same values.
    candidates: list[list[int]] = []
    for j in range(n_ref):
        ref_col = _column(reference.rows, j)
        found = [
            i for i in range(n_agent) if not _rows_match(_column(agent.rows, i), ref_col, ordered)
        ]
        if not found:
            return Verdict(False, f"no agent column has the values of reference column {j + 1}")
        candidates.append(found)

    # Pick distinct agent columns for the reference columns and compare whole rows, so columns
    # that agree one by one but pair up differently across rows are caught.
    last = ""
    for choice in product(*candidates):
        if len(set(choice)) < len(choice):
            continue
        projected = [[row[i] for i in choice] for row in agent.rows]
        last = _rows_match(projected, reference.rows, ordered)
        if not last:
            return Verdict(True)
    return Verdict(False, last or "the reference columns cannot be matched to distinct columns")
