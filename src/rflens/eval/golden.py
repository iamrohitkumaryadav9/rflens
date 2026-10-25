"""The golden set (eval/golden.yaml): one file for every question type.

Fields: id, type (sql | doc | mixed | refuse), author (human | assistant: who wrote the question;
a human edited and reviewed every "human" one), question, status (draft | reviewed), notes, and per
type: reference_sql + reference_result (sql, mixed), spec + expected_clauses / expected_tables +
confirmed_chunks (doc, mixed: the document part), expected_refusal (refuse). Optional: `ordered`
(the question asks for a ranking, so row order counts) and `must_mention` (words the agent's notes
must contain, case-insensitive; used for the ERP unit).

A doc question is answered by a chunk of `spec` whose clause equals one of expected_clauses
exactly, or whose table number is in expected_tables. Assistant-written doc questions have exactly
one expected clause or table, and `confirmed_chunks` lists chunks (ids) known to exist in the
corpus for it. Mixed questions are not scored yet (the week 4 orchestrator runs them).

`python -m rflens.eval.golden` runs every reference_sql (sql and mixed) through sql-mcp and stores
the result in the file (the reference result), so the data behind the golden answers is reviewable.
"""

import datetime
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from rflens.agents.sql_tools import SqlMcpTools
from rflens.eval.scorer import ResultSet

GOLDEN_PATH = Path(__file__).resolve().parents[3] / "eval" / "golden.yaml"
TYPES = ("sql", "doc", "mixed", "refuse")
STATUSES = ("draft", "reviewed")
AUTHORS = ("human", "assistant")
SPECS = ("36.101", "38.101-1")
MAX_REFERENCE_ROWS = 50


@dataclass
class GoldenQuestion:
    id: str
    type: str
    question: str
    status: str
    notes: str = ""
    reference_sql: str | None = None
    reference_result: ResultSet | None = None
    ordered: bool = False
    must_mention: list[str] = field(default_factory=list)
    expected_clauses: list[str] = field(default_factory=list)
    expected_tables: list[str] = field(default_factory=list)
    confirmed_chunks: list[str] = field(default_factory=list)
    spec: str | None = None
    author: str = "human"
    expected_refusal: str | None = None


def load_raw(path: Path = GOLDEN_PATH) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def parse(raw: dict) -> list[GoldenQuestion]:
    questions = []
    for q in raw["questions"]:
        result = q.get("reference_result")
        questions.append(
            GoldenQuestion(
                id=q["id"],
                type=q["type"],
                question=q["question"],
                status=q.get("status", "draft"),
                notes=q.get("notes", ""),
                reference_sql=q.get("reference_sql"),
                reference_result=(ResultSet(result["columns"], result["rows"]) if result else None),
                ordered=bool(q.get("ordered", False)),
                must_mention=list(q.get("must_mention", [])),
                expected_clauses=list(q.get("expected_clauses") or []),
                expected_tables=list(q.get("expected_tables") or []),
                confirmed_chunks=list(q.get("confirmed_chunks") or []),
                spec=q.get("spec"),
                author=q.get("author", ""),
                expected_refusal=q.get("expected_refusal"),
            )
        )
    return questions


def load_golden(path: Path = GOLDEN_PATH) -> list[GoldenQuestion]:
    return parse(load_raw(path))


def problems(questions: list[GoldenQuestion]) -> list[str]:
    """Structural problems of the golden set (empty list: fine)."""
    found: list[str] = []
    seen: set[str] = set()
    for q in questions:
        if q.id in seen:
            found.append(f"{q.id}: duplicate id")
        seen.add(q.id)
        if q.type not in TYPES:
            found.append(f"{q.id}: unknown type {q.type!r}")
        if q.status not in STATUSES:
            found.append(f"{q.id}: unknown status {q.status!r}")
        if not q.question.strip():
            found.append(f"{q.id}: empty question")
        if q.author not in AUTHORS:
            found.append(f"{q.id}: author must be one of {AUTHORS}, got {q.author!r}")
        if q.type in ("sql", "mixed"):
            if not q.reference_sql:
                found.append(f"{q.id}: {q.type} question without reference_sql")
            if q.reference_result is None:
                found.append(f"{q.id}: reference_sql has no stored reference_result")
            if not q.notes.strip():
                found.append(f"{q.id}: no notes (state the interpretation)")
        if q.type in ("doc", "mixed"):
            found += _doc_problems(q)
        if q.type == "refuse" and not q.expected_refusal:
            found.append(f"{q.id}: refuse question without expected_refusal")
    return found


def _doc_problems(q: GoldenQuestion) -> list[str]:
    """Problems of the document part of a doc or mixed question."""
    found = []
    if q.spec not in SPECS:
        found.append(f"{q.id}: spec must be one of {SPECS}, got {q.spec!r}")
    expected = len(q.expected_clauses) + len(q.expected_tables)
    if expected == 0:
        found.append(f"{q.id}: no expected clause or table")
    if q.author == "assistant" and (expected != 1 or not q.confirmed_chunks):
        found.append(
            f"{q.id}: assistant-written: exactly one expected clause or table, and confirmed_chunks"
        )
    for chunk_id in q.confirmed_chunks:
        if not chunk_id.startswith(f"{q.spec}@"):
            found.append(f"{q.id}: confirmed chunk {chunk_id} is not from spec {q.spec}")
        elif (
            q.expected_clauses
            and chunk_id.split("/")[1] not in q.expected_clauses
            and not q.expected_tables
        ):
            found.append(f"{q.id}: confirmed chunk {chunk_id} is not in the expected clause")
    return found


# --- storing reference results -----------------------------------------------------------------


class _Dumper(yaml.SafeDumper):
    pass


def _str_representer(dumper: yaml.SafeDumper, value: str):
    style = "|" if "\n" in value else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style=style)


_Dumper.add_representer(str, _str_representer)


def refresh(path: Path, tools: SqlMcpTools, today: datetime.date | None = None) -> list[str]:
    """Run each sql question's reference_sql through sql-mcp and write the result into the file.
    Returns one report line per question."""
    raw = load_raw(path)
    today = today or datetime.date.today()
    lines = []
    for q in raw["questions"]:
        if q["type"] not in ("sql", "mixed"):
            continue
        outcome = tools.run_query(q["reference_sql"], f"golden reference {q['id']}")
        if not outcome.ok:
            raise SystemExit(f"{q['id']}: reference_sql failed: {outcome.error}")
        if outcome.row_count > MAX_REFERENCE_ROWS:  # also catches the server's default LIMIT
            raise SystemExit(f"{q['id']}: reference result too large ({outcome.row_count} rows)")
        q["reference_result"] = {
            "columns": outcome.columns,
            "rows": outcome.rows,
            "row_count": outcome.row_count,
            "checked_on": today.isoformat(),
            "via": "sql-mcp",
        }
        lines.append(f"{q['id']}: {outcome.row_count} row(s) {outcome.rows[:5]}")
    path.write_text(
        yaml.dump(raw, Dumper=_Dumper, sort_keys=False, allow_unicode=True, width=100_000),
        encoding="utf-8",
    )
    return lines


def main() -> None:
    from rflens.config import get_agent_settings

    tools = SqlMcpTools(get_agent_settings().sql_mcp_url)
    for line in refresh(GOLDEN_PATH, tools):
        print(line)
    problem_list = problems(load_golden())
    if problem_list:
        print("\n".join(problem_list), file=sys.stderr)
        raise SystemExit(1)
    print(f"stored reference results in {GOLDEN_PATH}")


if __name__ == "__main__":
    main()
