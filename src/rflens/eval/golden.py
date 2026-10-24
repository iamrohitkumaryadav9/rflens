"""The golden set (eval/golden.yaml): one file for every question type.

Fields: id, type (sql | doc | mixed | refuse), question, status (draft | reviewed), notes, and per
type: reference_sql + reference_result (sql), expected_clauses (doc, later), expected_refusal
(refuse). Optional: `ordered` (the question asks for a ranking, so row order counts) and
`must_mention` (words the agent's notes must contain, case-insensitive; used for the ERP unit).

`python -m rflens.eval.golden` runs every reference_sql through sql-mcp and stores the result in
the file (the reference result), so the data behind the golden answers is reviewable.
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
                expected_clauses=list(q.get("expected_clauses", [])),
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
        if q.type == "sql":
            if not q.reference_sql:
                found.append(f"{q.id}: sql question without reference_sql")
            if q.reference_result is None:
                found.append(f"{q.id}: reference_sql has no stored reference_result")
            if not q.notes.strip():
                found.append(f"{q.id}: no notes (state the interpretation)")
        if q.type == "refuse" and not q.expected_refusal:
            found.append(f"{q.id}: refuse question without expected_refusal")
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
        if q["type"] != "sql":
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
