"""The golden set (eval/golden.yaml): one file for every question type.

Fields: id, type (sql | doc | mixed | stats | refuse), author (human | assistant: who wrote the
question; a human edited and reviewed every "human" one), question, status (draft | reviewed),
notes, and per type: reference_sql + reference_result (sql, mixed), spec + expected_clauses /
expected_tables + confirmed_chunks (doc, mixed: the document part), expected_refusal (refuse; also a
stats question that must be refused), reference_analysis + reference_stats (stats: the menu call
that answers it and the numbers to check, computed independently of the analytics code: see
below). Optional: `ordered` (the question asks for a ranking, so row order counts) and
`must_mention` (words the agent's notes must contain, case-insensitive; used for the ERP unit).

A doc question is answered by a chunk of `spec` whose clause equals one of expected_clauses
exactly, or whose table number is in expected_tables. Assistant-written doc questions have exactly
one expected clause or table, and `confirmed_chunks` lists chunks (ids) known to exist in the
corpus for it. Mixed questions are not scored yet (the week 4 orchestrator runs them).

A stats question is answered by a menu analysis (`reference_analysis`: name and parameters). Its
`reference_stats.checks` are (path into the analysis result, value) pairs, compared at 1e-6
relative; `answer_rows` are the values the answer text must contain. They come from
analytics_runner/reference.py (`make golden-stats-refresh`), which shares no code with the
analyses, and `notes` says how. For a ranked top N, ranks N and N+1 must differ clearly (the
refresh refuses a near-tie at the cutoff).

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
TYPES = ("sql", "doc", "mixed", "stats", "refuse")
CUTOFF_RATIO = 10.0  # p(rank N+1) / p(rank N) at the cutoff of a ranked top N
ORDER_RATIO = 2.0  # p(rank i+1) / p(rank i) inside the top N, so the order is not a coin flip
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
    reference_analysis: dict | None = None
    reference_stats: dict | None = None


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
                reference_analysis=q.get("reference_analysis"),
                reference_stats=q.get("reference_stats"),
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
        if q.type == "stats":
            found += _stats_problems(q)
    return found


def _stats_problems(q: GoldenQuestion) -> list[str]:
    """Problems of a stats question: a refusal, or a menu analysis with independent references."""
    from rflens.analytics import menu

    found = []
    if not q.notes.strip():
        found.append(f"{q.id}: no notes (state how the reference was computed)")
    if q.expected_refusal:
        if q.reference_analysis or q.reference_stats:
            found.append(f"{q.id}: a refused stats question has no reference_analysis/stats")
        return found
    ref = q.reference_analysis or {}
    try:
        menu.validate(ref.get("analysis", ""), ref.get("params", {}))
    except (ValueError, TypeError) as e:
        found.append(f"{q.id}: reference_analysis does not validate: {e}")
    stats = q.reference_stats or {}
    if not stats.get("checks"):
        found.append(f"{q.id}: no reference_stats.checks (run make golden-stats-refresh)")
    if not stats.get("answer_rows"):
        found.append(f"{q.id}: no reference_stats.answer_rows")
    if not stats.get("via"):
        found.append(f"{q.id}: reference_stats.via must say how the numbers were computed")
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


# --- stats references -------------------------------------------------------------------------


def _checks(prefix: str, values: dict) -> list[dict]:
    return [{"path": f"{prefix}{k}", "value": v} for k, v in values.items()]


def stats_reference(qid: str, params: dict, ref: dict) -> dict:
    """The checks and answer values of one stats question, from the independent computation `ref`
    (analytics_runner/reference.py). Raises SystemExit when a ranked cutoff is a near-tie."""
    if qid == "stats-01":  # feature_ranking, top N
        n = params["top_k"]
        top = ref["ranking"]
        if len(top) <= n:
            raise SystemExit(f"{qid}: the reference lists {len(top)} features, need more than {n}")
        for i in range(n - 1):  # inside the top N the order must not be a coin flip
            if top[i + 1]["p_value"] / top[i]["p_value"] < ORDER_RATIO:
                raise SystemExit(
                    f"{qid}: ranks {i + 1} and {i + 2} are near-tied "
                    f"(p {top[i]['p_value']:.3g} vs {top[i + 1]['p_value']:.3g}); change top_k"
                )
        ratio = top[n]["p_value"] / top[n - 1]["p_value"]
        if ratio < CUTOFF_RATIO:
            raise SystemExit(
                f"{qid}: near-tie at the cutoff: p of rank {n} is {top[n - 1]['p_value']:.3g}, of "
                f"rank {n + 1} {top[n]['p_value']:.3g} (ratio {ratio:.1f} < {CUTOFF_RATIO:g})"
            )
        checks = []
        for i in range(n):
            row = top[i]
            checks += _checks(
                f"top.{i}.",
                {
                    "feature": row["feature"],
                    "n_pass": row["n_pass"],
                    "n_fail": row["n_fail"],
                    "p_value": row["p_value"],
                    "q_value": row["q_value"],
                    "rank_biserial": row["rank_biserial"],
                },
            )
        return {
            "checks": checks,
            "answer_rows": [[top[i]["feature"]] for i in range(n)],
            "ordered": True,
            "cutoff": {
                "top_k": n,
                f"p_rank_{n}": top[n - 1]["p_value"],
                f"p_rank_{n + 1}": top[n]["p_value"],
                "ratio": ratio,
            },
        }
    if qid == "stats-02":  # group_compare, feature 60
        c = ref["compare60"]
        checks = _checks(
            "",
            {
                "feature": c["feature"],
                "groups.pass.n": c["n_pass"],
                "groups.fail.n": c["n_fail"],
                "groups.pass.median": c["median_pass"],
                "groups.fail.median": c["median_fail"],
                "groups.pass.mean": c["mean_pass"],
                "groups.fail.mean": c["mean_fail"],
                "p_value": c["p_value"],
                "rank_biserial": c["rank_biserial"],
                "higher_in": "fail" if c["rank_biserial"] > 0 else "pass",
            },
        )
        return {
            "checks": checks,
            # the group sizes are checked in the result, not required in the text: the question
            # asks for the medians and the effect size, not for the sample sizes
            "answer_rows": [[c["median_pass"], c["median_fail"]]],
            "ordered": False,
        }
    if qid == "stats-03":  # distribution, feature 104
        d = ref["dist104"]
        keys = ("n", "n_missing", "min", "q1", "median", "q3", "max", "counts")
        return {
            "checks": _checks("", {k: d[k] for k in keys}),
            # n is checked in the result, not required in the text: the question does not ask for it
            "answer_rows": [[d["n_missing"], d["q1"], d["median"], d["q3"]]],
            "ordered": False,
        }
    raise SystemExit(f"{qid}: no reference builder")


def refresh_stats(path: Path, ref: dict, today: datetime.date | None = None) -> list[str]:
    """Write reference_stats into every stats question that has a reference_analysis."""
    raw = load_raw(path)
    today = today or datetime.date.today()
    lines = []
    for q in raw["questions"]:
        if q["type"] != "stats" or not q.get("reference_analysis"):
            continue
        built = stats_reference(q["id"], q["reference_analysis"]["params"], ref)
        q["reference_stats"] = {
            "via": "analytics_runner/reference.py, run in the sandbox image: scipy.stats called "
            "directly and every Mann-Whitney p-value recomputed from the textbook formula "
            "(shares no code with analytics_runner/analyses.py)",
            "scipy": ref["scipy"],
            "extract_sha256": ref["extract_sha256"],
            "n_runs": ref["n_runs"],
            "checked_on": today.isoformat(),
            **built,
        }
        lines.append(f"{q['id']}: {len(built['checks'])} check(s) {built['answer_rows'][:2]}")
    path.write_text(
        yaml.dump(raw, Dumper=_Dumper, sort_keys=False, allow_unicode=True, width=100_000),
        encoding="utf-8",
    )
    return lines


def main() -> None:
    from rflens.config import get_agent_settings

    if "--stats" in sys.argv[1:]:
        from rflens.analytics.reference import run_reference

        for line in refresh_stats(GOLDEN_PATH, run_reference()):
            print(line)
        problem_list = problems(load_golden())
        if problem_list:
            print("\n".join(problem_list), file=sys.stderr)
            raise SystemExit(1)
        print(f"stored stats references in {GOLDEN_PATH}")
        return
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
