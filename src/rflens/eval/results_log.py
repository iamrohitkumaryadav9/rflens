"""eval/RESULTS.md: the golden-set counts and one row per eval run.

The file has a generated header (golden-set questions per type and author, from eval/golden.yaml;
`make results-header` refreshes it, every appended row refreshes it, and a test fails when it is
stale) and two tables, each with its rows appended automatically:

- SQL agent runs (`make eval-sql`): date (UTC), run_id (the gateway run, also in llm_calls and
  data/eval/*.json), commit (HEAD when the run started, "+dirty" when the working tree had
  uncommitted changes), model and tier, golden set (questions run), execution accuracy (sql
  questions matched), refusal correctness (refuse questions matched), LLM calls, cost, note.
- End-to-end runs (`make eval-e2e`): date, run_id, commit, model and tier, questions run of the
  golden set, correct per type (doc: answer correct / evidence recall), correct per author
  group, calls, cost, note.
- Retrieval runs (`make eval-docs`): date, run_id, commit, the questions (author group and count),
  recall@5 for vector, keyword and hybrid search without / with the spec filter, note. Two rows
  per run (human and assistant doc questions). Free: no LLM is called.
"""

import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_PATH = REPO_ROOT / "eval" / "RESULTS.md"

SQL_TITLE = "## SQL agent runs (make eval-sql)"
DOCS_TITLE = "## Retrieval runs (make eval-docs)"
E2E_TITLE = "## End-to-end runs (make eval-e2e)"
COUNTS_START = "<!-- golden-counts:start (generated: make results-header) -->"
COUNTS_END = "<!-- golden-counts:end -->"

COLUMNS = [
    "date (UTC)",
    "run_id",
    "commit",
    "model / tier",
    "golden set",
    "execution accuracy",
    "refusal correctness",
    "calls",
    "cost",
    "note",
]
DOCS_COLUMNS = [
    "date (UTC)",
    "run_id",
    "commit",
    "questions",
    "vector (no filter / spec filter)",
    "keyword (no filter / spec filter)",
    "hybrid (no filter / spec filter)",
    "note",
]

E2E_COLUMNS = [
    "date (UTC)",
    "run_id",
    "commit",
    "model / tier",
    "questions run",
    "sql",
    "doc (answer / evidence recall)",
    "mixed",
    "refuse",
    "human",
    "assistant",
    "calls",
    "cost",
    "note",
]

INTRO = """\
# Eval results

Rows are appended automatically, newest last (`NOTE=` / `--note` fills the last column). The golden
set grew between runs, so compare rows through their question counts. "commit" is HEAD when the run
started, "+dirty" if the working tree had uncommitted changes.

"""


def _table_header(columns: list[str]) -> str:
    return "| " + " | ".join(columns) + " |\n|" + "---|" * len(columns) + "\n"


# --- golden-set counts -----------------------------------------------------------------------


def golden_counts_block(questions: list | None = None) -> str:
    """The generated header: golden questions per type and author, and per status."""
    if questions is None:
        from rflens.eval.golden import load_golden

        questions = load_golden()
    lines = [
        f"**Golden set** (eval/golden.yaml): {len(questions)} questions. author = who wrote the "
        "question (human: edited and reviewed by the project owner).",
        "",
        "| type | human | assistant | total | draft | reviewed |",
        "|---|---|---|---|---|---|",
    ]
    for kind in ("sql", "doc", "mixed", "refuse", None):
        group = [q for q in questions if kind is None or q.type == kind]
        cells = [
            sum(q.author == "human" for q in group),
            sum(q.author == "assistant" for q in group),
            len(group),
            sum(q.status == "draft" for q in group),
            sum(q.status == "reviewed" for q in group),
        ]
        lines.append(f"| {kind or 'total'} | " + " | ".join(str(c) for c in cells) + " |")
    return "\n".join(lines)


def _counts_section(questions: list | None = None) -> str:
    return f"{COUNTS_START}\n{golden_counts_block(questions)}\n{COUNTS_END}\n"


def _template(
    sql_rows: str = "", docs_rows: str = "", questions: list | None = None, e2e_rows: str = ""
) -> str:
    return (
        INTRO
        + _counts_section(questions)
        + f"\n{SQL_TITLE}\n\n"
        + _table_header(COLUMNS)
        + sql_rows
        + f"\n{DOCS_TITLE}\n\n"
        + _table_header(DOCS_COLUMNS)
        + docs_rows
        + f"\n{E2E_TITLE}\n\n"
        + _table_header(E2E_COLUMNS)
        + e2e_rows
    )


def _replace_counts(text: str, questions: list | None = None) -> str:
    start, end = text.index(COUNTS_START), text.index(COUNTS_END) + len(COUNTS_END) + 1
    return text[:start] + _counts_section(questions) + text[end:]


def _migrate_old_format(text: str) -> str:
    """The first version of the file had one SQL table and no counts; keep its rows."""
    rows = [
        ln + "\n"
        for ln in text.splitlines()
        if ln.startswith("| ") and not ln.startswith("| date (UTC)")
    ]
    return _template(sql_rows="".join(rows))


def _load(path: Path) -> str:
    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        return _template()
    text = path.read_text(encoding="utf-8")
    if DOCS_TITLE not in text:
        return _migrate_old_format(text)
    if E2E_TITLE not in text:  # files written before the end-to-end table existed
        text = text.rstrip("\n") + f"\n\n{E2E_TITLE}\n\n" + _table_header(E2E_COLUMNS)
    return text


def _insert_row(text: str, title: str, row: str) -> str:
    """Add `row` after the last row of the table under `title`."""
    lines = text.splitlines(keepends=True)
    at = next(i for i, ln in enumerate(lines) if ln.strip() == title)
    sep = next(i for i in range(at, len(lines)) if lines[i].startswith("|---"))
    last = sep
    while last + 1 < len(lines) and lines[last + 1].startswith("|"):
        last += 1
    if not lines[last].endswith("\n"):
        lines[last] += "\n"
    lines.insert(last + 1, row)
    return "".join(lines)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def refresh_counts(path: Path = RESULTS_PATH, questions: list | None = None) -> None:
    _write(path, _replace_counts(_load(path), questions))


# --- rows --------------------------------------------------------------------------------------


def git_commit(repo: Path = REPO_ROOT) -> str:
    """Short HEAD hash, with '+dirty' if there are uncommitted changes; 'unknown' without git."""
    try:
        head = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{head}+dirty" if status else head


def _cell(text: str) -> str:
    return text.replace("|", "/").replace("\n", " ").strip()


def _row(cells: list[str]) -> str:
    return "| " + " | ".join(_cell(c) for c in cells) + " |\n"


def append_row(
    *,
    run_id: str,
    commit: str,
    model: str,
    tier: str,
    sql_matched: int,
    sql_total: int,
    refuse_matched: int,
    refuse_total: int,
    calls: int,
    cost_usd: float,
    note: str = "",
    when: datetime | None = None,
    path: Path = RESULTS_PATH,
    questions: list | None = None,
) -> str:
    """Append one SQL-agent run (creating the file if needed); returns the row."""
    when = when or datetime.now(UTC)
    row = _row(
        [
            when.strftime("%Y-%m-%d %H:%M"),
            run_id,
            commit,
            f"{model} / {tier}",
            f"{sql_total + refuse_total} ({sql_total} sql + {refuse_total} refuse)",
            f"{sql_matched}/{sql_total}",
            f"{refuse_matched}/{refuse_total}",
            str(calls),
            f"${cost_usd:.6f}",
            note,
        ]
    )
    text = _replace_counts(_load(path), questions)
    _write(path, _insert_row(text, SQL_TITLE, row))
    return row


def append_docs_rows(
    *,
    run_id: str,
    commit: str,
    groups: dict[str, dict[str, tuple[int, int, int]]],
    note: str = "",
    when: datetime | None = None,
    path: Path = RESULTS_PATH,
    questions: list | None = None,
) -> list[str]:
    """Append one retrieval-run row per author group. `groups` maps a group label (for example
    'human') to {mode: (hits without filter, hits with filter, question count)}."""
    when = when or datetime.now(UTC)
    text = _replace_counts(_load(path), questions)
    rows = []
    for label, modes in groups.items():
        n = next(iter(modes.values()))[2]
        cells = [when.strftime("%Y-%m-%d %H:%M"), run_id, commit, f"{label} doc questions ({n})"]
        for mode in ("vector", "keyword", "hybrid"):
            plain, filtered, total = modes[mode]
            cells.append(f"{plain}/{total} / {filtered}/{total}")
        cells.append(note)
        row = _row(cells)
        text = _insert_row(text, DOCS_TITLE, row)
        rows.append(row)
    _write(path, text)
    return rows


def append_e2e_row(
    *,
    run_id: str,
    commit: str,
    model: str,
    tier: str,
    questions_run: int,
    questions_total: int,
    cells: dict[str, str],
    calls: int,
    cost_usd: float,
    note: str = "",
    when: datetime | None = None,
    path: Path = RESULTS_PATH,
    questions: list | None = None,
) -> str:
    """Append one end-to-end run. `cells` holds the result cells keyed sql, doc, mixed, refuse,
    human, assistant (each already formatted, for example '12/19')."""
    when = when or datetime.now(UTC)
    row = _row(
        [
            when.strftime("%Y-%m-%d %H:%M"),
            run_id,
            commit,
            f"{model} / {tier}",
            f"{questions_run}/{questions_total}",
            cells["sql"],
            cells["doc"],
            cells["mixed"],
            cells["refuse"],
            cells["human"],
            cells["assistant"],
            str(calls),
            f"${cost_usd:.6f}",
            note,
        ]
    )
    text = _replace_counts(_load(path), questions)
    _write(path, _insert_row(text, E2E_TITLE, row))
    return row


def parse_table(path: Path, title: str) -> list[list[str]]:
    """The data rows of the table under `title`, as lists of cells (for tests and tools)."""
    lines = path.read_text(encoding="utf-8").splitlines()
    at = next(i for i, ln in enumerate(lines) if ln.strip() == title)
    sep = next(i for i in range(at, len(lines)) if lines[i].startswith("|---"))
    rows = []
    for ln in lines[sep + 1 :]:
        if not ln.startswith("|"):
            break
        rows.append([c.strip() for c in ln.strip()[1:-1].split("|")])  # cells never hold "|"
    return rows


def main() -> None:
    refresh_counts()
    print(f"golden-set counts refreshed in {RESULTS_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
