"""eval/RESULTS.md: one row per eval run, appended automatically by `make eval-sql`.

Columns: date (UTC), run_id (the gateway run, also in llm_calls and data/eval/*.json), commit (HEAD
at the time of the run, "+dirty" when the working tree had uncommitted changes), model and tier,
golden set (questions run: sql + refuse), execution accuracy (sql questions matched), refusal
correctness (refuse questions matched), LLM calls, cost, and a short note.
"""

import subprocess
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_PATH = REPO_ROOT / "eval" / "RESULTS.md"

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

HEADER = """\
# Eval results

One row per `make eval-sql` run, newest last; rows are appended automatically (`--note` / `NOTE=`
adds the last column). Execution accuracy and refusal correctness are matches over the questions
of that run; the golden set changed between runs, so compare rows with their "golden set" column.
"commit" is HEAD when the run started, "+dirty" if the working tree had uncommitted changes.

"""
HEADER += "| " + " | ".join(COLUMNS) + " |\n" + "|" + "---|" * len(COLUMNS) + "\n"


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
) -> str:
    """Append one row (creating the file with its header if needed); returns the row."""
    when = when or datetime.now(UTC)
    cells = [
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
    row = "| " + " | ".join(_cell(c) for c in cells) + " |\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        path.write_text(HEADER, encoding="utf-8")
    existing = path.read_text(encoding="utf-8")
    if not existing.endswith("\n"):
        row = "\n" + row
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(row)
    return row
