"""eval/RESULTS.md: the row log that `make eval-sql` appends to, and the checked-in history."""

import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from rflens.eval.results_log import COLUMNS, HEADER, RESULTS_PATH, append_row, git_commit


def row_kwargs(**kw) -> dict:
    base = dict(
        run_id="eval-sql-abc12345",
        commit="abc1234",
        model="gemini-3.1-flash-lite",
        tier="standard",
        sql_matched=14,
        sql_total=15,
        refuse_matched=3,
        refuse_total=3,
        calls=18,
        cost_usd=0.026761,
        note="a note",
        when=datetime(2026, 10, 25, 9, 30, tzinfo=UTC),
    )
    return base | kw


def table_rows(path: Path) -> list[list[str]]:
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.startswith("|")]
    return [[c.strip() for c in ln.strip("|").split("|")] for ln in lines[2:]]  # skip header rows


def test_first_row_creates_the_file_with_its_header(tmp_path: Path) -> None:
    path = tmp_path / "RESULTS.md"
    row = append_row(path=path, **row_kwargs())
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# Eval results") and HEADER in text
    assert "| " + " | ".join(COLUMNS) + " |" in text
    assert [ln for ln in text.splitlines() if ln.startswith("|---")] == ["|" + "---|" * 10]
    assert table_rows(path) == [
        [
            "2026-10-25 09:30",
            "eval-sql-abc12345",
            "abc1234",
            "gemini-3.1-flash-lite / standard",
            "18 (15 sql + 3 refuse)",
            "14/15",
            "3/3",
            "18",
            "$0.026761",
            "a note",
        ]
    ]
    assert row.endswith("|\n")


def test_later_rows_are_appended_and_the_header_is_not_repeated(tmp_path: Path) -> None:
    path = tmp_path / "RESULTS.md"
    append_row(path=path, **row_kwargs(run_id="eval-sql-1"))
    append_row(path=path, **row_kwargs(run_id="eval-sql-2", sql_matched=15, note=""))
    append_row(path=path, **row_kwargs(run_id="eval-sql-3", tier="flex"))
    rows = table_rows(path)
    assert [r[1] for r in rows] == ["eval-sql-1", "eval-sql-2", "eval-sql-3"]  # newest last
    assert rows[1][5] == "15/15" and rows[1][9] == "" and rows[2][3].endswith("/ flex")
    assert path.read_text(encoding="utf-8").count("# Eval results") == 1


def test_a_note_cannot_break_the_table(tmp_path: Path) -> None:
    path = tmp_path / "RESULTS.md"
    append_row(path=path, **row_kwargs(note="a | b\nsecond line"))
    (row,) = table_rows(path)
    assert len(row) == len(COLUMNS) and row[-1] == "a / b second line"


def test_a_file_without_a_final_newline_still_gets_a_clean_row(tmp_path: Path) -> None:
    path = tmp_path / "RESULTS.md"
    path.write_text(HEADER.rstrip("\n"), encoding="utf-8")
    append_row(path=path, **row_kwargs())
    assert len(table_rows(path)) == 1


def git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *args],
        check=True,
        capture_output=True,
    )


def test_git_commit_reports_head_and_marks_a_dirty_tree(tmp_path: Path) -> None:
    git(tmp_path, "init", "-q")
    (tmp_path / "a.txt").write_text("one", encoding="utf-8")
    git(tmp_path, "add", "a.txt")
    git(tmp_path, "commit", "-q", "-m", "first")
    clean = git_commit(tmp_path)
    assert re.fullmatch(r"[0-9a-f]{7,}", clean)
    (tmp_path / "a.txt").write_text("two", encoding="utf-8")
    assert git_commit(tmp_path) == clean + "+dirty"
    assert git_commit(tmp_path / "not-a-repo") == "unknown"


def test_the_checked_in_log_has_the_two_runs_so_far() -> None:
    rows = table_rows(RESULTS_PATH)
    by_run = {r[1]: r for r in rows}
    first, second = by_run["eval-sql-9cf4bce9"], by_run["eval-sql-6d590871"]
    assert first[4:9] == ["17 (14 sql + 3 refuse)", "14/14", "3/3", "18", "$0.024675"]
    assert "BEFORE the lease fix" in first[9]
    assert second[4:9] == ["18 (15 sql + 3 refuse)", "14/15", "3/3", "18", "$0.026761"]
    assert "sql-06" in second[9] and "licensee_frn AND licensee_name" in second[9]
    assert all(len(r) == len(COLUMNS) for r in rows)
    assert rows.index(first) < rows.index(second)  # oldest first
