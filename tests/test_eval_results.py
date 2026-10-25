"""eval/RESULTS.md: generated golden-set counts, the SQL-run table and the retrieval-run table."""

import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from rflens.eval.golden import GoldenQuestion, load_golden
from rflens.eval.results_log import (
    COLUMNS,
    COUNTS_END,
    COUNTS_START,
    DOCS_COLUMNS,
    DOCS_TITLE,
    RESULTS_PATH,
    SQL_TITLE,
    append_docs_rows,
    append_row,
    git_commit,
    golden_counts_block,
    parse_table,
    refresh_counts,
)

WHEN = datetime(2026, 10, 25, 9, 30, tzinfo=UTC)


def gq(qid: str, kind: str, author: str, status: str = "reviewed") -> GoldenQuestion:
    return GoldenQuestion(id=qid, type=kind, question="?", status=status, author=author)


SAMPLE = [
    gq("s1", "sql", "human"),
    gq("s2", "sql", "assistant", "draft"),
    gq("d1", "doc", "human"),
    gq("d2", "doc", "assistant", "draft"),
    gq("d3", "doc", "assistant", "draft"),
    gq("r1", "refuse", "human"),
]


def sql_kwargs(**kw) -> dict:
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
        when=WHEN,
        questions=SAMPLE,
    )
    return base | kw


GROUPS = {
    "human": {"vector": (8, 9, 10), "keyword": (0, 2, 10), "hybrid": (6, 7, 10)},
    "assistant": {"vector": (5, 7, 12), "keyword": (0, 0, 12), "hybrid": (2, 3, 12)},
}


# --- the generated golden-set header ----------------------------------------------------------


def test_counts_block_counts_per_type_and_author() -> None:
    block = golden_counts_block(SAMPLE)
    assert "6 questions" in block
    assert "| sql | 1 | 1 | 2 | 1 | 1 |" in block
    assert "| doc | 1 | 2 | 3 | 2 | 1 |" in block
    assert "| mixed | 0 | 0 | 0 | 0 | 0 |" in block
    assert "| refuse | 1 | 0 | 1 | 0 | 1 |" in block
    assert "| total | 3 | 3 | 6 | 3 | 3 |" in block


def test_the_header_of_the_checked_in_file_is_current() -> None:
    """Fails when eval/golden.yaml changed and `make results-header` was not run."""
    text = RESULTS_PATH.read_text(encoding="utf-8")
    start, end = text.index(COUNTS_START), text.index(COUNTS_END)
    shown = text[start + len(COUNTS_START) : end].strip()
    assert shown == golden_counts_block(load_golden())


def test_the_golden_set_has_the_expected_shape() -> None:
    questions = load_golden()
    by = {(q.type, q.author) for q in questions}
    assert {t for t, _ in by} == {"sql", "doc", "mixed", "refuse"}
    assert all(q.author in ("human", "assistant") for q in questions)
    assert len(questions) >= 50


# --- file creation and row placement ----------------------------------------------------------


def test_first_row_creates_the_file_with_counts_and_both_tables(tmp_path: Path) -> None:
    path = tmp_path / "RESULTS.md"
    row = append_row(path=path, **sql_kwargs())
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# Eval results")
    assert COUNTS_START in text and "| total | 3 | 3 | 6 | 3 | 3 |" in text
    assert SQL_TITLE in text and DOCS_TITLE in text
    assert "| " + " | ".join(COLUMNS) + " |" in text
    assert "| " + " | ".join(DOCS_COLUMNS) + " |" in text
    assert parse_table(path, SQL_TITLE) == [
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
    assert parse_table(path, DOCS_TITLE) == [] and row.endswith("|\n")


def test_rows_go_to_their_own_table_newest_last(tmp_path: Path) -> None:
    path = tmp_path / "RESULTS.md"
    append_row(path=path, **sql_kwargs(run_id="eval-sql-1"))
    append_docs_rows(
        path=path,
        run_id="eval-docs-1",
        commit="abc1234",
        groups=GROUPS,
        when=WHEN,
        questions=SAMPLE,
    )
    append_row(path=path, **sql_kwargs(run_id="eval-sql-2", sql_matched=15, note=""))
    append_docs_rows(
        path=path,
        run_id="eval-docs-2",
        commit="abc1234",
        groups=GROUPS,
        when=WHEN,
        questions=SAMPLE,
    )
    sql_rows = parse_table(path, SQL_TITLE)
    docs_rows = parse_table(path, DOCS_TITLE)
    assert [r[1] for r in sql_rows] == ["eval-sql-1", "eval-sql-2"]
    assert [r[1] for r in docs_rows] == ["eval-docs-1", "eval-docs-1", "eval-docs-2", "eval-docs-2"]
    assert sql_rows[1][5] == "15/15" and sql_rows[1][9] == ""
    text = path.read_text(encoding="utf-8")
    assert text.index(SQL_TITLE) < text.index("eval-sql-2") < text.index(DOCS_TITLE)


def test_docs_rows_give_one_row_per_author_group_with_recall_cells(tmp_path: Path) -> None:
    path = tmp_path / "RESULTS.md"
    rows = append_docs_rows(
        path=path,
        run_id="eval-docs-x",
        commit="abc1234",
        groups=GROUPS,
        note="132 searches",
        when=WHEN,
        questions=SAMPLE,
    )
    assert len(rows) == 2
    human, assistant = parse_table(path, DOCS_TITLE)
    assert human[3] == "human doc questions (10)" and assistant[3] == "assistant doc questions (12)"
    assert human[4:7] == ["8/10 / 9/10", "0/10 / 2/10", "6/10 / 7/10"]
    assert assistant[4:7] == ["5/12 / 7/12", "0/12 / 0/12", "2/12 / 3/12"]
    assert human[7] == "132 searches"


def test_each_append_refreshes_the_counts(tmp_path: Path) -> None:
    path = tmp_path / "RESULTS.md"
    append_row(path=path, **sql_kwargs())
    more = [*SAMPLE, gq("m1", "mixed", "assistant", "draft")]
    append_row(path=path, **sql_kwargs(run_id="eval-sql-2", questions=more))
    text = path.read_text(encoding="utf-8")
    assert "7 questions" in text and "| mixed | 0 | 1 | 1 | 1 | 0 |" in text
    assert text.count(COUNTS_START) == 1  # replaced, not duplicated
    refresh_counts(path, SAMPLE)
    assert "6 questions" in path.read_text(encoding="utf-8")
    assert len(parse_table(path, SQL_TITLE)) == 2  # refreshing the header keeps the rows


def test_a_note_cannot_break_the_table(tmp_path: Path) -> None:
    path = tmp_path / "RESULTS.md"
    append_row(path=path, **sql_kwargs(note="a | b\nsecond line"))
    (row,) = parse_table(path, SQL_TITLE)
    assert len(row) == len(COLUMNS) and row[-1] == "a / b second line"


def test_the_first_version_of_the_file_is_migrated_keeping_its_rows(tmp_path: Path) -> None:
    path = tmp_path / "RESULTS.md"
    old_rows = [
        "| 2026-10-24 17:18 | eval-sql-old1 | abc | m / standard | 17 (14 sql + 3 refuse) | 14/14 "
        "| 3/3 | 18 | $0.024675 | first |",
        "| 2026-10-24 17:39 | eval-sql-old2 | abd | m / standard | 18 (15 sql + 3 refuse) | 14/15 "
        "| 3/3 | 18 | $0.026761 | second |",
    ]
    path.write_text(
        "# Eval results\n\nold text\n\n| date (UTC) | run_id | commit | model / tier | golden set "
        "| execution accuracy | refusal correctness | calls | cost | note |\n"
        "|---|---|---|---|---|---|---|---|---|---|\n" + "\n".join(old_rows) + "\n",
        encoding="utf-8",
    )
    append_row(path=path, **sql_kwargs(run_id="eval-sql-new"))
    assert [r[1] for r in parse_table(path, SQL_TITLE)] == [
        "eval-sql-old1",
        "eval-sql-old2",
        "eval-sql-new",
    ]
    assert DOCS_TITLE in path.read_text(encoding="utf-8")


def test_a_file_without_a_final_newline_still_gets_a_clean_row(tmp_path: Path) -> None:
    path = tmp_path / "RESULTS.md"
    append_row(path=path, **sql_kwargs())
    path.write_text(path.read_text(encoding="utf-8").rstrip("\n"), encoding="utf-8")
    append_row(path=path, **sql_kwargs(run_id="eval-sql-2"))
    assert len(parse_table(path, SQL_TITLE)) == 2


# --- git --------------------------------------------------------------------------------------


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


# --- the checked-in history -------------------------------------------------------------------


def test_the_checked_in_log_keeps_the_sql_runs_and_has_the_first_retrieval_run() -> None:
    sql_rows = parse_table(RESULTS_PATH, SQL_TITLE)
    by_run = {r[1]: r for r in sql_rows}
    first, second, third = (
        by_run[i] for i in ("eval-sql-9cf4bce9", "eval-sql-6d590871", "eval-sql-861790a2")
    )
    assert first[4:9] == ["17 (14 sql + 3 refuse)", "14/14", "3/3", "18", "$0.024675"]
    assert "BEFORE the lease fix" in first[9]
    assert second[4:9] == ["18 (15 sql + 3 refuse)", "14/15", "3/3", "18", "$0.026761"]
    assert third[4:9] == ["20 (17 sql + 3 refuse)", "17/17", "3/3", "21", "$0.031850"]
    assert sql_rows.index(first) < sql_rows.index(second) < sql_rows.index(third)
    assert all(len(r) == len(COLUMNS) for r in sql_rows)
    docs_rows = parse_table(RESULTS_PATH, DOCS_TITLE)
    assert len(docs_rows) >= 2 and all(len(r) == len(DOCS_COLUMNS) for r in docs_rows)
    first_run = [r for r in docs_rows if r[1] == docs_rows[0][1]]
    assert [r[3] for r in first_run] == ["human doc questions (10)", "assistant doc questions (12)"]
