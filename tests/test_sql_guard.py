"""sql_guard: the static guardrails of sql-mcp. Pure parsing, no database.

Every unsafe query is rejected with a specific, stable reason code; every safe query passes and
comes back with a LIMIT. The database-level checks (roles, cost gate, timeout, log) are in
test_sql_mcp.py.
"""

import pytest

from rflens.mcp.sql_guard import (
    ALLOWED_RELATIONS,
    ALLOWED_TABLES,
    ALLOWED_VIEWS,
    Rejection,
    check_query,
)

INSERT_HD = "INSERT INTO uls_hd (service, usi, line_no) VALUES ('x', 1, 1)"

# (sql, reason code)
UNSAFE = [
    # --- not a SELECT -------------------------------------------------------------------------
    ("DELETE FROM uls_hd", "not_select"),
    (INSERT_HD, "not_select"),
    ("UPDATE uls_hd SET call_sign = 'x'", "not_select"),
    ("DROP TABLE uls_hd", "not_select"),
    ("TRUNCATE uls_hd", "not_select"),
    ("CREATE TABLE x AS SELECT 1", "not_select"),
    ("ALTER TABLE uls_hd ADD COLUMN x int", "not_select"),
    ("GRANT ALL ON uls_hd TO PUBLIC", "not_select"),
    ("COPY uls_hd TO STDOUT", "not_select"),
    ("SET statement_timeout = 0", "not_select"),
    ("EXPLAIN SELECT 1", "not_select"),
    ("VACUUM uls_hd", "not_select"),
    # --- more than one statement ----------------------------------------------------------------
    ("SELECT 1; SELECT 2", "multiple_statements"),
    ("SELECT 1; DROP TABLE uls_hd", "multiple_statements"),
    ("SELECT * FROM uls_hd; DELETE FROM uls_hd", "multiple_statements"),
    # --- comment tricks -------------------------------------------------------------------------
    ("SELECT 1 /* ; */ ; DELETE FROM uls_hd", "multiple_statements"),
    ("SELECT 1 -- harmless\n; DROP TABLE uls_hd", "multiple_statements"),
    ("DELETE/**/FROM uls_hd", "not_select"),
    ("/* SELECT */ DELETE FROM uls_hd", "not_select"),
    # --- case tricks ----------------------------------------------------------------------------
    ("dElEtE FrOm uls_hd", "not_select"),
    ("SeLeCt * FrOm UlS_eN", "table_not_allowed"),
    ("SELECT * FROM PG_USER", "system_catalog"),
    # --- DML, INTO and locking hidden inside a SELECT -------------------------------------------
    ("WITH d AS (DELETE FROM uls_hd RETURNING *) SELECT * FROM d", "forbidden_construct"),
    (f"WITH i AS ({INSERT_HD} RETURNING usi) SELECT * FROM i", "forbidden_construct"),
    (
        "WITH u AS (UPDATE uls_hd SET auction_id = 1 RETURNING usi) SELECT * FROM u",
        "forbidden_construct",
    ),
    ("SELECT * INTO newt FROM uls_hd", "forbidden_construct"),
    ("SELECT * FROM uls_hd FOR UPDATE", "forbidden_construct"),
    ("SELECT * FROM uls_hd FOR SHARE", "forbidden_construct"),
    # --- uls_en: direct, quoted, schema-qualified, and every nesting ------------------------------
    ("SELECT * FROM uls_en", "table_not_allowed"),
    ('SELECT * FROM "uls_en"', "table_not_allowed"),
    ("SELECT * FROM public.uls_en", "table_not_allowed"),
    ('SELECT * FROM public."uls_en"', "table_not_allowed"),
    ("WITH x AS (SELECT * FROM uls_en) SELECT * FROM x", "table_not_allowed"),
    ("WITH uls_en AS (SELECT * FROM uls_en) SELECT * FROM uls_en", "table_not_allowed"),
    ("SELECT (SELECT count(*) FROM uls_en)", "table_not_allowed"),
    ("SELECT * FROM uls_hd WHERE usi IN (SELECT usi FROM uls_en)", "table_not_allowed"),
    ("SELECT usi FROM uls_hd UNION SELECT usi FROM uls_en", "table_not_allowed"),
    ("SELECT * FROM uls_hd h JOIN uls_en e USING (service, usi)", "table_not_allowed"),
    (
        "SELECT * FROM uls_hd h JOIN LATERAL (SELECT * FROM uls_en e WHERE e.usi = h.usi) x "
        "ON true",
        "table_not_allowed",
    ),
    ("SELECT * FROM uls_hd, ONLY uls_en", "table_not_allowed"),
    ("TABLE uls_en", "not_select"),  # PostgreSQL shorthand for SELECT * FROM uls_en
    ('SELECT * FROM U&"uls\\005fen"', "parse_error"),  # Unicode-escaped identifier spelling uls_en
    # --- other tables that are not on the allow-list --------------------------------------------
    ("SELECT * FROM chunks", "table_not_allowed"),
    ('SELECT * FROM "chunks"', "table_not_allowed"),
    ("SELECT * FROM public.documents", "table_not_allowed"),
    ("SELECT * FROM alembic_version", "table_not_allowed"),
    ("SELECT * FROM sql_query_log", "table_not_allowed"),
    ('SELECT * FROM "ULS_HD"', "table_not_allowed"),  # quoted names are case-sensitive
    ("SELECT * FROM other_schema.uls_hd", "table_not_allowed"),
    ('SELECT * FROM "PUBLIC".uls_hd', "table_not_allowed"),
    ("SELECT * FROM rflens.public.uls_hd", "table_not_allowed"),
    # --- system catalogs ------------------------------------------------------------------------
    ("SELECT * FROM pg_catalog.pg_user", "system_catalog"),
    ("SELECT * FROM pg_catalog.pg_authid", "system_catalog"),
    ("SELECT * FROM pg_shadow", "system_catalog"),
    ("SELECT * FROM pg_class", "system_catalog"),
    ("SELECT * FROM information_schema.tables", "system_catalog"),
    # --- functions in FROM ----------------------------------------------------------------------
    ("SELECT * FROM generate_series(1, 10)", "table_function"),
    ("SELECT * FROM dblink('host=x', 'select 1') AS t(a int)", "table_function"),
    ("SELECT * FROM pg_ls_dir('/')", "table_function"),
    # --- denied functions -----------------------------------------------------------------------
    ("SELECT pg_sleep(10)", "function_denied"),
    ("SELECT PG_SLEEP(10)", "function_denied"),
    ("SELECT pg_catalog.pg_sleep(1)", "function_denied"),
    ("SELECT * FROM uls_hd WHERE pg_sleep(1) IS NULL", "function_denied"),
    ("SELECT pg_read_file('/etc/passwd')", "function_denied"),
    ("SELECT lo_import('/etc/passwd')", "function_denied"),
    ("SELECT set_config('statement_timeout', '0', false)", "function_denied"),
    ("SELECT current_setting('is_superuser')", "function_denied"),
    ("SELECT dblink_exec('x', 'y')", "function_denied"),
    ("SELECT query_to_xml('select * from uls_en', true, true, '')", "function_denied"),
    ("SELECT nextval('sql_query_log_id_seq')", "function_denied"),
    ("SELECT to_regclass('uls_en')", "function_denied"),
    ("SELECT has_table_privilege('uls_en', 'select')", "function_denied"),
    # --- limits that cannot be honoured ---------------------------------------------------------
    ("SELECT * FROM uls_hd FETCH FIRST 5 ROWS ONLY", "unsupported_limit"),
    # --- empty ----------------------------------------------------------------------------------
    ("", "empty"),
    ("   ", "empty"),
    (";", "empty"),
    ("-- only a comment", "empty"),
]

SAFE = [
    "SELECT 1",
    "SELECT 1;",
    "SELECT count(*) FROM uls_hd",
    'SELECT * FROM "uls_hd"',
    "SELECT * FROM public.uls_hd",
    "select * from V_ULS_LICENCES",
    "SELECT * FROM v_uls_licences WHERE license_status = 'A'",
    "SELECT call_sign, frequency_assigned FROM v_uls_frequencies WHERE frequency_assigned > 869",
    "SELECT label, count(*) FROM v_secom_values WHERE feature_idx = 59 GROUP BY label",
    "SELECT * FROM fcc_grantees g JOIN fcc_test_firms t ON t.country = g.country",
    "WITH a AS (SELECT usi FROM uls_hd) SELECT count(*) FROM a",
    "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r WHERE n < 5) SELECT * FROM r",
    "SELECT usi FROM uls_hd UNION SELECT usi FROM uls_fr",
    "SELECT usi, row_number() OVER (PARTITION BY service ORDER BY usi) FROM uls_hd",
    "SELECT * FROM uls_hd WHERE usi IN (SELECT usi FROM uls_fr)",
    "SELECT * FROM secom_runs a JOIN secom_measurements m USING (run_id)",
    # words that look dangerous but are data, aliases or comments, not statements or tables
    "SELECT 'DELETE FROM uls_hd; DROP TABLE x' AS note",
    "SELECT * FROM uls_hd WHERE call_sign = 'uls_en'",
    "SELECT call_sign AS pg_sleep FROM uls_hd",
    "SELECT 1 /* ; DROP TABLE uls_hd; */",
    "SELECT 1 -- ; DROP TABLE uls_hd",
    "SELECT * FROM uls_hd WHERE call_sign LIKE 'KN%'",
    "SELECT '12:30'::time",
]


def test_the_catalogue_is_at_least_25_unsafe_queries() -> None:
    assert len(UNSAFE) >= 25
    codes = {code for _, code in UNSAFE}
    assert codes >= {
        "not_select", "multiple_statements", "forbidden_construct", "table_not_allowed",
        "system_catalog", "table_function", "function_denied", "unsupported_limit", "empty",
    }  # fmt: skip


@pytest.mark.parametrize(("sql", "code"), UNSAFE, ids=[f"{c}:{s[:50]!r}" for s, c in UNSAFE])
def test_unsafe_query_is_rejected_with_the_right_reason(sql: str, code: str) -> None:
    with pytest.raises(Rejection) as rejected:
        check_query(sql)
    assert rejected.value.code == code, rejected.value.message
    assert rejected.value.message


@pytest.mark.parametrize("sql", SAFE)
def test_safe_query_passes_and_gets_a_limit(sql: str) -> None:
    checked = check_query(sql)
    assert 1 <= checked.limit <= 1000
    assert "LIMIT" in checked.sql.upper()


def test_the_three_explicit_uls_en_cases_name_the_table() -> None:
    for sql in (
        "SELECT * FROM uls_en",
        'SELECT * FROM "uls_en"',
        "WITH e AS (SELECT * FROM uls_en) SELECT * FROM e",
    ):
        with pytest.raises(Rejection) as rejected:
            check_query(sql)
        assert rejected.value.code == "table_not_allowed" and "uls_en" in rejected.value.message


def test_the_allow_list_is_ten_tables_and_three_views_without_uls_en() -> None:
    # spec_operating_bands (migration 0013) is the tenth table
    assert len(ALLOWED_TABLES) == 10 and "spec_operating_bands" in ALLOWED_TABLES
    assert len(ALLOWED_VIEWS) == 3
    assert "uls_en" not in ALLOWED_RELATIONS
    for hidden in ("chunks", "documents", "alembic_version", "sql_query_log"):
        assert hidden not in ALLOWED_RELATIONS


# --- LIMIT ----------------------------------------------------------------------------------------


def test_limit_is_added_when_absent() -> None:
    checked = check_query("SELECT * FROM uls_hd")
    assert checked.limit == 100 and checked.sql.endswith("LIMIT 100")
    assert "added" in checked.notice and "100" in checked.notice


def test_a_limit_within_the_maximum_is_kept_silently() -> None:
    checked = check_query("SELECT * FROM uls_hd LIMIT 10")
    assert checked.limit == 10 and checked.notice is None and checked.sql.endswith("LIMIT 10")
    assert check_query("select * from uls_hd limit 1000").limit == 1000


@pytest.mark.parametrize("tail", ["LIMIT 5000", "LIMIT 1001", "LIMIT ALL", "LIMIT NULL"])
def test_a_limit_above_the_maximum_is_lowered(tail: str) -> None:
    checked = check_query(f"SELECT * FROM uls_hd {tail}")
    assert checked.limit == 1000 and checked.sql.endswith("LIMIT 1000")
    assert "lowered" in checked.notice


def test_limit_applies_to_the_whole_set_operation_and_keeps_offset() -> None:
    union = check_query("SELECT usi FROM uls_hd UNION ALL SELECT usi FROM uls_fr")
    assert union.sql.endswith("LIMIT 100")
    paged = check_query("SELECT * FROM uls_hd ORDER BY usi LIMIT 20 OFFSET 40")
    assert "OFFSET 40" in paged.sql and "LIMIT 20" in paged.sql


def test_limits_are_configurable() -> None:
    assert check_query("SELECT * FROM uls_hd", default_limit=7, max_limit=50).limit == 7
    assert check_query("SELECT * FROM uls_hd LIMIT 99", default_limit=7, max_limit=50).limit == 50


def test_the_sql_that_runs_is_regenerated_without_comments() -> None:
    checked = check_query("/* hello */ SELECT 1 -- ; DROP TABLE uls_hd\n")
    assert checked.sql == "SELECT 1 LIMIT 100"
    tricky = check_query("SELECT 1 FROM uls_hd -- */ ; DROP TABLE uls_hd\n")
    assert tricky.sql == "SELECT 1 FROM uls_hd LIMIT 100"  # no comment text survives at all


def test_fetch_sql_asks_for_one_extra_row_so_truncation_can_be_seen() -> None:
    added = check_query("SELECT * FROM uls_hd")
    assert added.sql.endswith("LIMIT 100") and added.fetch_sql.endswith("LIMIT 101")
    assert added.limit == 100
    explicit = check_query("SELECT * FROM uls_hd LIMIT 3")
    assert explicit.sql.endswith("LIMIT 3") and explicit.fetch_sql.endswith("LIMIT 4")
    lowered = check_query("SELECT * FROM uls_hd LIMIT ALL")
    assert lowered.sql.endswith("LIMIT 1000") and lowered.fetch_sql.endswith("LIMIT 1001")
    paged = check_query("SELECT * FROM uls_hd ORDER BY usi LIMIT 20 OFFSET 40")
    assert "LIMIT 21" in paged.fetch_sql and "OFFSET 40" in paged.fetch_sql
    assert "LIMIT 20" in paged.sql and "LIMIT 21" not in paged.sql
    union = check_query("SELECT usi FROM uls_hd UNION ALL SELECT usi FROM uls_fr")
    assert union.sql.endswith("LIMIT 100") and union.fetch_sql.endswith("LIMIT 101")
    assert check_query("SELECT 1", default_limit=7, max_limit=50).fetch_sql.endswith("LIMIT 8")
