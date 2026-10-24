"""sql-mcp tests: real Postgres, no mocks.

- the two database roles (rflens_sql_reader, rflens_sql_logger) and what each can and cannot do
- the three views on synthetic rows (inserted in a rolled-back transaction)
- the server through an in-process MCP client, connected as the real roles: tools, safe queries,
  rejections, cost gate, timeout, row caps, database errors, the audit log, fail-closed logging

The static guardrails are covered in test_sql_guard.py.
"""

import re
import time
import uuid
from collections.abc import Iterator

import anyio
import psycopg.errors as pg_errors
import pytest
from mcp import Client
from sqlalchemy import Connection, create_engine, text
from sqlalchemy.exc import DBAPIError, ProgrammingError
from starlette.testclient import TestClient

from rflens.config import SqlMcpSettings, get_settings, get_sql_mcp_settings
from rflens.db import analyze, get_engine
from rflens.mcp.docs_server import http_options
from rflens.mcp.sql_guard import ALLOWED_RELATIONS, ALLOWED_VIEWS
from rflens.mcp.sql_server import create_sql_server

PROTOCOL = "2026-07-28"
READABLE = [*ALLOWED_RELATIONS]
UNREADABLE = ["uls_en", "chunks", "documents", "alembic_version", "sql_query_log"]


@pytest.fixture
def conn() -> Iterator[Connection]:
    """Admin connection, rolled back so real data is never touched."""
    with get_engine().connect() as connection:
        transaction = connection.begin()
        try:
            yield connection
        finally:
            transaction.rollback()


@pytest.fixture(scope="module")
def reader_engine():
    engine = create_engine(get_sql_mcp_settings().reader_url())
    yield engine
    engine.dispose()


@pytest.fixture(scope="module")
def logger_engine():
    engine = create_engine(get_sql_mcp_settings().logger_url())
    yield engine
    engine.dispose()


# --- the database roles ---------------------------------------------------------------------------


def assert_denied(engine, statement: str) -> None:
    """The statement fails in both layers: the read-only default, and the privileges alone."""
    with engine.connect() as c, pytest.raises(DBAPIError) as err:
        c.execute(text(statement))
    assert isinstance(
        err.value.orig, pg_errors.InsufficientPrivilege | pg_errors.ReadOnlySqlTransaction
    ), statement
    with engine.connect() as c, pytest.raises(ProgrammingError) as err:
        c.execute(text("SET TRANSACTION READ WRITE"))
        c.execute(text(statement))
    assert isinstance(err.value.orig, pg_errors.InsufficientPrivilege), statement


def test_reader_role_identity_and_defaults(reader_engine) -> None:
    with reader_engine.connect() as c:
        assert c.execute(text("SELECT current_user")).scalar_one() == "rflens_sql_reader"
        assert c.execute(text("SHOW default_transaction_read_only")).scalar_one() == "on"
        assert c.execute(text("SHOW statement_timeout")).scalar_one() == "15s"
        privileged = c.execute(
            text(
                "SELECT rolsuper OR rolcreatedb OR rolcreaterole FROM pg_roles "
                "WHERE rolname = current_user"
            )
        ).scalar_one()
        assert privileged is False


@pytest.mark.parametrize("relation", READABLE)
def test_reader_role_can_select_every_allow_listed_relation(reader_engine, relation: str) -> None:
    with reader_engine.connect() as c:
        assert c.execute(text(f"SELECT count(*) FROM {relation}")).scalar_one() >= 0


@pytest.mark.parametrize("relation", UNREADABLE)
def test_reader_role_cannot_read_the_rest(reader_engine, relation: str) -> None:
    with reader_engine.connect() as c, pytest.raises(ProgrammingError) as err:
        c.execute(text(f"SELECT * FROM {relation}"))
    assert isinstance(err.value.orig, pg_errors.InsufficientPrivilege)


def test_reader_role_cannot_write_anywhere(reader_engine) -> None:
    statements = [
        "INSERT INTO uls_hd (service, usi, line_no) VALUES ('x', 1, 1)",
        "INSERT INTO sql_query_log (tool, sql_text, status) VALUES ('run_query', 'x', 'ok')",
        "INSERT INTO data_sources (name) VALUES ('x')",
        "UPDATE uls_hd SET call_sign = 'x'",
        "UPDATE sql_query_log SET status = 'ok'",
        "UPDATE uls_en SET city = 'x'",
        "DELETE FROM uls_hd",
        "DELETE FROM sql_query_log",
        "DELETE FROM secom_runs",
        "TRUNCATE uls_fr",
        "TRUNCATE sql_query_log",
        "CREATE TABLE rflens_should_not_exist (a int)",
        "DROP TABLE uls_hd",
        "DROP VIEW v_uls_licences",
        "ALTER TABLE uls_hd ADD COLUMN x int",
        "GRANT SELECT ON uls_en TO PUBLIC",
    ]
    for statement in statements:
        assert_denied(reader_engine, statement)


def test_logger_role_can_only_insert_into_the_log(logger_engine) -> None:
    with logger_engine.connect() as c:
        assert c.execute(text("SELECT current_user")).scalar_one() == "rflens_sql_logger"
        c.execute(
            text(
                "INSERT INTO sql_query_log (tool, sql_text, status, question) "
                "VALUES ('run_query', 'SELECT 1', 'ok', 'pytest-logger-role-rolled-back')"
            )
        )
        c.rollback()  # nothing is kept
    for statement in (
        "SELECT * FROM sql_query_log",
        "UPDATE sql_query_log SET status = 'ok'",
        "DELETE FROM sql_query_log",
        "TRUNCATE sql_query_log",
        "SELECT * FROM uls_hd",
        "SELECT * FROM chunks",
        "INSERT INTO uls_hd (service, usi, line_no) VALUES ('x', 1, 1)",
        "INSERT INTO data_sources (name) VALUES ('x')",
        "CREATE TABLE rflens_should_not_exist (a int)",
    ):
        with logger_engine.connect() as c, pytest.raises(ProgrammingError) as err:
            c.execute(text(statement))
        assert isinstance(err.value.orig, pg_errors.InsufficientPrivilege), statement


def test_log_table_rejects_unknown_status_and_tool(conn: Connection) -> None:
    for tool, status in (("run_query", "maybe"), ("drop_all", "ok")):
        with pytest.raises(DBAPIError) as err:
            with conn.begin_nested():
                conn.execute(
                    text("INSERT INTO sql_query_log (tool, sql_text, status) VALUES (:t, 'x', :s)"),
                    {"t": tool, "s": status},
                )
        assert isinstance(err.value.orig, pg_errors.CheckViolation)


def test_role_settings_swap_only_the_credentials() -> None:
    from sqlalchemy.engine import make_url

    s = get_sql_mcp_settings()
    admin = make_url(get_settings().database_url)
    for url, user in ((s.reader_url(), "rflens_sql_reader"), (s.logger_url(), "rflens_sql_logger")):
        assert url.username == user and url.password != admin.password
        assert (url.host, url.port, url.database) == (admin.host, admin.port, admin.database)
    assert s.reader_url().password != s.logger_url().password


# --- settings -------------------------------------------------------------------------------------


def test_limit_settings_default_and_are_env_overridable(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "SQL_MAX_COST", "SQL_STATEMENT_TIMEOUT_MS", "SQL_DEFAULT_LIMIT", "SQL_MAX_LIMIT",
        "SQL_MCP_PORT",
    ):  # fmt: skip
        monkeypatch.delenv(name, raising=False)
    s = SqlMcpSettings(_env_file=None, sql_reader_password="a", sql_logger_password="b")
    assert (s.sql_max_cost, s.sql_statement_timeout_ms) == (500_000, 15_000)
    assert (s.sql_default_limit, s.sql_max_limit, s.sql_mcp_port) == (100, 1000, 8102)
    monkeypatch.setenv("SQL_MAX_COST", "123")
    monkeypatch.setenv("SQL_STATEMENT_TIMEOUT_MS", "2000")
    monkeypatch.setenv("SQL_DEFAULT_LIMIT", "5")
    monkeypatch.setenv("SQL_MAX_LIMIT", "50")
    s = SqlMcpSettings(_env_file=None, sql_reader_password="a", sql_logger_password="b")
    assert (s.sql_max_cost, s.sql_statement_timeout_ms) == (123, 2000)
    assert (s.sql_default_limit, s.sql_max_limit) == (5, 50)


def test_inconsistent_limits_are_refused() -> None:
    kw = {"_env_file": None, "sql_reader_password": "a", "sql_logger_password": "b"}
    with pytest.raises(ValueError, match="SQL_DEFAULT_LIMIT"):
        SqlMcpSettings(**kw, sql_default_limit=2000)
    with pytest.raises(ValueError, match="SQL_MAX_COST"):
        SqlMcpSettings(**kw, sql_max_cost=0)


# --- the views, on synthetic rows -----------------------------------------------------------------


def seed_uls(conn: Connection) -> None:
    conn.execute(
        text(
            """
            INSERT INTO uls_hd (service, usi, line_no, call_sign, license_status) VALUES
              ('zz', 900001, 1, 'ZZTEST1', 'A'), ('zz', 900002, 2, 'ZZTEST2', 'C');
            INSERT INTO uls_en (service, usi, line_no, entity_type, entity_name, first_name,
                                last_name, city, state, frn) VALUES
              ('zz', 900001, 1, 'L', 'Acme Wireless', NULL, NULL, 'Austin', 'TX', '0001'),
              ('zz', 900001, 2, 'CL', NULL, 'Pat', 'Contact', 'Dallas', 'TX', '0002'),
              ('zz', 900001, 3, 'O', 'Owner Inc', NULL, NULL, 'Reno', 'NV', '0003');
            INSERT INTO uls_fr (service, line_no, usi, location_number, antenna_number,
                                freq_seq_id, frequency_assigned, frequency_upper_band, power_erp)
            VALUES ('zz', 1, 900001, 1, 1, 1, 870.0, 871.0, 100),
                   ('zz', 2, 900001, 1, 1, 2, 880.0, 881.0, 200),
                   ('zz', 3, 900002, 1, 1, 1, 824.0, 825.0, 50);
            """
        )
    )


def test_v_uls_licences_is_one_row_per_licence_with_the_licensee_only(conn: Connection) -> None:
    seed_uls(conn)
    rows = conn.execute(
        text(
            "SELECT usi, call_sign, license_status, licensee_name, licensee_city, licensee_frn "
            "FROM v_uls_licences WHERE service = 'zz' ORDER BY usi"
        )
    ).all()
    assert [tuple(r) for r in rows] == [
        (900001, "ZZTEST1", "A", "Acme Wireless", "Austin", "0001"),  # 'L' only: no CL, no O
        (900002, "ZZTEST2", "C", None, None, None),  # licence without an 'L' row is kept
    ]
    columns = conn.execute(text("SELECT * FROM v_uls_licences LIMIT 0")).keys()
    assert not {"first_name", "last_name", "entity_type", "zip_code"} & set(columns)


def test_v_uls_frequencies_joins_every_frequency_row_to_its_licence(conn: Connection) -> None:
    seed_uls(conn)
    rows = conn.execute(
        text(
            "SELECT usi, call_sign, license_status, licensee_name, frequency_assigned "
            "FROM v_uls_frequencies WHERE service = 'zz' ORDER BY usi, freq_seq_id"
        )
    ).all()
    assert [(r.usi, r.call_sign, r.licensee_name, float(r.frequency_assigned)) for r in rows] == [
        (900001, "ZZTEST1", "Acme Wireless", 870.0),
        (900001, "ZZTEST1", "Acme Wireless", 880.0),
        (900002, "ZZTEST2", None, 824.0),
    ]


def test_v_secom_values_carries_label_and_timestamp(conn: Connection) -> None:
    conn.execute(
        text(
            """
            INSERT INTO secom_runs (run_id, label, run_ts) VALUES
              (990001, 'fail', '2008-07-19 10:00:00'), (990002, 'pass', '2008-07-20 11:00:00');
            INSERT INTO secom_measurements (run_id, feature_idx, value) VALUES
              (990001, 7, 1.5), (990001, 8, NULL), (990002, 7, 2.5);
            """
        )
    )
    rows = conn.execute(
        text(
            "SELECT run_id, label, feature_idx, value, run_ts::text AS ts "
            "FROM v_secom_values WHERE run_id >= 990001 ORDER BY run_id, feature_idx"
        )
    ).all()
    assert [tuple(r) for r in rows] == [
        (990001, "fail", 7, 1.5, "2008-07-19 10:00:00"),
        (990001, "fail", 8, None, "2008-07-19 10:00:00"),
        (990002, "pass", 7, 2.5, "2008-07-20 11:00:00"),
    ]


def test_every_allow_listed_relation_and_column_has_a_comment(conn: Connection) -> None:
    missing = conn.execute(
        text(
            """
            SELECT c.relname, a.attname
            FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
            JOIN pg_attribute a ON a.attrelid = c.oid AND a.attnum > 0 AND NOT a.attisdropped
            WHERE n.nspname = 'public' AND c.relname = ANY(CAST(:names AS text[]))
              AND col_description(c.oid, a.attnum) IS NULL
            """
        ),
        {"names": [*ALLOWED_RELATIONS, "uls_en"]},
    ).all()
    assert missing == []
    no_table_comment = conn.execute(
        text(
            "SELECT relname FROM pg_class WHERE relname = ANY(CAST(:n AS text[])) "
            "AND obj_description(oid, 'pg_class') IS NULL"
        ),
        {"n": [*ALLOWED_RELATIONS, "uls_en"]},
    ).all()
    assert no_table_comment == []


def test_analyze_updates_planner_statistics() -> None:
    analyze(["data_sources"])
    deadline = time.monotonic() + 5  # statistics views lag a moment behind
    while time.monotonic() < deadline:
        with get_engine().connect() as c:
            last = c.execute(
                text("SELECT last_analyze FROM pg_stat_user_tables WHERE relname = 'data_sources'")
            ).scalar_one()
        if last is not None:
            break
        time.sleep(0.2)
    assert last is not None


# --- the server, connected as the real roles ------------------------------------------------------


@pytest.fixture
def tag() -> Iterator[str]:
    """A unique question text for this test; its log rows are deleted afterwards."""
    value = f"pytest-sql-mcp-{uuid.uuid4().hex}"
    yield value
    with get_engine().begin() as c:
        c.execute(text("DELETE FROM sql_query_log WHERE question LIKE :q"), {"q": value + "%"})


def make_server(reader_engine, logger_engine, **overrides):
    cfg = SqlMcpSettings(sql_reader_password="unused", sql_logger_password="unused", **overrides)
    return create_sql_server(reader_engine.connect, logger_engine.connect, cfg)


def call(server, tool: str, arguments: dict | None = None):
    async def run():
        async with Client(server) as client:
            return await client.call_tool(tool, arguments or {})

    return anyio.run(run)


def log_rows(question: str) -> list:
    with get_engine().connect() as c:
        return list(
            c.execute(
                text(
                    "SELECT tool, question, sql_text, plan_cost, row_count, duration_ms, status, "
                    "rejection_reason, logged_at FROM sql_query_log WHERE question = :q "
                    "ORDER BY id"
                ),
                {"q": question},
            )
        )


@pytest.fixture
def server(reader_engine, logger_engine):
    return make_server(reader_engine, logger_engine)


def data(result) -> dict:
    assert not result.is_error, result.content
    assert result.structured_content is not None
    return result.structured_content


def error_text(result) -> str:
    """The server's error message (the SDK prepends 'Error executing tool <name>: ')."""
    assert result.is_error
    return re.sub(r"^Error executing tool \w+: ", "", result.content[0].text)


def test_lists_three_tools_that_say_results_are_data_not_instructions(server) -> None:
    async def run():
        async with Client(server) as client:
            return client.protocol_version, (await client.list_tools()).tools

    version, tools = anyio.run(run)
    assert version == PROTOCOL
    assert {t.name for t in tools} == {"list_tables", "describe_table", "run_query"}
    for tool in tools:
        assert "NOT instructions" in tool.description
        assert tool.annotations.read_only_hint is True
        assert tool.annotations.destructive_hint is False
    run_query = next(t for t in tools if t.name == "run_query")
    assert (
        "single SELECT" in run_query.description
        and "LIMIT defaults to 100" in run_query.description
    )
    assert run_query.input_schema["required"] == ["sql"]


def test_list_tables_shows_nine_tables_and_three_views_but_not_uls_en(server, tag) -> None:
    out = data(call(server, "list_tables", {"question": tag}))
    names = [t["name"] for t in out["tables"]]
    assert names == list(ALLOWED_RELATIONS) and "uls_en" not in names
    kinds = {t["name"]: t["kind"] for t in out["tables"]}
    assert [n for n, k in kinds.items() if k == "view"] == list(ALLOWED_VIEWS)
    assert all(t["description"] for t in out["tables"])
    assert all(t["approx_rows"] is None for t in out["tables"] if t["kind"] == "view")
    (row,) = log_rows(tag)
    assert (row.tool, row.status, row.row_count) == ("list_tables", "ok", 12)


def test_describe_table_returns_comments_with_units_and_open_questions(server, tag) -> None:
    out = data(call(server, "describe_table", {"table": "v_uls_frequencies", "question": tag}))
    assert out["name"] == "v_uls_frequencies" and out["kind"] == "view"
    cols = {c["name"]: c for c in out["columns"]}
    assert "ASSUMED MHz" in cols["frequency_assigned"]["description"]
    assert "UNKNOWN" in cols["power_erp"]["description"]
    assert (
        "UNKNOWN" in cols["eirp"]["description"]
        and "UNKNOWN" in cols["power_output"]["description"]
    )
    assert "licensee" in cols["licensee_name"]["description"].lower()
    assert all(c["description"] for c in out["columns"]) and cols["usi"]["type"] == "bigint"
    assert "ASSUMED" in out["description"] and "UNKNOWN" in out["description"]
    (row,) = log_rows(tag)
    assert (row.tool, row.status) == ("describe_table", "ok")
    assert row.sql_text == "describe_table('v_uls_frequencies')"


def test_describe_table_is_case_insensitive_and_refuses_what_is_not_listed(server, tag) -> None:
    ok = data(call(server, "describe_table", {"table": "ULS_HD", "question": tag}))
    assert ok["name"] == "uls_hd"
    for bad in ("uls_en", "chunks", "sql_query_log", "pg_user", "uls_hd; DROP TABLE uls_hd", ""):
        message = error_text(call(server, "describe_table", {"table": bad, "question": tag}))
        assert "unknown table" in message and "v_uls_licences" in message
    rows = log_rows(tag)
    assert [r.status for r in rows] == ["ok"] + ["rejected"] * 6


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT count(*) AS n FROM v_uls_licences",
        "SELECT call_sign, licensee_name, count(*) FROM v_uls_frequencies GROUP BY 1, 2",
        "SELECT label, count(*) FROM v_secom_values GROUP BY label",
        "SELECT count(*) FROM uls_hd h JOIN uls_fr f USING (service, usi)",
        "SELECT * FROM fcc_grantees WHERE grantee_code = 'ZZZZZ'",
    ],
)
def test_safe_queries_run_through_each_view_and_table(server, tag, sql: str) -> None:
    out = data(call(server, "run_query", {"sql": sql, "question": tag}))
    assert out["columns"] and isinstance(out["rows"], list)
    assert out["plan_cost"] >= 0 and out["executed_sql"].endswith("LIMIT 100")
    (row,) = log_rows(tag)
    assert (row.tool, row.status, row.sql_text) == ("run_query", "ok", sql)
    assert row.plan_cost is not None and row.row_count == out["row_count"]


def test_run_query_result_log_and_json_values(server, tag) -> None:
    out = data(
        call(
            server,
            "run_query",
            {
                "sql": "SELECT DATE '2026-01-02' AS d, 1.5::numeric AS n, NULL AS z, "
                "'NaN'::float8 AS nan, 'a:b %' AS s",
                "question": tag,
            },
        )
    )
    assert out["columns"] == ["d", "n", "z", "nan", "s"]
    assert out["rows"] == [["2026-01-02", 1.5, None, None, "a:b %"]]
    assert out["row_count"] == 1 and out["truncated"] is False and out["limit"] == 100
    assert "LIMIT 100 added" in out["notice"]
    (row,) = log_rows(tag)
    assert row.question == tag and row.duration_ms is not None and row.rejection_reason is None


def test_unsafe_queries_are_rejected_and_logged_with_the_reason(server, tag) -> None:
    cases = [
        ("DELETE FROM uls_hd", "not_select"),
        ("SELECT * FROM uls_en", "table_not_allowed"),
        ('SELECT * FROM "uls_en"', "table_not_allowed"),
        ("WITH e AS (SELECT * FROM uls_en) SELECT * FROM e", "table_not_allowed"),
        ("SELECT 1; SELECT 2", "multiple_statements"),
        ("SELECT pg_sleep(5)", "function_denied"),
    ]
    for sql, code in cases:
        message = error_text(call(server, "run_query", {"sql": sql, "question": tag}))
        assert message.startswith(f"rejected ({code})"), (sql, message)
    rows = log_rows(tag)
    assert [r.status for r in rows] == ["rejected"] * len(cases)
    assert [r.sql_text for r in rows] == [s for s, _ in cases]
    assert [r.rejection_reason.split(":")[0] for r in rows] == [c for _, c in cases]
    assert all(r.row_count is None for r in rows)


def test_the_uls_data_is_untouched_by_a_rejected_delete(server, tag) -> None:
    with get_engine().connect() as c:
        before = c.execute(text("SELECT count(*) FROM uls_hd")).scalar_one()
    error_text(call(server, "run_query", {"sql": "DELETE FROM uls_hd", "question": tag}))
    with get_engine().connect() as c:
        assert c.execute(text("SELECT count(*) FROM uls_hd")).scalar_one() == before


def test_cost_gate_rejects_with_the_cost_shown_and_logs_it(
    reader_engine, logger_engine, tag
) -> None:
    sql = "SELECT * FROM uls_fr a, uls_lo b"
    strict = make_server(reader_engine, logger_engine, sql_max_cost=0.001)
    message = error_text(call(strict, "run_query", {"sql": sql, "question": tag}))
    assert message.startswith("rejected (cost)") and "estimated planner cost" in message
    assert "above the limit of 0" in message
    (row,) = log_rows(tag)
    assert row.status == "rejected" and row.plan_cost is not None and row.plan_cost > 0.001
    assert row.rejection_reason.startswith("cost:") and row.row_count is None
    # the same query under the normal threshold is not a cost problem for a trivial query
    normal = make_server(reader_engine, logger_engine)
    assert not call(normal, "run_query", {"sql": "SELECT 1", "question": tag}).is_error


def test_timeout_cancels_a_runaway_query_and_logs_it(reader_engine, logger_engine, tag) -> None:
    forever = (
        "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r) SELECT count(*) FROM r"
    )
    quick = make_server(reader_engine, logger_engine, sql_statement_timeout_ms=300)
    started = time.monotonic()
    message = error_text(call(quick, "run_query", {"sql": forever, "question": tag}))
    assert time.monotonic() - started < 10
    assert message.startswith("query failed (timeout)") and "0.3 s" in message
    (row,) = log_rows(tag)
    assert row.status == "error" and row.rejection_reason.startswith("timeout:")
    assert row.plan_cost is not None  # it passed the cost gate: the timeout was the backstop


def test_rows_are_capped_by_the_default_and_the_maximum_limit(server, tag) -> None:
    series = "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r WHERE n < 5000) "
    default = data(call(server, "run_query", {"sql": series + "SELECT n FROM r", "question": tag}))
    assert default["row_count"] == 100 and default["truncated"] is True
    assert default["limit"] == 100 and "added" in default["notice"]
    over = data(
        call(server, "run_query", {"sql": series + "SELECT n FROM r LIMIT 5000", "question": tag})
    )
    assert over["row_count"] == 1000 and over["limit"] == 1000 and "lowered" in over["notice"]
    small = data(
        call(server, "run_query", {"sql": series + "SELECT n FROM r LIMIT 7", "question": tag})
    )
    assert small["row_count"] == 7 and small["notice"] is None and small["truncated"] is True
    exact = data(call(server, "run_query", {"sql": "SELECT 1 AS one LIMIT 5", "question": tag}))
    assert exact["row_count"] == 1 and exact["truncated"] is False


def test_limits_come_from_settings(reader_engine, logger_engine, tag) -> None:
    tight = make_server(reader_engine, logger_engine, sql_default_limit=3, sql_max_limit=10)
    sql = "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r WHERE n < 50) "
    default = data(call(tight, "run_query", {"sql": sql + "SELECT n FROM r", "question": tag}))
    assert default["row_count"] == 3
    capped = data(
        call(tight, "run_query", {"sql": sql + "SELECT n FROM r LIMIT 99", "question": tag})
    )
    assert capped["row_count"] == 10


def test_database_errors_are_clear_logged_and_do_not_echo_the_sql(server, tag) -> None:
    message = error_text(
        call(server, "run_query", {"sql": "SELECT no_such_column FROM uls_hd", "question": tag})
    )
    assert message.startswith("query failed (database_error)")
    assert "UndefinedColumn" in message and "no_such_column" in message
    assert "FROM uls_hd" not in message
    (row,) = log_rows(tag)
    assert row.status == "error" and row.rejection_reason.startswith("database_error:")


def test_overlong_sql_and_question_are_bounded(server, tag) -> None:
    huge = {"sql": "SELECT 1 -- " + "x" * 25_000, "question": tag}
    message = error_text(call(server, "run_query", huge))
    assert message.startswith("rejected (too_long)")
    long_question = tag + "q" * 2000
    data(call(server, "run_query", {"sql": "SELECT 1", "question": long_question}))
    with get_engine().connect() as c:
        lengths = c.execute(
            text("SELECT length(question) FROM sql_query_log WHERE question LIKE :p ORDER BY id"),
            {"p": tag + "%"},
        ).scalars()
        assert list(lengths) == [len(tag), 1000]  # the rejected huge query, then the cut question


def test_if_the_log_cannot_be_written_the_result_is_withheld(reader_engine, tag) -> None:
    broken = create_engine(
        get_sql_mcp_settings().reader_url()  # the reader role has no INSERT on the log
    )
    try:
        cfg = SqlMcpSettings(sql_reader_password="unused", sql_logger_password="unused")
        server = create_sql_server(reader_engine.connect, broken.connect, cfg)
        for tool, args in (
            ("run_query", {"sql": "SELECT 1", "question": tag}),
            ("list_tables", {"question": tag}),
            ("run_query", {"sql": "DELETE FROM uls_hd", "question": tag}),
        ):
            message = error_text(call(server, tool, args))
            assert "audit log write failed" in message and "withheld" in message
    finally:
        broken.dispose()
    assert log_rows(tag) == []


def test_a_query_cannot_change_data_even_if_the_guard_were_bypassed(reader_engine) -> None:
    """The role is the boundary: these statements never get past the database."""
    for statement in ("DELETE FROM uls_hd", "UPDATE secom_runs SET label = 'pass'"):
        assert_denied(reader_engine, statement)


SERIES = (
    "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r WHERE n < {top}) "
    "SELECT n FROM r"
)


@pytest.mark.parametrize(
    ("top", "tail", "returned", "truncated"),
    [
        (3, " LIMIT 3", 3, False),  # explicit LIMIT equal to the row count: nothing was cut
        (2, " LIMIT 3", 2, False),  # fewer rows than the LIMIT
        (4, " LIMIT 3", 3, True),  # one more row than the LIMIT existed
        (100, "", 100, False),  # guard-applied default limit, exactly 100 rows exist
        (101, "", 100, True),  # guard-applied limit cuts a larger result
        (5000, "", 100, True),
        (1000, " LIMIT 5000", 1000, False),  # lowered to the maximum; exactly 1000 exist
        (1001, " LIMIT 5000", 1000, True),
    ],
)
def test_truncated_means_more_rows_existed_than_were_returned(
    server, tag, top: int, tail: str, returned: int, truncated: bool
) -> None:
    out = data(call(server, "run_query", {"sql": SERIES.format(top=top) + tail, "question": tag}))
    assert out["row_count"] == len(out["rows"]) == returned
    assert out["truncated"] is truncated
    assert out["rows"][0] == [1] and out["rows"][-1] == [returned]  # the first rows, in order
    assert f"LIMIT {out['limit'] + 1}" not in out["executed_sql"]  # the probe row is not shown
    (row,) = log_rows(tag)
    assert row.row_count == returned  # the log counts rows returned, not the probe row


# --- the HTTP layer (same options as `make sql-mcp`), without opening a port ----------------------


def test_http_stateless_discover_and_tool_call(server, tag) -> None:
    meta = {
        "io.modelcontextprotocol/protocolVersion": PROTOCOL,
        "io.modelcontextprotocol/clientCapabilities": {},
    }
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "MCP-Protocol-Version": PROTOCOL,
    }
    app = server.streamable_http_app(host="127.0.0.1", **http_options("127.0.0.1"))
    with TestClient(app, base_url="http://127.0.0.1:8102") as client:
        discover = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "server/discover",
                "params": {"_meta": meta},
            },
            headers=headers | {"Mcp-Method": "server/discover"},
        )
        assert discover.status_code == 200 and "mcp-session-id" not in discover.headers
        call_ = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "run_query",
                    "_meta": meta,
                    "arguments": {"sql": "SELECT 41 + 1 AS answer", "question": tag},
                },
            },
            headers=headers | {"Mcp-Method": "tools/call", "Mcp-Name": "run_query"},
        )
        assert call_.status_code == 200
        result = call_.json()["result"]
        assert not result.get("isError") and result["structuredContent"]["rows"] == [[42]]
        evil = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 3,
                "method": "server/discover",
                "params": {"_meta": meta},
            },
            headers=headers | {"Mcp-Method": "server/discover", "Origin": "http://evil.example"},
        )
        assert evil.status_code == 403
    assert [r.status for r in log_rows(tag)] == ["ok"]
