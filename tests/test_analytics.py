"""analytics-mcp tests (no Docker needed): real Postgres, no mocked database.

- the menu: parameter validation, the features an extract needs, the analysis id
- the database roles (rflens_analytics_reader, rflens_analytics_logger): what each can and cannot do
- the extract that goes into the sandbox, read as the reader role
- the isolation flags of the `docker run` command line
- the server through an in-process MCP client with a fake sandbox: tools, validation and rejections,
  the audit log, fail-closed logging, artifacts, the HTTP layer

The container itself (analyses, real isolation) is covered by `make test-analytics`.
"""

# ruff: noqa: F811
# (the secom_rows fixture imported from analytics_support is used by name)

import json
import re
import uuid
from collections.abc import Iterator
from pathlib import Path

import anyio
import psycopg.errors as pg_errors
import pytest
from mcp import Client
from pydantic import ValidationError
from sqlalchemy import create_engine, text
from sqlalchemy.exc import DBAPIError, ProgrammingError
from starlette.testclient import TestClient

from analytics_support import (  # noqa: F401  (secom_rows is a fixture)
    FIRST_RUN,
    N_RUNS,
    FakeSandbox,
    ok_outcome,
    secom_rows,
)
from rflens.analytics import menu
from rflens.analytics.extract import ExtractError, build_extract
from rflens.analytics.sandbox import SandboxResult, docker_command
from rflens.config import AnalyticsSettings, get_analytics_settings
from rflens.db import get_engine
from rflens.mcp.analytics_server import create_analytics_server
from rflens.mcp.docs_server import http_options

PROTOCOL = "2026-07-28"


def settings(tmp_path: Path | None = None, **kw) -> AnalyticsSettings:
    return AnalyticsSettings(
        analytics_reader_password="unused",
        analytics_logger_password="unused",
        analytics_artifacts_dir=str(tmp_path) if tmp_path else "",
        **kw,
    )


# --- the menu -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("analysis", "params"),
    [
        ("describe", {"feature": 60}),
        ("describe", {"feature": 1, "filters": {"labels": ["fail"]}}),
        ("distribution", {"feature": 104, "bins": 20, "by_label": True}),
        ("group_compare", {"feature": 60, "test": "welch_t", "min_n": 30}),
        ("correlation", {"feature_a": 1, "feature_b": 2, "method": "pearson"}),
        ("correlation_top", {"feature": 5, "top_k": 50}),
        ("feature_ranking", {"top_k": 5}),
        ("feature_ranking", {}),
        ("missing_summary", {"top_k": 3}),
        ("outliers", {"feature": 3, "method": "zscore", "z": 2.5}),
        ("plot", {"kind": "histogram", "feature": 60}),
        ("plot", {"kind": "scatter", "feature_a": 1, "feature_b": 2}),
        ("plot", {"kind": "ranking_bar", "top_k": 30}),
        (
            "describe",
            {
                "feature": 1,
                "filters": {
                    "run_date_from": "2008-08-01",
                    "conditions": [{"feature": 2, "op": ">", "value": 1.5}],
                },
            },
        ),
    ],
)
def test_valid_parameters_are_accepted(analysis, params) -> None:
    parsed = menu.validate(analysis, params)
    assert isinstance(menu.normalised(parsed), dict)


@pytest.mark.parametrize(
    ("analysis", "params", "message"),
    [
        ("describe", {"feature": 0}, "greater than or equal to 1"),
        ("describe", {"feature": 591}, "less than or equal to 590"),
        ("describe", {}, "Field required"),
        ("describe", {"feature": 1, "code": "import os"}, "Extra inputs are not permitted"),
        ("describe", {"feature": 1, "sql": "SELECT 1"}, "Extra inputs are not permitted"),
        ("describe", {"feature": "60; DROP TABLE x"}, "valid integer"),
        ("distribution", {"feature": 1, "bins": 4}, "greater than or equal to 5"),
        ("distribution", {"feature": 1, "bins": 51}, "less than or equal to 50"),
        ("feature_ranking", {"top_k": 51}, "less than or equal to 50"),
        ("feature_ranking", {"top_k": 0}, "greater than or equal to 1"),
        ("feature_ranking", {"method": "random_forest"}, "mann_whitney"),
        ("feature_ranking", {"positive": "pass"}, "fail"),
        ("group_compare", {"feature": 1, "test": "ttest"}, "mann_whitney"),
        ("group_compare", {"feature": 1, "min_n": 3}, "greater than or equal to 10"),
        ("correlation", {"feature_a": 1, "feature_b": 1}, "must differ"),
        ("outliers", {"feature": 1, "k": 0.1}, "greater than or equal to 0.5"),
        ("plot", {"kind": "histogram"}, "needs exactly"),
        ("plot", {"kind": "scatter", "feature_a": 1}, "needs exactly"),
        ("plot", {"kind": "ranking_bar", "feature": 1}, "needs exactly"),
        ("plot", {"kind": "pie", "feature": 1}, "histogram"),
        ("plot", {"kind": "ranking_bar", "top_k": 31}, "less than or equal to 30"),
        ("describe", {"feature": 1, "filters": {"labels": ["maybe"]}}, "pass"),
        ("describe", {"feature": 1, "filters": {"run_date_from": "yesterday"}}, "date"),
        (
            "describe",
            {"feature": 1, "filters": {"conditions": [{"feature": 2, "op": ">"}]}},
            "needs a value",
        ),
        (
            "describe",
            {
                "feature": 1,
                "filters": {"conditions": [{"feature": 2, "op": "is_null", "value": 1}]},
            },
            "takes no value",
        ),
        (
            "describe",
            {"feature": 1, "filters": {"conditions": [{"feature": 2, "op": "~", "value": 1}]}},
            "is_null",
        ),
        (
            "describe",
            {"feature": 1, "filters": {"conditions": [{"feature": 2, "op": ">", "value": 1}] * 4}},
            "at most 3",
        ),
    ],
)
def test_invalid_parameters_are_rejected(analysis, params, message) -> None:
    with pytest.raises(ValidationError) as e:
        menu.validate(analysis, params)
    assert message in str(e.value)


def test_unknown_analysis_and_non_object_params_are_rejected() -> None:
    with pytest.raises(ValueError, match="unknown analysis 'exec'"):
        menu.validate("exec", {})
    with pytest.raises(ValueError, match="JSON object"):
        menu.validate("describe", "feature=1")


def test_the_menu_has_nine_analyses_and_no_parameter_can_hold_code() -> None:
    assert list(menu.ANALYSES) == [
        "describe",
        "missing_summary",
        "distribution",
        "group_compare",
        "correlation",
        "correlation_top",
        "feature_ranking",
        "outliers",
        "plot",
    ]
    for analysis in menu.ANALYSES.values():
        schema = analysis.model.model_json_schema()
        assert schema.get("additionalProperties") is False, analysis.name
        text_fields = [
            k
            for k, v in schema["properties"].items()
            if v.get("type") == "string" and "enum" not in v and "const" not in v
        ]
        # no top-level free string (a single-value Literal is a const); the only free strings
        # are the two ISO dates inside filters, validated as dates
        assert text_fields == [], (analysis.name, text_fields)


def test_defaults_are_filled_in_and_the_filters_normalised() -> None:
    normal = menu.normalised(menu.validate("distribution", {"feature": 7}))
    assert normal == {
        "feature": 7,
        "bins": 20,
        "by_label": False,
        "filters": {"labels": None, "run_date_from": None, "run_date_to": None, "conditions": []},
    }


def test_features_needed_is_the_named_features_plus_filter_features() -> None:
    def need(analysis, params):
        return menu.features_needed(analysis, menu.validate(analysis, params))

    assert need("describe", {"feature": 9}) == [9]
    cond = {"conditions": [{"feature": 3, "op": "not_null"}]}
    assert need("describe", {"feature": 9, "filters": cond}) == [3, 9]
    assert need("plot", {"kind": "scatter", "feature_a": 8, "feature_b": 2}) == [2, 8]
    for analysis, params in (
        ("feature_ranking", {}),
        ("missing_summary", {}),
        ("correlation_top", {"feature": 1}),
        ("plot", {"kind": "ranking_bar"}),
    ):
        assert need(analysis, params) is None


def test_analysis_id_is_deterministic_and_depends_on_everything() -> None:
    a = menu.analysis_id("describe", {"feature": 1}, "x" * 64)
    assert re.fullmatch(r"[0-9a-f]{64}", a)
    assert a == menu.analysis_id("describe", {"feature": 1}, "x" * 64)
    assert a != menu.analysis_id("describe", {"feature": 2}, "x" * 64)
    assert a != menu.analysis_id("distribution", {"feature": 1}, "x" * 64)
    assert a != menu.analysis_id("describe", {"feature": 1}, "y" * 64)


def test_list_analyses_document_describes_dataset_and_schemas() -> None:
    doc = menu.describe_menu()
    assert doc["dataset"]["name"] == "secom" and doc["dataset"]["n_features"] == 590
    assert "UNKNOWN" in doc["dataset"]["feature_unit"]
    assert [a["name"] for a in doc["analyses"]] == list(menu.ANALYSES)
    assert all(a["params_schema"]["type"] == "object" for a in doc["analyses"])
    json.dumps(doc)  # JSON-able


# --- the database roles ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def reader_engine():
    engine = create_engine(get_analytics_settings().reader_url())
    yield engine
    engine.dispose()


@pytest.fixture(scope="module")
def logger_engine():
    engine = create_engine(get_analytics_settings().logger_url())
    yield engine
    engine.dispose()


def test_reader_role_identity_defaults_and_reach(reader_engine) -> None:
    with reader_engine.connect() as c:
        assert c.execute(text("SELECT current_user")).scalar_one() == "rflens_analytics_reader"
        assert c.execute(text("SHOW default_transaction_read_only")).scalar_one() == "on"
        assert c.execute(text("SHOW statement_timeout")).scalar_one() == "15s"
        privileged = c.execute(
            text(
                "SELECT rolsuper OR rolcreatedb OR rolcreaterole FROM pg_roles "
                "WHERE rolname = current_user"
            )
        ).scalar_one()
        assert privileged is False
        for relation in ("secom_runs", "secom_measurements", "v_secom_values"):
            assert c.execute(text(f"SELECT count(*) FROM {relation}")).scalar_one() >= 0


@pytest.mark.parametrize(
    "relation",
    [
        "uls_hd",
        "uls_en",
        "uls_fr",
        "fcc_grantees",
        "spec_operating_bands",
        "chunks",
        "documents",
        "data_sources",
        "sql_query_log",
        "analytics_call_log",
        "llm_calls",
        "alembic_version",
    ],
)
def test_reader_role_reads_nothing_but_secom(reader_engine, relation) -> None:
    with reader_engine.connect() as c, pytest.raises(ProgrammingError) as err:
        c.execute(text(f"SELECT * FROM {relation}"))
    assert isinstance(err.value.orig, pg_errors.InsufficientPrivilege)


def test_reader_role_cannot_write(reader_engine) -> None:
    for statement in (
        "INSERT INTO secom_runs (run_id, label, run_ts) VALUES (1, 'pass', now())",
        "UPDATE secom_runs SET label = 'pass'",
        "DELETE FROM secom_measurements",
        "TRUNCATE secom_runs",
        "INSERT INTO analytics_call_log (tool, status) VALUES ('describe', 'ok')",
        "CREATE TABLE should_not_exist (a int)",
    ):
        with reader_engine.connect() as c, pytest.raises(DBAPIError) as err:
            c.execute(text(statement))
        assert isinstance(
            err.value.orig, pg_errors.InsufficientPrivilege | pg_errors.ReadOnlySqlTransaction
        ), statement
        with reader_engine.connect() as c, pytest.raises(ProgrammingError) as err:
            c.execute(text("SET TRANSACTION READ WRITE"))
            c.execute(text(statement))
        assert isinstance(err.value.orig, pg_errors.InsufficientPrivilege), statement


def test_logger_role_can_only_insert_into_the_call_log(logger_engine) -> None:
    with logger_engine.connect() as c:
        assert c.execute(text("SELECT current_user")).scalar_one() == "rflens_analytics_logger"
        c.execute(text("SELECT nextval('analytics_call_log_id_seq')"))
        c.rollback()
    for statement in (
        "SELECT * FROM analytics_call_log",
        "UPDATE analytics_call_log SET status = 'ok'",
        "DELETE FROM analytics_call_log",
        "SELECT * FROM secom_runs",
        "INSERT INTO sql_query_log (tool, sql_text, status) VALUES ('run_query', 'x', 'ok')",
        "INSERT INTO secom_runs (run_id, label, run_ts) VALUES (1, 'pass', now())",
    ):
        with logger_engine.connect() as c, pytest.raises(ProgrammingError) as err:
            c.execute(text(statement))
        assert isinstance(err.value.orig, pg_errors.InsufficientPrivilege), statement
    with logger_engine.connect() as c:
        c.execute(
            text(
                "INSERT INTO analytics_call_log (tool, status, question) "
                "VALUES ('describe', 'ok', 'pytest-logger-role-probe')"
            )
        )
        c.rollback()


def test_extract_is_a_wide_csv_with_empty_cells_for_missing(secom_rows, reader_engine) -> None:
    with reader_engine.connect() as c:
        ex = build_extract(c, [3, 1])
    lines = ex.csv_bytes.decode().splitlines()
    mine = [ln for ln in lines if ln.split(",")[0].isdigit() and int(ln.split(",")[0]) >= FIRST_RUN]
    assert lines[0] == "run_id,label,run_ts,f1,f3" and ex.features == [1, 3]
    assert len(mine) == N_RUNS == ex.n_runs
    assert mine[0] == f"{FIRST_RUN},pass,2008-08-01 10:30:00,1.5,"  # NULL value: empty cell
    assert mine[1] == f"{FIRST_RUN + 1},pass,2008-08-02 10:30:00,11.5,13.5"
    assert mine[-1].split(",")[1] == "fail"
    with reader_engine.connect() as c:
        again = build_extract(c, [1, 3, 3])
    assert again.sha256 == ex.sha256  # same columns, same bytes


def test_extract_of_all_features_has_590_columns_and_obeys_the_row_cap(
    secom_rows, reader_engine
) -> None:
    with reader_engine.connect() as c:
        ex = build_extract(c, None)
        assert len(ex.features) == 590 and ex.csv_bytes.decode().splitlines()[0].count(",") == 592
        with pytest.raises(ExtractError, match="the limit is 10"):
            build_extract(c, [1], max_runs=10)


# --- the docker run command line ------------------------------------------------------------------


def test_docker_command_carries_every_isolation_flag(tmp_path) -> None:
    cfg = settings(tmp_path)
    cmd = docker_command(cfg, "rflens-analytics-abc", tmp_path / "in", tmp_path / "out")
    joined = " ".join(cmd)
    assert cmd[:3] == ["docker", "run", "--rm"]
    assert cmd[cmd.index("--network") + 1] == "none"
    assert "--read-only" in cmd
    assert cmd[cmd.index("--cap-drop") + 1] == "ALL"
    assert cmd[cmd.index("--security-opt") + 1] == "no-new-privileges"
    assert cmd[cmd.index("--memory") + 1] == cmd[cmd.index("--memory-swap") + 1] == "512m"
    assert cmd[cmd.index("--cpus") + 1] == "1"
    assert cmd[cmd.index("--pids-limit") + 1] == "64"
    assert "noexec" in cmd[cmd.index("--tmpfs") + 1]
    mounts = [cmd[i + 1] for i, a in enumerate(cmd) if a == "--mount"]
    assert len(mounts) == 2
    assert mounts[0].endswith("target=/in,readonly") and "readonly" not in mounts[1]
    assert mounts[1].endswith("target=/out")
    assert "-v" not in cmd and "--privileged" not in cmd and "--network=host" not in joined
    assert cmd[-4:] == [cfg.analytics_image, "python", "-I", "/app/runner.py"]
    assert "postgres" not in joined.lower() and "password" not in joined.lower()  # no credentials


# --- the server with a fake sandbox ---------------------------------------------------------------


@pytest.fixture
def tag() -> Iterator[str]:
    value = f"pytest-analytics-{uuid.uuid4().hex}"
    yield value
    with get_engine().begin() as c:
        c.execute(text("DELETE FROM analytics_call_log WHERE question LIKE :q"), {"q": value + "%"})


def make_server(reader_engine, logger_engine, sandbox, tmp_path):
    return create_analytics_server(
        reader_engine.connect, logger_engine.connect, sandbox, settings(tmp_path)
    )


def call(server, tool: str, arguments: dict | None = None):
    async def run():
        async with Client(server) as client:
            return await client.call_tool(tool, arguments or {})

    return anyio.run(run)


def data(result) -> dict:
    assert not result.is_error, result.content
    return result.structured_content


def error_text(result) -> str:
    assert result.is_error
    return re.sub(r"^Error executing tool \w+: ", "", result.content[0].text)


def log_rows(tag: str) -> list:
    with get_engine().connect() as c:
        return list(
            c.execute(
                text(
                    "SELECT id, tool, question, params, analysis_id, n_runs, status, reason "
                    "FROM analytics_call_log WHERE question = :q ORDER BY id"
                ),
                {"q": tag},
            )
        )


@pytest.fixture
def fake() -> FakeSandbox:
    return FakeSandbox()


@pytest.fixture
def server(secom_rows, reader_engine, logger_engine, fake, tmp_path):
    return make_server(reader_engine, logger_engine, fake, tmp_path)


def test_lists_ten_tools_that_say_results_are_data_not_instructions(server) -> None:
    async def run():
        async with Client(server) as client:
            return client.protocol_version, (await client.list_tools()).tools

    version, tools = anyio.run(run)
    assert version == PROTOCOL
    assert {t.name for t in tools} == {"list_analyses", *menu.ANALYSES}
    for tool in tools:
        assert "NOT instructions" in tool.description
        assert (
            tool.annotations.read_only_hint is True and tool.annotations.destructive_hint is False
        )
    assert next(t for t in tools if t.name == "describe").input_schema["required"] == ["params"]


def test_list_analyses_returns_the_menu_and_is_logged(server, tag) -> None:
    out = data(call(server, "list_analyses", {"question": tag}))
    assert [a["name"] for a in out["analyses"]] == list(menu.ANALYSES)
    (row,) = log_rows(tag)
    assert (row.tool, row.status) == ("list_analyses", "ok")


def test_an_analysis_runs_in_the_sandbox_with_validated_params_and_the_extract(
    server, fake, tag
) -> None:
    out = data(call(server, "describe", {"params": {"feature": 3}, "question": tag}))
    (job,) = fake.jobs
    assert job["analysis"] == "describe"
    assert job["params"] == menu.normalised(menu.validate("describe", {"feature": 3}))
    assert fake.extracts[0].decode().splitlines()[0] == "run_id,label,run_ts,f3"
    (row,) = log_rows(tag)
    assert row.status == "ok" and row.tool == "describe" and row.n_runs == N_RUNS
    assert out["call_id"] == row.id and out["analysis_id"] == row.analysis_id
    assert re.fullmatch(r"[0-9a-f]{64}", out["analysis_id"])
    assert out["params"]["filters"]["conditions"] == [] and out["result"] == {
        "n": 42,
        "p_value": 0.01,
    }
    assert out["dataset"]["name"] == "secom" and out["warnings"] == []
    assert json.loads(row.params) == {"feature": 3}  # the call as received


def test_same_call_same_analysis_id_and_a_json_artifact(server, tag, tmp_path) -> None:
    a = data(call(server, "describe", {"params": {"feature": 1}, "question": tag}))
    b = data(call(server, "describe", {"params": {"feature": 1}, "question": tag}))
    c = data(call(server, "describe", {"params": {"feature": 2}, "question": tag}))
    assert a["analysis_id"] == b["analysis_id"] != c["analysis_id"]
    assert (tmp_path / f"{a['analysis_id']}.json").exists()


def test_a_plot_is_stored_under_its_analysis_id(
    secom_rows, reader_engine, logger_engine, tag, tmp_path
) -> None:
    png = b"\x89PNG\r\n\x1a\n" + b"fake"
    fake = FakeSandbox(ok_outcome("plot", png=png))
    server = make_server(reader_engine, logger_engine, fake, tmp_path)
    out = data(
        call(server, "plot", {"params": {"kind": "histogram", "feature": 1}, "question": tag})
    )
    (art,) = out["artifacts"]
    assert art == {"id": out["analysis_id"], "kind": "png", "bytes": len(png)}
    assert (tmp_path / f"{art['id']}.png").read_bytes() == png


def test_the_filters_warning_says_how_many_runs_were_kept(
    secom_rows, reader_engine, logger_engine, tag, tmp_path
) -> None:
    server = make_server(
        reader_engine, logger_engine, FakeSandbox(ok_outcome("describe", used=20)), tmp_path
    )
    params = {"feature": 1, "filters": {"labels": ["fail"]}}
    out = data(call(server, "describe", {"params": params, "question": tag}))
    assert out["warnings"] == [f"the filters kept 20 of {N_RUNS} runs"]


@pytest.mark.parametrize(
    ("params", "needle"),
    [
        ({"feature": 999}, "less than or equal to 590"),
        ({"feature": 1, "code": "__import__('os')"}, "Extra inputs are not permitted"),
        ({}, "Field required"),
    ],
)
def test_invalid_params_are_rejected_logged_and_never_reach_the_sandbox(
    server, fake, tag, params, needle
) -> None:
    message = error_text(call(server, "describe", {"params": params, "question": tag}))
    assert message.startswith("rejected (invalid_params)") and needle in message
    assert fake.jobs == []
    (row,) = log_rows(tag)
    assert row.status == "rejected" and row.reason.startswith("invalid_params:")
    assert row.analysis_id is None and row.n_runs is None


def test_overlong_params_are_rejected(server, fake, tag) -> None:
    huge = {"feature": 1, "filters": {"run_date_from": "x" * 25_000}}
    message = error_text(call(server, "describe", {"params": huge, "question": tag}))
    assert message.startswith("rejected (too_long)") and fake.jobs == []
    (row,) = log_rows(tag)
    assert len(row.params) <= 20_000


def test_an_analysis_the_data_cannot_answer_is_a_logged_rejection(
    secom_rows, reader_engine, logger_engine, tag, tmp_path
) -> None:
    refusal = SandboxResult(
        status="ok",
        payload={
            "ok": False,
            "error": {"code": "too_few_values", "message": "each group needs at least 20"},
        },
    )
    server = make_server(reader_engine, logger_engine, FakeSandbox(refusal), tmp_path)
    message = error_text(call(server, "group_compare", {"params": {"feature": 4}, "question": tag}))
    assert message == "rejected (too_few_values): each group needs at least 20"
    (row,) = log_rows(tag)
    assert row.status == "rejected" and row.reason.startswith("too_few_values:")
    assert row.analysis_id is not None and row.n_runs == N_RUNS


@pytest.mark.parametrize(
    ("outcome", "status", "needle"),
    [
        (SandboxResult(status="timeout", message="ran longer than 60 s"), "timeout", "60 s"),
        (SandboxResult(status="error", message="cannot run docker: boom"), "error", "boom"),
        (
            SandboxResult(
                status="ok",
                payload={"ok": False, "error": {"code": "internal", "message": "KeyError: 'x'"}},
            ),
            "error",
            "internal: KeyError",
        ),
    ],
)
def test_sandbox_failures_are_logged_and_reported(
    secom_rows, reader_engine, logger_engine, tag, tmp_path, outcome, status, needle
) -> None:
    server = make_server(reader_engine, logger_engine, FakeSandbox(outcome), tmp_path)
    message = error_text(call(server, "describe", {"params": {"feature": 1}, "question": tag}))
    assert message.startswith(f"analysis failed ({status})") and needle in message
    (row,) = log_rows(tag)
    assert row.status == status and needle in row.reason


def test_if_the_log_cannot_be_written_the_result_is_withheld(
    secom_rows, reader_engine, tag, tmp_path
) -> None:
    broken = create_engine(get_analytics_settings().reader_url())  # the reader cannot INSERT
    try:
        server = make_server(reader_engine, broken, FakeSandbox(), tmp_path)
        for tool, args in (
            ("describe", {"params": {"feature": 1}, "question": tag}),
            ("list_analyses", {"question": tag}),
            ("describe", {"params": {"feature": 0}, "question": tag}),
        ):
            message = error_text(call(server, tool, args))
            assert "audit log write failed" in message and "withheld" in message
    finally:
        broken.dispose()
    assert log_rows(tag) == []
    assert list(tmp_path.glob("*.json")) == []  # nothing was published either


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
    with TestClient(app, base_url="http://127.0.0.1:8103") as client:
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
        reply = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "describe",
                    "_meta": meta,
                    "arguments": {"params": {"feature": 2}, "question": tag},
                },
            },
            headers=headers | {"Mcp-Method": "tools/call", "Mcp-Name": "describe"},
        )
        assert reply.status_code == 200
        assert reply.json()["result"]["structuredContent"]["analysis"] == "describe"
        evil = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 3,
                "method": "server/discover",
                "params": {"_meta": meta},
            },
            headers=headers | {"Mcp-Method": "server/discover", "Host": "evil.example"},
        )
        assert evil.status_code in (400, 403, 421)
