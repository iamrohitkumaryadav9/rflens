"""Migration 0012: the two roles' privileges, the query_id of run_query, and ledger seeding."""

import uuid
from collections.abc import Iterator

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import DBAPIError

from rflens.agents.sql_tools import SqlMcpTools
from rflens.config import (
    LlmSettings,
    get_checkpointer_settings,
    get_llm_settings,
    get_sql_mcp_settings,
)
from rflens.db import get_engine
from rflens.llm.base import Pricing, Rate
from rflens.llm.fake import FakeProvider
from rflens.llm.gateway import Gateway
from rflens.llm.guards import RateLimiter
from rflens.llm.ledger import Ledger
from rflens.mcp.sql_server import create_sql_server

PRICING = Pricing({"fake": {"fake-1": {"standard": Rate(1.0, 1.0)}}}, usd_to_inr=85.7)


@pytest.fixture(scope="module")
def checkpointer_engine():
    engine = create_engine(get_checkpointer_settings().sa_url())
    yield engine
    engine.dispose()


@pytest.fixture(scope="module")
def ledger_engine():
    engine = create_engine(get_llm_settings().ledger_url())
    yield engine
    engine.dispose()


def denied(engine, sql: str) -> bool:
    try:
        with engine.begin() as c:
            c.execute(text(sql))
    except DBAPIError as e:
        return "permission denied" in str(e.orig) or "must be owner" in str(e.orig)
    return False


def test_checkpointer_role_has_dml_on_the_four_tables_and_nothing_else(checkpointer_engine) -> None:
    thread = f"role-test-{uuid.uuid4().hex[:8]}"
    with checkpointer_engine.begin() as c:
        assert c.execute(text("SELECT current_schemas(false)")).scalar_one() == ["checkpoints"]
        for table in (
            "checkpoints",
            "checkpoint_blobs",
            "checkpoint_writes",
            "checkpoint_migrations",
        ):
            c.execute(text(f"SELECT count(*) FROM {table}"))
        c.execute(
            text(
                "INSERT INTO checkpoint_writes (thread_id, checkpoint_ns, checkpoint_id, task_id, "
                "idx, channel, type, blob) VALUES (:t, '', 'c', 't', 0, 'ch', 'x', '\\x00')"
            ),
            {"t": thread},
        )
        c.execute(text("DELETE FROM checkpoint_writes WHERE thread_id = :t"), {"t": thread})
    for sql in (
        "CREATE TABLE checkpoints.x (a int)",
        "CREATE TABLE public.x (a int)",
        "SELECT 1 FROM public.documents LIMIT 1",
        "SELECT 1 FROM public.llm_calls LIMIT 1",
        "SELECT 1 FROM public.sql_query_log LIMIT 1",
        "DROP TABLE checkpoints.checkpoints",
        "TRUNCATE checkpoints.checkpoints",
    ):
        assert denied(checkpointer_engine, sql), sql


def test_ledger_role_reaches_llm_calls_only(ledger_engine) -> None:
    with ledger_engine.begin() as c:
        c.execute(text("SELECT count(*) FROM llm_calls"))
    for sql in (
        "SELECT 1 FROM documents LIMIT 1",
        "SELECT 1 FROM sql_query_log LIMIT 1",
        "SELECT 1 FROM checkpoints.checkpoints LIMIT 1",
        "DELETE FROM llm_calls",
        "TRUNCATE llm_calls",
        "CREATE TABLE x (a int)",
    ):
        assert denied(ledger_engine, sql), sql


def test_the_gateway_ledger_runs_as_the_ledger_role() -> None:
    from rflens.db import get_ledger_engine

    with get_ledger_engine().connect() as c:
        assert c.execute(text("SELECT current_user")).scalar_one() == "rflens_llm_ledger"


# --- run_query returns the id of its sql_query_log row ------------------------------------------


@pytest.fixture(scope="module")
def sql_server():
    cfg = get_sql_mcp_settings()
    reader, logger = create_engine(cfg.reader_url()), create_engine(cfg.logger_url())
    yield create_sql_server(reader.connect, logger.connect, cfg)
    reader.dispose()
    logger.dispose()


@pytest.fixture
def question() -> Iterator[str]:
    value = f"query-id test {uuid.uuid4().hex}"
    yield value
    with get_engine().begin() as c:
        c.execute(text("DELETE FROM sql_query_log WHERE question = :q"), {"q": value})


def test_run_query_returns_the_log_row_id(sql_server, question) -> None:
    tools = SqlMcpTools(sql_server)
    first = tools.run_query("SELECT 1 AS a", question)
    second = tools.run_query("SELECT 2 AS a", question)
    assert first.ok and second.ok and second.query_id > first.query_id
    with get_engine().connect() as c:
        rows = c.execute(
            text("SELECT id, sql_text FROM sql_query_log WHERE question = :q ORDER BY id"),
            {"q": question},
        ).all()
    assert [r.id for r in rows] == [first.query_id, second.query_id]
    assert rows[1].sql_text == "SELECT 2 AS a"


def test_a_rejected_query_has_no_query_id(sql_server, question) -> None:
    outcome = SqlMcpTools(sql_server).run_query("SELECT * FROM uls_en", question)
    assert not outcome.ok and outcome.query_id is None


# --- the per-question counter is seeded from the ledger ---------------------------------------


def ledger_row(run_id: str, question_id: str, status: str, attempt: int = 1) -> None:
    with get_engine().begin() as c:
        c.execute(
            text(
                "INSERT INTO llm_calls (run_id, question_id, attempt, provider, model, purpose, "
                "status) VALUES (:r, :q, :a, 'fake', 'fake-1', 'planner', :s)"
            ),
            {"r": run_id, "q": question_id, "a": attempt, "s": status},
        )


@pytest.fixture
def qid() -> Iterator[str]:
    value = f"seed-{uuid.uuid4().hex[:8]}"
    yield value
    with get_engine().begin() as c:
        c.execute(text("DELETE FROM llm_calls WHERE question_id = :q"), {"q": value})


def make_gateway() -> Gateway:
    settings = LlmSettings(_env_file=None, llm_provider="fake", llm_fallback="")
    return Gateway(
        settings,
        {"fake": FakeProvider()},
        Ledger(get_engine(), settings.llm_budget_usd, 85.7),
        PRICING,
        rate_limiter=RateLimiter(10_000),
        sleep=lambda s: None,
        emit=lambda line: None,
    )


def test_question_attempts_counts_calls_that_reached_a_provider(qid) -> None:
    gateway = make_gateway()
    assert gateway.ledger.question_attempts(qid) == 0
    for attempt, status in enumerate(("ok", "error", "pending", "blocked"), start=1):
        ledger_row("earlier", qid, status, attempt)
    assert gateway.ledger.question_attempts(qid) == 3  # a blocked row never reached a provider
    assert gateway.ledger.question_attempts("another-question") == 0


def test_seed_question_starts_the_counter_at_the_ledger_count(qid) -> None:
    for attempt in (1, 2):
        ledger_row("earlier", qid, "ok", attempt)
    run = make_gateway().run("seed-test")
    assert run.seed_question(qid) == 2 and run.guard.per_question[qid] == 2
    run.guard.per_question[qid] = 5
    run.seed_question(qid)
    assert run.guard.per_question[qid] == 5  # never lowered


# --- migration 0014: a NULL key is not an entity -----------------------------------------------


@pytest.mark.parametrize(
    ("table", "column"),
    [
        ("fcc_grantees", "country"),
        ("fcc_test_firms", "country"),
        ("v_uls_licences", "licensee_frn"),
    ],
)
def test_key_columns_say_that_a_null_key_is_not_an_entity(table, column) -> None:
    with get_engine().connect() as c:
        comment = c.execute(
            text(
                "SELECT col_description(a.attrelid, a.attnum) FROM pg_attribute a "
                "WHERE a.attrelid = CAST(:t AS regclass) AND a.attname = :c"
            ),
            {"t": table, "c": column},
        ).scalar_one()
    assert "A NULL key is not an entity" in comment and f"{column} IS NOT NULL" in comment
    assert comment.count("A NULL key is not an entity") == 1  # applied once, not stacked


def test_the_sql_prompt_tells_the_agent_to_leave_out_null_keys() -> None:
    from rflens.agents.sql_prompt import RULES

    assert "leave out rows whose\n   key is NULL" in RULES and "IS NOT NULL" in RULES
    assert RULES.index("11. Rankings and counts") < RULES.index("12. Answer with the JSON object")
