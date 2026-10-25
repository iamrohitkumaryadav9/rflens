"""Orchestrator graph tests: the fake provider writes the model's answers; sql-mcp and docs-mcp are
the real servers (in process, over MCP) on the real test database; the checkpointer is the real
PostgresSaver connected as rflens_checkpointer. No paid call is possible here.
"""

import json
import subprocess
import sys
import uuid
from collections.abc import Iterator
from contextlib import contextmanager

import pytest
from sqlalchemy import Connection, create_engine, text

from rflens.agents.sql_tools import SCHEMA_QUESTION, SqlMcpTools
from rflens.config import GraphSettings, LlmSettings, get_sql_mcp_settings
from rflens.db import get_engine
from rflens.graph import app as graph_app
from rflens.graph.app import (
    CheckpointSchemaError,
    ask,
    build_graph,
    check_checkpoint_schema,
    final_text,
    open_checkpointer,
    resume,
)
from rflens.graph.docs_tools import DocsMcpTools
from rflens.graph.nodes import GraphDeps
from rflens.ingest.spec_fetch import SpecSource
from rflens.llm.base import Pricing, Rate
from rflens.llm.fake import FakeProvider
from rflens.llm.gateway import Gateway
from rflens.llm.guards import RateLimiter
from rflens.llm.ledger import Ledger
from rflens.mcp.docs_server import create_docs_server
from rflens.mcp.sql_server import create_sql_server
from rflens.retrieval.embedder import HashEmbedder
from rflens.retrieval.index import embed_pending, load_chunks

PRICING = Pricing({"fake": {"fake-1": {"standard": Rate(1.0, 1.0)}}}, usd_to_inr=85.7)
LTE = SpecSource("36.101", "36_series", "t-lte.zip", "19.0.0", 19, "1" * 64)
NR = SpecSource("38.101-1", "38_series", "t-nr.zip", "19.0.0", 19, "0" * 64)

BAND_TEXT = (
    "TS 36.101 V19.0.0 clause 5.5\n\nE-UTRA operating band 5: uplink 824 MHz to 849 MHz, "
    "downlink 869 MHz to 894 MHz."
)
NR_TEXT = "TS 38.101-1 V19.0.0 clause 6.2.1\n\nUE power class 3 maximum output power is 23 dBm."


def plan(route="sql", spec="none", reason="", doc_query="", sql_question="", bands=None):
    return json.dumps(
        {
            "route": route,
            "spec": spec,
            "doc_query": doc_query,
            "sql_question": sql_question,
            "bands": bands or [],
            "refuse_reason": reason,
        }
    )


def sql_answer(sql="", refuse=False, reason=""):
    return json.dumps(
        {
            "refuse": refuse,
            "refuse_reason": reason,
            "sql": sql,
            "tables_used": [],
            "assumptions": [],
        }
    )


def write(answer):
    return json.dumps({"answer": answer})


GOOD_SQL = sql_answer("SELECT 41 + 1 AS answer")
BAND_SQL = (
    "SELECT count(*) AS n FROM v_uls_frequencies f WHERE f.frequency_assigned >= {low} "
    "AND COALESCE(f.frequency_upper_band, f.frequency_assigned) <= {high}"
)
MIXED_DRAFT = (
    "Band 5 uplink spans 824 MHz to 849 MHz [D1]. The database count is 0 [S1], "
    "with the units of the ULS frequencies assumed to be MHz."
)


def chunk_row(source: SpecSource, clause: str, body: str) -> dict:
    return {
        "chunk_id": f"{source.spec}@19.0.0/{clause}/1",
        "spec": source.spec,
        "version": "19.0.0",
        "release": 19,
        "clause": clause,
        "clause_title": "Title",
        "parent_path": [],
        "parent_titles": [],
        "kind": "text",
        "position": 1,
        "table_caption": None,
        "table_number": None,
        "table_part": None,
        "source_file": "x.md",
        "text": body,
        "raw_text": body,
        "media": [],
        "image_placeholders": 0,
        "forced_split": False,
        "char_count": len(body),
    }


@pytest.fixture(scope="module")
def sql_server():
    cfg = get_sql_mcp_settings()
    reader, logger = create_engine(cfg.reader_url()), create_engine(cfg.logger_url())
    yield create_sql_server(reader.connect, logger.connect, cfg)
    reader.dispose()
    logger.dispose()


@pytest.fixture(scope="module")
def saver():
    with open_checkpointer() as s:
        yield s


@pytest.fixture
def docs_conn() -> Iterator[Connection]:
    """Synthetic spec chunks in a rolled-back transaction; docs-mcp is handed this connection."""
    with get_engine().connect() as connection:
        transaction = connection.begin()
        try:
            for source, clause, body in ((LTE, "5.5", BAND_TEXT), (NR, "6.2.1", NR_TEXT)):
                rows = [chunk_row(source, clause, body)]
                ids = load_chunks(connection, source, rows, prune=False)
                embed_pending(connection, HashEmbedder(), ids, progress=lambda s: None)
            yield connection
        finally:
            transaction.rollback()


@pytest.fixture
def tag() -> Iterator[str]:
    """A unique token in every question; its log rows, checkpoints and ledger rows are removed."""
    value = f"t{uuid.uuid4().hex[:12]}"
    yield value
    with get_engine().begin() as c:
        c.execute(text("DELETE FROM sql_query_log WHERE question LIKE :q"), {"q": f"%{value}%"})
        c.execute(text("DELETE FROM sql_query_log WHERE question = :q"), {"q": SCHEMA_QUESTION})
        c.execute(text("DELETE FROM llm_calls"))


class Env:
    def __init__(
        self,
        sql_server,
        docs_conn: Connection,
        saver,
        script: list,
        *,
        approval_mode="auto",
        paid=False,
        llm_kw=None,
        **graph_kw,
    ):
        self.provider = FakeProvider(script=script, is_paid=paid)
        self.llm = LlmSettings(
            _env_file=None, llm_provider="fake", llm_fallback="", **(llm_kw or {})
        )
        self.gateway = Gateway(
            self.llm,
            {"fake": self.provider},
            Ledger(get_engine(), self.llm.llm_budget_usd, 85.7),
            PRICING,
            rate_limiter=RateLimiter(10_000),
            sleep=lambda s: None,
            emit=lambda line: None,
        )
        self.run = self.gateway.run("graph-test")
        self.sql_tools = SqlMcpTools(sql_server)

        @contextmanager
        def connect():
            yield docs_conn

        self.docs_tools = DocsMcpTools(create_docs_server(connect, HashEmbedder(), "vector"))
        self.settings = GraphSettings(_env_file=None, **graph_kw)
        self.deps = GraphDeps(
            self.run, self.sql_tools, self.docs_tools, self.settings, approval_mode=approval_mode
        )
        self.saver = saver
        self.graph = build_graph(self.deps, saver)
        self.thread = f"graph-test-{uuid.uuid4().hex[:10]}"

    def ask(self, question: str):
        return ask(self.graph, question, self.thread)

    def close(self) -> None:
        self.saver.delete_thread(self.thread)


@pytest.fixture
def make_env(sql_server, docs_conn, saver):
    made: list[Env] = []

    def make(script, **kw) -> Env:
        env = Env(sql_server, docs_conn, saver, script, **kw)
        made.append(env)
        return env

    yield make
    for env in made:
        env.close()


# --- the four routes ----------------------------------------------------------------------------


def test_sql_route_answers_with_a_source_that_is_the_logged_query(make_env, tag) -> None:
    env = make_env([plan("sql"), GOOD_SQL, write("The answer is 42 [S1].")])
    question = f"What is forty one plus one? {tag}"
    out = env.ask(question)
    assert out.status == "answered" and out.state["route"] == "sql"
    assert out.state["verification"]["ok"]
    qid = out.state["sql"]["query_id"]
    with get_engine().connect() as c:
        logged = c.execute(
            text("SELECT id FROM sql_query_log WHERE question = :q AND tool = 'run_query'"),
            {"q": question},
        ).scalar_one()
    assert qid == logged
    assert f"[S1] sql-mcp query #{qid} (1 row(s)): SELECT 41 + 1 AS answer" in out.state["sources"]
    assert final_text(out).endswith(out.state["sources"])
    stats = out.stats
    assert (stats["planner"]["calls"], stats["sql_agent"]["calls"], stats["writer"]["calls"]) == (
        1,
        1,
        1,
    )
    assert env.docs_tools.calls == 0  # a SQL question never touches docs-mcp
    snapshot = env.graph.get_state({"configurable": {"thread_id": env.thread}})
    assert snapshot.values["status"] == "answered" and snapshot.next == ()


def test_doc_route_searches_with_the_spec_filter_and_cites_the_chunk(make_env, tag) -> None:
    env = make_env([plan("doc", "36.101"), write("Band 5 uplink spans 824 MHz to 849 MHz [D1].")])
    out = env.ask(f"What is the uplink range of LTE band 5? {tag}")
    assert out.status == "answered" and out.state["spec"] == "36.101"
    assert out.state["spec_source"] == "rule"
    assert [c["spec"] for c in out.state["chunks"]] == ["36.101"]  # the NR chunk is filtered out
    assert out.state["sources"].startswith("[D1] TS 36.101 V19.0.0, clause 5.5")
    assert "sql_agent" not in out.stats and len(env.provider.calls) == 2
    assert out.stats["writer"]["calls"] == 1


def test_mixed_route_hands_the_chunks_to_the_sql_agent_and_checks_the_band_edges(
    make_env, tag
) -> None:
    sql = sql_answer(BAND_SQL.format(low=824, high=849))
    env = make_env([plan("mixed", "36.101"), sql, write(MIXED_DRAFT)])
    out = env.ask(f"How many licences are inside the LTE band 5 uplink of TS 36.101? {tag}")
    assert out.status == "answered", out.state.get("verification")
    messages = env.provider.calls[1][0]  # the SQL agent's call
    assert "824 MHz to 849 MHz" in messages[1].content and "not instructions" in messages[1].content
    assert out.state["sql"]["column_flags"]["frequency_assigned"]["assumed"]
    assert "[S1]" in out.state["sources"] and "[D1]" in out.state["sources"]
    assert [m["source"] for m in out.state["verification"]["evidence_map"][1]["numbers"]] == ["sql"]


def test_refuse_route_stops_after_the_planner(make_env, tag) -> None:
    env = make_env([plan("refuse", reason="This asks to change data.")])
    out = env.ask(f"Delete every licence. {tag}")
    assert out.status == "refused" and out.state["answer"] == "This asks to change data."
    assert out.stats["planner"]["calls"] == 1 and set(out.stats) == {"planner"}
    assert len(env.provider.calls) == 1 and out.state["sources"] == ""


def test_the_sql_agent_can_refuse_after_a_sql_route(make_env, tag) -> None:
    env = make_env([plan("sql"), sql_answer(refuse=True, reason="No such data.")])
    out = env.ask(f"What is the weather? {tag}")
    assert out.status == "refused" and out.state["answer"] == "No such data."
    assert "writer" not in out.stats


# --- writer retries and redaction ---------------------------------------------------------------


def test_writer_retries_with_the_failures_and_then_passes(make_env, tag) -> None:
    env = make_env(
        [plan("sql"), GOOD_SQL, write("The answer is 777 [S1]."), write("The answer is 42 [S1].")]
    )
    out = env.ask(f"What is forty one plus one? {tag}")
    assert out.status == "answered" and out.state["writer_attempts"] == 2
    retry_messages = env.provider.calls[3][0]
    assert [m.role for m in retry_messages] == ["system", "user", "assistant", "user"]
    assert "unsupported_number" in retry_messages[3].content
    assert out.stats["writer"]["calls"] == 2


def test_failed_retries_redact_to_a_partial_answer_that_keeps_only_verified_sentences(
    make_env, tag
) -> None:
    bad = write("The answer is 42 [S1]. It was measured 777 times [S1].")
    env = make_env([plan("sql"), GOOD_SQL, bad, bad, bad])
    out = env.ask(f"What is forty one plus one? {tag}")
    assert out.status == "partial" and out.state["writer_attempts"] == 3
    assert out.stats["writer"]["calls"] == 3
    assert "The answer is 42 [S1]." in out.state["answer"]
    assert "777" not in out.state["answer"]
    assert "partial" in out.state["answer"] and "unsupported_number" in out.state["answer"]
    assert out.state["verification"]["redacted"] == ["unsupported_number"]
    assert "[S1]" in out.state["sources"]


def test_nothing_verifiable_falls_back_to_the_evidence_itself(make_env, tag) -> None:
    bad = write("It was measured 777 times [S1].")
    env = make_env([plan("sql"), GOOD_SQL, bad, bad, bad])
    out = env.ask(f"What is forty one plus one? {tag}")
    assert out.status == "partial"
    assert (
        "The query returned 1 row(s) [S1]" in out.state["answer"]
        and "- 42 [S1]" in (out.state["answer"])
    )
    assert "777" not in out.state["answer"]


def test_an_invented_band_edge_withholds_the_sql_result(make_env, tag) -> None:
    sql = sql_answer(BAND_SQL.format(low=814, high=849))
    draft = write(MIXED_DRAFT)
    # the SQL is sent back once (the same invented edge again), the writer is not retried
    env = make_env([plan("mixed", "36.101"), sql, draft, sql, draft])
    out = env.ask(f"How many licences are inside the LTE band 5 uplink of TS 36.101? {tag}")
    assert out.status == "partial"
    assert (out.stats["sql_agent"]["calls"], out.stats["writer"]["calls"]) == (2, 2)
    assert out.state["verification"]["redacted"] == ["unsupported_sql_literal"]
    assert out.state["sql"] is None and out.state["verification"]["withheld_sql_query_id"]
    assert "[S1]" not in out.state["answer"].split("Note:")[0]
    assert "[S1]" not in out.state["sources"] and "[D1]" in out.state["sources"]


# --- approval: interrupt and resume -------------------------------------------------------------

NR_QUESTION = "What is the maximum output power of power class 3 in NR band n78? {tag}"


def test_a_spec_conflict_interrupts_and_resumes_with_the_human_choice(make_env, tag) -> None:
    env = make_env(
        [plan("doc", "36.101"), write("Power class 3 allows 23 dBm [D1].")],
        approval_mode="interrupt",
    )
    out = env.ask(NR_QUESTION.format(tag=tag))
    assert out.status is None and out.interrupt is not None
    assert out.interrupt["reasons"][0].startswith("spec_conflict")
    assert out.interrupt["proposed_spec"] == "38.101-1" and out.interrupt["spec_hint"] == "36.101"
    assert env.docs_tools.calls == 0  # nothing ran past the gate
    snapshot = env.graph.get_state({"configurable": {"thread_id": env.thread}})
    assert snapshot.next == ("approval",)

    out = resume(env.graph, env.thread, {"approve": True, "spec": "38.101-1"})
    assert out.interrupt is None and out.status == "answered"
    assert out.state["spec"] == "38.101-1" and out.state["spec_source"] == "human"
    assert out.state["approval"]["decision"] == {"approve": True, "by": "human"}
    assert [c["spec"] for c in out.state["chunks"]] == ["38.101-1"]
    assert out.stats["planner"]["calls"] == 1  # the planner did not run again on resume


def test_resume_works_from_a_new_graph_and_a_new_connection(make_env, tag) -> None:
    env = make_env(
        [plan("doc", "36.101"), write("Power class 3 allows 23 dBm [D1].")],
        approval_mode="interrupt",
    )
    out = env.ask(NR_QUESTION.format(tag=tag))
    assert out.interrupt is not None
    with open_checkpointer() as fresh:  # a restarted process: new connection, new graph objects
        again = build_graph(env.deps, fresh)
        out = resume(again, env.thread, {"approve": True, "spec": None})
    assert out.status == "answered" and out.state["spec"] is None
    assert sorted(c["spec"] for c in out.state["chunks"])[0] in ("36.101", "38.101-1")


def test_declining_the_approval_cancels_the_question(make_env, tag) -> None:
    env = make_env([plan("doc", "36.101")], approval_mode="interrupt")
    out = env.ask(NR_QUESTION.format(tag=tag))
    assert out.interrupt is not None
    out = resume(env.graph, env.thread, {"approve": False})
    assert out.status == "cancelled" and out.state["sources"] == ""
    assert env.docs_tools.calls == 0 and len(env.provider.calls) == 1


def test_a_question_mentioning_both_technologies_asks_which_spec(make_env, tag) -> None:
    env = make_env([plan("doc", "36.101")], approval_mode="interrupt")
    out = env.ask(f"How do NR n78 and LTE band 42 differ in power class 3? {tag}")
    assert out.interrupt is not None
    assert out.interrupt["reasons"][0].startswith("spec_ambiguous")
    assert out.interrupt["proposed_spec"] is None


def test_auto_approval_records_who_decided(make_env, tag) -> None:
    env = make_env(
        [plan("doc", "36.101"), write("Power class 3 allows 23 dBm [D1].")], approval_mode="auto"
    )
    out = env.ask(NR_QUESTION.format(tag=tag))
    assert out.interrupt is None and out.status == "answered"
    assert out.state["approval"]["decision"] == {"approve": True, "by": "auto"}
    assert out.state["approval"]["needed"] and out.state["spec"] == "38.101-1"


def test_a_planned_cost_over_the_threshold_asks_for_approval(make_env, tag) -> None:
    env = make_env(
        [plan("sql"), GOOD_SQL, write("The answer is 42 [S1].")],
        approval_mode="interrupt",
        paid=True,
        approval_cost_usd=1e-9,
    )
    out = env.ask(f"What is forty one plus one? {tag}")
    assert out.interrupt is not None and out.interrupt["reasons"][0].startswith("cost:")
    assert out.interrupt["planned_worst_case_usd"] > 1e-9
    out = resume(env.graph, env.thread, {"approve": True})
    assert out.status == "answered"


def test_a_cheap_plan_needs_no_approval(make_env, tag) -> None:
    env = make_env(
        [plan("sql"), GOOD_SQL, write("The answer is 42 [S1].")],
        approval_mode="interrupt",
        paid=True,
    )
    out = env.ask(f"What is forty one plus one? {tag}")
    assert out.status == "answered" and out.state["approval"]["needed"] is False


def test_ask_always_asks(make_env, tag) -> None:
    env = make_env([plan("sql")], approval_mode="interrupt", ask=True)
    out = env.ask(f"What is forty one plus one? {tag}")
    assert out.interrupt is not None and out.interrupt["reasons"] == ["ask: ASK is set"]


def test_a_remaining_budget_under_twice_the_planned_cost_asks(make_env, tag) -> None:
    env = make_env(
        [plan("sql")],
        approval_mode="interrupt",
        paid=True,
        llm_kw={"llm_budget_usd": 0.05},
        approval_cost_usd=10.0,
    )
    out = env.ask(f"What is forty one plus one? {tag}")
    assert out.interrupt is not None
    assert [r.split(":")[0] for r in out.interrupt["reasons"]] == ["budget"]


# --- planner doc_query / sql_question -----------------------------------------------------------


def test_the_doc_agent_searches_with_the_planners_doc_query(make_env, tag) -> None:
    env = make_env(
        [
            plan("doc", "36.101", doc_query="E-UTRA operating band 5 uplink frequency range"),
            write("Band 5 uplink spans 824 MHz to 849 MHz [D1]."),
        ]
    )
    seen: list[tuple] = []
    real = env.docs_tools.search
    env.docs_tools.search = lambda *a: (seen.append(a), real(*a))[1]
    question = f"Tell me the uplink range of LTE band 5 please {tag}"
    out = env.ask(question)
    assert out.status == "answered"
    assert seen[0][0] == "E-UTRA operating band 5 uplink frequency range" != question
    assert seen[0][1] == "36.101"
    assert out.state["plan"]["doc_query"] == seen[0][0]


def test_a_missing_doc_query_falls_back_to_the_question(make_env, tag) -> None:
    env = make_env([plan("doc", "36.101"), write("Band 5 uplink spans 824 MHz to 849 MHz [D1].")])
    seen: list[tuple] = []
    real = env.docs_tools.search
    env.docs_tools.search = lambda *a: (seen.append(a), real(*a))[1]
    question = f"Uplink range of LTE band 5 {tag}"
    env.ask(question)
    assert seen[0][0] == question


def test_the_sql_agent_of_a_mixed_question_gets_the_planners_sql_question(make_env, tag) -> None:
    sql = sql_answer(BAND_SQL.format(low=824, high=849))
    env = make_env(
        [
            plan(
                "mixed",
                "36.101",
                doc_query="E-UTRA operating band 5 uplink frequency range",
                sql_question=f"How many frequency records lie in the range given? {tag}",
            ),
            sql,
            write(MIXED_DRAFT),
        ]
    )
    out = env.ask(f"How many licences are inside the LTE band 5 uplink of TS 36.101? {tag}")
    assert out.status == "answered"
    sql_input = env.provider.calls[1][0][1].content
    assert sql_input.startswith(MIXED_Q.format(tag=tag))  # the original question, first
    assert "How many frequency records lie in the range given?" in sql_input  # and the restatement
    assert "original question above takes precedence" in sql_input
    assert out.state["verification"]["band_edges"] == [
        {
            "column": "frequency_assigned",
            "value": 824.0,
            "cited_chunk": "D1",
            "chunk_id": "36.101@19.0.0/5.5/1",
        },
        {
            "column": "frequency_upper_band",
            "value": 849.0,
            "cited_chunk": "D1",
            "chunk_id": "36.101@19.0.0/5.5/1",
        },
    ]


# --- caps ---------------------------------------------------------------------------------------


def test_the_per_question_cap_stops_the_graph_run(make_env, tag) -> None:
    env = make_env(
        [plan("sql"), GOOD_SQL, write("The answer is 42 [S1].")],
        orch_max_calls_per_question=2,
    )
    out = env.ask(f"What is forty one plus one? {tag}")
    assert out.status == "failed" and "per-question call cap" in out.state["answer"]
    assert len(env.provider.calls) == 2  # planner and SQL agent; the writer was refused
    assert env.run.guard.max_calls_per_question == 2


def test_the_cap_counts_calls_a_previous_process_already_made(make_env, tag) -> None:
    env = make_env([plan("sql")], orch_max_calls_per_question=2)
    with get_engine().begin() as c:
        for attempt in (1, 2):
            c.execute(
                text(
                    "INSERT INTO llm_calls (run_id, question_id, attempt, provider, model, "
                    "purpose, status) VALUES ('earlier', :q, :a, 'fake', 'fake-1', "
                    "'planner', 'ok')"
                ),
                {"q": env.thread, "a": attempt},
            )
    out = env.ask(f"What is forty one plus one? {tag}")
    assert out.status == "failed" and "per-question call cap" in out.state["answer"]
    assert env.provider.calls == []  # the resumed question could not start over


def test_the_orchestrated_cap_defaults_to_ten(make_env) -> None:
    env = make_env([])
    assert env.run.guard.max_calls_per_question == 10 > env.llm.llm_max_calls_per_question


# --- checkpoint store, imports --------------------------------------------------------------------


def test_the_checkpoint_schema_check_accepts_the_migrated_schema(saver) -> None:
    check_checkpoint_schema(saver)


def test_the_checkpoint_schema_check_refuses_an_older_schema(saver, monkeypatch) -> None:
    monkeypatch.setattr(graph_app.PostgresSaver, "MIGRATIONS", [*saver.MIGRATIONS, "SELECT 1"])
    with pytest.raises(CheckpointSchemaError, match="make migrate"):
        check_checkpoint_schema(saver)


def run_python(code: str, **env) -> subprocess.CompletedProcess:
    import os

    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={**os.environ, **env},
        timeout=120,
    )


def test_the_graph_imports_no_langchain_package_beyond_langchain_core() -> None:
    code = (
        "import sys, rflens.graph.app, rflens.graph.nodes;"
        "print(sorted({m.split('.')[0] for m in sys.modules if m.startswith('langchain')}))"
    )
    done = run_python(code)
    assert done.returncode == 0, done.stderr
    # langgraph requires langchain-core, which requires langchain-protocol: nothing else
    from importlib.metadata import requires

    needed = {"langchain_core", "langchain_protocol"}
    assert any(r.startswith("langchain-core") for r in requires("langgraph"))
    assert any(r.startswith("langchain-protocol") for r in requires("langchain-core"))
    assert done.stdout.strip() == str(sorted(needed))


def test_the_graph_imports_no_provider_sdk() -> None:
    code = (
        "import sys, rflens.graph.app;"
        "bad = [m for m in sys.modules if m.split('.')[0] in "
        "('openai', 'anthropic', 'google', 'ollama')];"
        "print(sorted(set(m.split('.')[0] for m in bad)))"
    )
    done = run_python(code)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "[]"


def test_importing_the_graph_turns_tracing_off_and_refuses_it_when_on() -> None:
    code = (
        "import os, rflens.graph;"
        "print(os.environ['LANGSMITH_TRACING'], os.environ['LANGGRAPH_STRICT_MSGPACK'])"
    )
    off = run_python(code, LANGSMITH_TRACING="false")
    assert off.returncode == 0, off.stderr
    assert off.stdout.split() == ["false", "true"]
    on = run_python("import rflens.graph", LANGSMITH_TRACING="true")
    assert on.returncode != 0 and "TracingEnabledError" in on.stderr


# --- band lookup (spec_operating_bands through sql-mcp) -------------------------------------------

BAND_ROWS = [  # synthetic rows of version 19.0.0, pointing at the synthetic test chunks
    ("36.101", "5", 824, 849, 869, 894, "FDD", "36.101@19.0.0/5.5/1"),
    ("36.101", "29", None, None, 717, 728, "FDD", "36.101@19.0.0/5.5/1"),
    ("38.101-1", "n5", 824, 849, 869, 894, "FDD", "38.101-1@19.0.0/6.2.1/1"),
    ("38.101-1", "n80", 1710, 1785, None, None, "SUL", "38.101-1@19.0.0/6.2.1/1"),
]
LOOKUP = [{"spec": "36.101", "band": "5"}]


@pytest.fixture
def band_rows() -> Iterator[None]:
    with get_engine().begin() as c:
        for spec, band, ul_low, ul_high, dl_low, dl_high, duplex, chunk in BAND_ROWS:
            c.execute(
                text(
                    "INSERT INTO spec_operating_bands (spec, version, band, ul_low_mhz, "
                    "ul_high_mhz, dl_low_mhz, dl_high_mhz, duplex_mode, source_chunk_id) "
                    "VALUES (:s, '19.0.0', :b, :a, :c, :d, :e, :m, :k)"
                ),
                {
                    "s": spec,
                    "b": band,
                    "a": ul_low,
                    "c": ul_high,
                    "d": dl_low,
                    "e": dl_high,
                    "m": duplex,
                    "k": chunk,
                },
            )
    yield
    with get_engine().begin() as c:
        c.execute(text("DELETE FROM spec_operating_bands WHERE version = '19.0.0'"))


def spy_search(env: Env) -> list[tuple]:
    seen: list[tuple] = []
    real = env.docs_tools.search
    env.docs_tools.search = lambda *a: (seen.append(a), real(*a))[1]
    return seen


MIXED_Q = "How many licences are inside the LTE band 5 uplink of TS 36.101? {tag}"


def test_named_bands_are_looked_up_not_searched(make_env, tag, band_rows) -> None:
    sql = sql_answer(BAND_SQL.format(low=824, high=849))
    env = make_env(
        [
            plan("mixed", "36.101", doc_query="never used", bands=LOOKUP),
            sql,
            write(MIXED_DRAFT),
        ]
    )
    seen = spy_search(env)
    out = env.ask(MIXED_Q.format(tag=tag))
    assert out.status == "answered", out.state.get("verification")
    assert seen == []  # no search_docs call
    (chunk,) = out.state["chunks"]
    assert chunk["chunk_id"] == "36.101@19.0.0/5.5/1" and chunk["ref"] == "D1"
    assert chunk["band_lines"] == [
        "TS 36.101 band 5: uplink 824-849 MHz, downlink 869-894 MHz, FDD"
    ]
    info = out.state["band_lookup"]
    assert (info["requested"], info["found"], info["missing"]) == (["36.101 5"], ["36.101 5"], [])
    # the lookup is a logged sql-mcp query, and the SQL agent got the rows as context
    with get_engine().connect() as c:
        logged = c.execute(
            text("SELECT id FROM sql_query_log WHERE question = :q AND tool = 'run_query'"),
            {"q": f"band lookup for: {MIXED_Q.format(tag=tag)}"},
        ).scalar_one()
    assert info["query_id"] == logged
    context = env.provider.calls[1][0][1].content
    assert "TS 36.101 band 5: uplink 824-849 MHz" in context and "operating bands (chunk" in context
    assert "E-UTRA operating band 5:" not in context  # the whole table text is not sent on
    assert out.state["verification"]["band_edges"] == [
        {
            "column": "frequency_assigned",
            "value": 824.0,
            "cited_chunk": "D1",
            "chunk_id": "36.101@19.0.0/5.5/1",
        },
        {
            "column": "frequency_upper_band",
            "value": 849.0,
            "cited_chunk": "D1",
            "chunk_id": "36.101@19.0.0/5.5/1",
        },
    ]
    assert out.stats["doc_agent"]["calls"] == 0


def test_without_named_bands_the_doc_agent_searches(make_env, tag, band_rows) -> None:
    sql = sql_answer(BAND_SQL.format(low=824, high=849))
    env = make_env(
        [
            plan("mixed", "36.101", doc_query="E-UTRA operating band 5 uplink", bands=[]),
            sql,
            write(MIXED_DRAFT),
        ]
    )
    seen = spy_search(env)
    out = env.ask(MIXED_Q.format(tag=tag))
    assert seen and seen[0][0] == "E-UTRA operating band 5 uplink"
    assert out.state.get("band_lookup") is None
    assert "band_lines" not in out.state["chunks"][0]


def test_a_band_that_does_not_exist_stops_the_question_before_any_sql_or_search(
    make_env, tag, band_rows
) -> None:
    env = make_env([plan("mixed", "36.101", bands=[{"spec": "36.101", "band": "99"}])])
    seen = spy_search(env)
    out = env.ask(MIXED_Q.format(tag=tag))
    assert out.status == "failed" and seen == []
    assert (
        "none of the named bands (36.101 99) is in the operating-band tables"
        in (out.state["answer"])
    )
    assert len(env.provider.calls) == 1  # the planner only
    assert out.state["band_lookup"]["missing"] == ["36.101 99"]


def test_a_missing_band_next_to_a_found_one_is_reported_to_the_writer(
    make_env, tag, band_rows
) -> None:
    sql = sql_answer(BAND_SQL.format(low=824, high=849))
    env = make_env(
        [
            plan("mixed", "36.101", bands=[*LOOKUP, {"spec": "36.101", "band": "98"}]),
            sql,
            write(MIXED_DRAFT),
        ]
    )
    out = env.ask(MIXED_Q.format(tag=tag))
    assert out.state["band_lookup"]["missing"] == ["36.101 98"]
    writer_prompt = env.provider.calls[2][0][1].content
    assert "Band lookup: 36.101 98 is not in the operating-band tables." in writer_prompt


def test_bands_without_an_uplink_or_downlink_are_described_with_n_a(
    make_env, tag, band_rows
) -> None:
    env = make_env(
        [
            plan(
                "mixed",
                "none",
                bands=[
                    {"spec": "36.101", "band": "29"},
                    {"spec": "38.101-1", "band": "n80"},
                    {"spec": "38.101-1", "band": "n5"},
                ],
            ),
            GOOD_SQL,
            write("The answer is 42 [S1]."),
        ]
    )
    out = env.ask(f"How many licences use band 29, n80 or n5? {tag}")
    lines = {c["chunk_id"]: c["band_lines"] for c in out.state["chunks"]}
    assert lines == {
        "36.101@19.0.0/5.5/1": ["TS 36.101 band 29: uplink N/A, downlink 717-728 MHz, FDD"],
        "38.101-1@19.0.0/6.2.1/1": [
            "TS 38.101-1 band n80: uplink 1710-1785 MHz, downlink N/A, SUL",
            "TS 38.101-1 band n5: uplink 824-849 MHz, downlink 869-894 MHz, FDD",
        ],
    }
    assert [c["ref"] for c in out.state["chunks"]] == ["D1", "D2"]


def test_the_planners_bands_are_only_kept_for_mixed_questions(make_env, tag, band_rows) -> None:
    env = make_env([plan("doc", "36.101", doc_query="uplink", bands=LOOKUP), write("Band 5 [D1].")])
    seen = spy_search(env)
    out = env.ask(f"What is the uplink range of LTE band 5? {tag}")
    assert out.state["plan"]["bands"] == [] and seen


def test_clean_bands_validates_and_infers() -> None:
    from rflens.graph.band_lookup import clean_bands, lookup_sql

    raw = [
        {"spec": "36.101", "band": "5"},
        {"spec": "36.101", "band": "5"},  # duplicate
        {"spec": "36.101", "band": "N5"},  # the label says NR
        {"spec": "38.101-1", "band": "5"},  # the label says LTE
        {"band": "Band 41"},  # no spec
        {"spec": "36.101", "band": "5'; DROP TABLE x; --"},
        {"spec": "36.101", "band": "5 OR 1=1"},
        {"spec": "36.101", "band": ""},
        {"spec": "36.101"},
        "5",
    ]
    assert clean_bands(raw) == [
        {"spec": "36.101", "band": "5"},
        {"spec": "38.101-1", "band": "n5"},
        {"spec": "36.101", "band": "41"},
    ]
    assert clean_bands("n5") == [] and clean_bands(None) == []
    assert len(clean_bands([{"spec": "36.101", "band": str(i)} for i in range(1, 20)])) == 6
    sql = lookup_sql(clean_bands(raw))
    assert "DROP" not in sql and sql.count("(spec = ") == 3
    with pytest.raises(ValueError):
        lookup_sql([{"spec": "36.101", "band": "5'"}])
    with pytest.raises(ValueError):
        lookup_sql([])


def sql_with_band_join(low: str, high: str, band: str = "5") -> str:
    return sql_answer(
        "SELECT count(*) AS n FROM v_uls_frequencies f "
        f"JOIN spec_operating_bands b ON b.spec = '36.101' AND b.band = '{band}' "
        f"WHERE f.frequency_assigned >= b.{low} AND "
        f"COALESCE(f.frequency_upper_band, f.frequency_assigned) <= b.{high}"
    )


JOIN_DRAFT = (
    "Band 5 uplink spans 824 MHz to 849 MHz [D1]. The database count is 0 [S1], "
    "with the units of the ULS frequencies assumed to be MHz."
)


def test_the_original_question_is_what_sql_query_log_records_for_a_mixed_question(
    make_env, tag, band_rows
) -> None:
    env = make_env(
        [
            plan("mixed", "36.101", bands=LOOKUP, sql_question=f"Count records in the range {tag}"),
            sql_with_band_join("ul_low_mhz", "ul_high_mhz"),
            write(JOIN_DRAFT),
        ]
    )
    question = MIXED_Q.format(tag=tag)
    out = env.ask(question)
    assert out.status == "answered", out.state["verification"]
    with get_engine().connect() as c:
        logged = c.execute(
            text("SELECT question FROM sql_query_log WHERE id = :i"),
            {"i": out.state["sql"]["query_id"]},
        ).scalar_one()
    assert logged == question


def test_a_join_on_the_band_table_needs_no_typed_edges(make_env, tag, band_rows) -> None:
    env = make_env(
        [
            plan("mixed", "36.101", bands=LOOKUP),
            sql_with_band_join("ul_low_mhz", "ul_high_mhz"),
            write(JOIN_DRAFT),
        ]
    )
    out = env.ask(MIXED_Q.format(tag=tag))
    assert out.status == "answered", out.state["verification"]
    verification = out.state["verification"]
    assert verification["band_edges"] == []  # no literal edges in this SQL
    assert verification["band_joins"] == [{"spec": "36.101", "band": "5"}]
    context = env.provider.calls[1][0][1].content
    assert "Join spec_operating_bands by spec and band" in context


def test_downlink_columns_for_an_uplink_question_are_refused_and_withheld(
    make_env, tag, band_rows
) -> None:
    wrong = sql_with_band_join("dl_low_mhz", "dl_high_mhz")
    env = make_env(
        [plan("mixed", "36.101", bands=LOOKUP), wrong, write(JOIN_DRAFT), wrong, write(JOIN_DRAFT)]
    )
    out = env.ask(MIXED_Q.format(tag=tag))
    assert out.status == "partial"
    assert out.state["verification"]["redacted"] == ["edge_side_mismatch"]
    assert out.state["sql"] is None
    assert "[S1]" not in out.state["sources"]


def test_every_writer_draft_and_its_failures_are_kept_in_the_checkpoint(make_env, tag) -> None:
    bad = write("It was measured 777 times [S1].")
    env = make_env([plan("sql"), GOOD_SQL, bad, write("The answer is 99 [S1]."), bad])
    out = env.ask(f"What is forty one plus one? {tag}")
    assert out.status == "partial"
    history = out.state["draft_history"]
    assert [h["attempt"] for h in history] == [1, 2, 3]
    assert [h["draft"] for h in history] == [
        "It was measured 777 times [S1].",
        "The answer is 99 [S1].",
        "It was measured 777 times [S1].",
    ]
    assert all(not h["ok"] and h["failures"][0]["type"] == "unsupported_number" for h in history)
    with open_checkpointer() as fresh:  # read back through another connection
        snapshot = build_graph(env.deps, fresh).get_state(
            {"configurable": {"thread_id": env.thread}}
        )
    assert snapshot.values["draft_history"] == history


# --- retry routing: SQL-side failures to the SQL agent, prose to the writer --------------------


def test_a_sql_side_failure_goes_back_to_the_sql_agent_once_with_the_failure(
    make_env, tag, band_rows
) -> None:
    wrong = sql_with_band_join("dl_low_mhz", "dl_high_mhz")  # downlink columns, uplink question
    right = sql_with_band_join("ul_low_mhz", "ul_high_mhz")
    env = make_env(
        [
            plan("mixed", "36.101", bands=LOOKUP),
            wrong,
            write(JOIN_DRAFT),
            right,
            write(JOIN_DRAFT),
        ]
    )
    out = env.ask(MIXED_Q.format(tag=tag))
    assert out.status == "answered", out.state["verification"]
    assert [c[0][0].role for c in env.provider.calls] == ["system"] * 5
    retry_input = env.provider.calls[3][0][1].content  # the second SQL attempt's user message
    assert "failed verification" in retry_input and "edge_side_mismatch" in retry_input
    assert "b.dl_low_mhz" in retry_input  # the previous SQL is shown
    assert out.state["sql_retries"] == 1 and out.state["sql_feedback"] == []
    stats = out.stats
    assert (stats["planner"]["calls"], stats["sql_agent"]["calls"], stats["writer"]["calls"]) == (
        1,
        2,
        2,
    )
    history = out.state["draft_history"]
    assert [(h["attempt"], h["ok"]) for h in history] == [(1, False), (2, True)]
    assert history[0]["failures"][0]["type"] == "edge_side_mismatch"
    assert "b.ul_low_mhz" in out.state["sql"]["executed_sql"]
    # the writer of the second round starts fresh: no correction message about the old SQL
    assert [m.role for m in env.provider.calls[4][0]] == ["system", "user"]


def test_a_prose_failure_goes_to_the_writer_and_not_to_the_sql_agent(make_env, tag) -> None:
    env = make_env(
        [
            plan("sql"),
            GOOD_SQL,
            write("It was measured 777 times [S1]."),
            write("The answer is 42 [S1]."),
        ]
    )
    out = env.ask(f"What is forty one plus one? {tag}")
    assert out.status == "answered"
    assert (out.stats["sql_agent"]["calls"], out.stats["writer"]["calls"]) == (1, 2)
    assert out.state.get("sql_retries", 0) == 0


def test_a_sql_failure_next_to_a_prose_failure_goes_to_the_writer_first(
    make_env, tag, band_rows
) -> None:
    wrong = sql_with_band_join("dl_low_mhz", "dl_high_mhz")
    both = write("The database count is 777 [S1]. Band 5 uplink spans 824 MHz to 849 MHz [D1].")
    env = make_env([plan("mixed", "36.101", bands=LOOKUP), wrong, both, both, both])
    out = env.ask(MIXED_Q.format(tag=tag))
    kinds = {f["type"] for f in out.state["draft_history"][0]["failures"]}
    assert {"edge_side_mismatch", "unsupported_number"} <= kinds
    assert out.stats["sql_agent"]["calls"] == 1  # mixed failures are not a SQL-only retry
    assert out.stats["writer"]["calls"] == 3 and out.status == "partial"


def test_the_sql_retry_is_bounded_by_the_per_question_cap(make_env, tag, band_rows) -> None:
    wrong = sql_with_band_join("dl_low_mhz", "dl_high_mhz")
    env = make_env(
        [plan("mixed", "36.101", bands=LOOKUP), wrong, write(JOIN_DRAFT), wrong, write(JOIN_DRAFT)],
        orch_max_calls_per_question=3,
    )
    out = env.ask(MIXED_Q.format(tag=tag))
    # planner, SQL agent, writer used the 3 calls; the SQL retry could not start
    assert out.status == "failed" and "per-question call cap" in out.state["answer"]
    assert len(env.provider.calls) == 3


# --- frn_name_grouping is a SQL-side failure: one SQL retry, then redact --------------------------

SPLIT_SQL = sql_answer(
    "SELECT licensee_frn, licensee_name, count(*) AS n FROM v_uls_licences "
    "GROUP BY licensee_frn, licensee_name HAVING count(*) < 0"
)
FRN_SQL = sql_answer(
    "SELECT licensee_frn, min(licensee_name) AS licensee, count(*) AS n FROM v_uls_licences "
    "GROUP BY licensee_frn HAVING count(*) < 0"
)


def test_grouping_by_name_and_frn_is_sent_back_to_the_sql_agent_once(make_env, tag) -> None:
    env = make_env(
        [
            plan("sql"),
            SPLIT_SQL,
            write("No licensee matched [S1]."),
            FRN_SQL,
            write("No licensee matched [S1]."),
        ]
    )
    out = env.ask(f"Which licensees hold the most licences? {tag}")
    assert out.status == "answered", out.state["verification"]
    assert out.state["sql_retries"] == 1 and out.stats["sql_agent"]["calls"] == 2
    retry_input = env.provider.calls[3][0][1].content
    assert "frn_name_grouping" in retry_input and "group by licensee_frn alone" in retry_input
    assert out.state["draft_history"][0]["failures"][0]["type"] == "frn_name_grouping"


def test_a_grouping_that_the_retry_does_not_fix_is_withheld(make_env, tag) -> None:
    env = make_env(
        [
            plan("sql"),
            SPLIT_SQL,
            write("No licensee matched [S1]."),
            SPLIT_SQL,
            write("No licensee matched [S1]."),
        ]
    )
    out = env.ask(f"Which licensees hold the most licences? {tag}")
    assert out.status == "partial"
    assert out.state["verification"]["redacted"] == ["frn_name_grouping"]
    assert out.state["sql"] is None and out.stats["writer"]["calls"] == 2
