"""The stats agent and the stats route of the graph.

Real pieces: Postgres (test database), analytics-mcp over MCP in process (its roles, validation and
audit log), the LangGraph graph with the real PostgresSaver. Faked: the model (FakeProvider, no paid
call is possible) and the sandbox container (FakeSandbox with canned analysis results, because the
container itself is tested by `make test-analytics`).
"""

# ruff: noqa: F811
# (the fixtures imported from analytics_support and api_support are used by name)

import json
import re
import uuid
from collections.abc import Iterator

import pytest
from sqlalchemy import create_engine, text

from analytics_support import (  # noqa: F401  (fixtures)
    COMPARE_60,
    RANKING,
    FakeSandbox,
    ok_outcome,
    secom_rows,
)
from api_support import docs_conn, sql_server  # noqa: F401  (fixtures)
from rflens.agents.analytics_tools import AnalyticsMcpTools, AnalyticsMcpUnavailable
from rflens.agents.sql_tools import SqlMcpTools
from rflens.agents.stats_agent import answer_stats
from rflens.agents.stats_prompt import MENU_TEXT, STATS_SCHEMA, STATS_SYSTEM
from rflens.analytics import menu
from rflens.analytics.sandbox import SandboxResult
from rflens.config import GraphSettings, LlmSettings, get_analytics_settings
from rflens.db import get_engine
from rflens.graph.app import ask, build_graph, open_checkpointer
from rflens.graph.docs_tools import DocsMcpTools
from rflens.graph.nodes import GraphDeps
from rflens.graph.view import thread_view
from rflens.llm.base import Message, Pricing, Rate, estimate_tokens
from rflens.llm.fake import FakeProvider
from rflens.llm.gateway import Gateway
from rflens.llm.guards import RateLimiter
from rflens.llm.ledger import Ledger
from rflens.mcp.analytics_server import create_analytics_server
from rflens.mcp.docs_server import create_docs_server
from rflens.retrieval.embedder import HashEmbedder

PRICING = Pricing({"fake": {"fake-1": {"standard": Rate(1.0, 1.0)}}}, usd_to_inr=85.7)


def stats_json(analysis="group_compare", params=None, refuse=False, reason=""):
    return json.dumps(
        {
            "refuse": refuse,
            "refuse_reason": reason,
            "analysis": "none" if refuse else analysis,
            "params_json": json.dumps({"feature": 60} if params is None else params),
        }
    )


def plan(route="stats", reason="", stats_question=""):
    return json.dumps(
        {
            "route": route,
            "spec": "none",
            "doc_query": "",
            "sql_question": "",
            "stats_question": stats_question,
            "bands": [],
            "refuse_reason": reason,
        }
    )


def write(answer):
    return json.dumps({"answer": answer})


GOOD_DRAFT = (
    "Feature 60 is higher in failing runs than in passing runs [A1]. "
    "The Mann-Whitney test used 1463 passing and 97 failing values, with missing values left "
    "out, and gives p = 3.4e-10 and a rank-biserial effect size of 0.38 [A1]. "
    "This is an association, not a cause."
)


@pytest.fixture(scope="module")
def reader_logger():
    cfg = get_analytics_settings()
    reader, logger = create_engine(cfg.reader_url()), create_engine(cfg.logger_url())
    yield reader, logger
    reader.dispose()
    logger.dispose()


@pytest.fixture
def tag() -> Iterator[str]:
    value = f"stats{uuid.uuid4().hex[:12]}"
    yield value
    with get_engine().begin() as c:
        c.execute(
            text("DELETE FROM analytics_call_log WHERE question LIKE :q"), {"q": f"%{value}%"}
        )
        c.execute(text("DELETE FROM analytics_call_log WHERE question = 'stats-agent menu'"))
        c.execute(text("DELETE FROM sql_query_log WHERE question LIKE :q"), {"q": f"%{value}%"})
        c.execute(text("DELETE FROM llm_calls"))


def make_gateway(script, paid=False):
    provider = FakeProvider(script=script, is_paid=paid)
    llm = LlmSettings(_env_file=None, llm_provider="fake", llm_fallback="")
    gateway = Gateway(
        llm,
        {"fake": provider},
        Ledger(get_engine(), llm.llm_budget_usd, 85.7),
        PRICING,
        rate_limiter=RateLimiter(10_000),
        sleep=lambda s: None,
        emit=lambda line: None,
    )
    return provider, gateway


def analytics_tools(reader_logger, sandbox, tmp_path):
    reader, logger = reader_logger
    cfg = get_analytics_settings().model_copy(update={"analytics_artifacts_dir": str(tmp_path)})
    return AnalyticsMcpTools(create_analytics_server(reader.connect, logger.connect, sandbox, cfg))


def calls_logged(tag):
    with get_engine().connect() as c:
        return list(
            c.execute(
                text(
                    "SELECT tool, status, reason FROM analytics_call_log "
                    "WHERE question LIKE :q ORDER BY id"
                ),
                {"q": f"%{tag}%"},
            )
        )


# --- the stats agent ----------------------------------------------------------------------------


def run_agent(script, reader_logger, tmp_path, tag, sandbox=None, question=None):
    provider, gateway = make_gateway(script)
    sandbox = sandbox or FakeSandbox(
        by_analysis={"group_compare": ok_outcome("group_compare", COMPARE_60)}
    )
    tools = analytics_tools(reader_logger, sandbox, tmp_path)
    run = gateway.run("stats-test")
    qid = f"q-{uuid.uuid4().hex[:8]}"
    result = answer_stats(
        question or f"Do passing and failing runs differ in feature 60? {tag}",
        run=run,
        tools=tools,
        question_id=qid,
    )
    return result, provider, sandbox


def test_the_agent_picks_an_analysis_and_gets_its_result(secom_rows, reader_logger, tmp_path, tag):
    result, provider, sandbox = run_agent([stats_json()], reader_logger, tmp_path, tag)
    assert result.status == "answered" and result.analysis == "group_compare"
    assert result.params["feature"] == 60 and result.params["test"] == "mann_whitney"
    assert result.result["p_value"] == 3.4e-10 and result.n_runs_used == 60
    assert result.calls == 1 and result.corrections == 0 and result.call_id
    (job,) = sandbox.jobs
    assert job["analysis"] == "group_compare" and "code" not in json.dumps(job)
    system = provider.calls[0][0][0].content
    assert "ONE statistical analysis" in system and "never write code" in system
    assert [(r.tool, r.status) for r in calls_logged(tag)] == [("group_compare", "ok")]


def test_the_agent_refuses_what_the_menu_cannot_do_without_calling_the_service(
    secom_rows, reader_logger, tmp_path, tag
):
    script = [stats_json(refuse=True, reason="Prediction needs a fitted model.")]
    result, _, sandbox = run_agent(script, reader_logger, tmp_path, tag)
    assert result.status == "refused" and result.notes == ["Prediction needs a fitted model."]
    assert sandbox.jobs == [] and calls_logged(tag) == [] and result.calls == 1


def test_one_correction_after_the_service_rejects_the_parameters(
    secom_rows, reader_logger, tmp_path, tag
):
    script = [stats_json(params={"feature": 999}), stats_json(params={"feature": 60})]
    result, provider, sandbox = run_agent(script, reader_logger, tmp_path, tag)
    assert result.status == "answered" and result.corrections == 1 and result.calls == 2
    retry = provider.calls[1][0]
    assert "The analysis service said" in retry[-1].content
    assert "less than or equal to 590" in retry[-1].content
    assert len(sandbox.jobs) == 1  # the rejected call never reached the sandbox
    assert [r.status for r in calls_logged(tag)] == ["rejected", "ok"]


def test_at_most_one_correction_then_the_agent_gives_up(secom_rows, reader_logger, tmp_path, tag):
    script = [stats_json(params={"feature": 999}), stats_json(params={"feature": 0})]
    result, provider, sandbox = run_agent(script, reader_logger, tmp_path, tag)
    assert result.status == "failed" and result.calls == 2 and len(provider.calls) == 2
    assert "gave up after 1 correction(s)" in result.notes[-1]
    assert sandbox.jobs == []


def test_unusable_model_output_is_corrected_the_same_way(secom_rows, reader_logger, tmp_path, tag):
    bad_json = json.dumps(
        {"refuse": False, "refuse_reason": "", "analysis": "describe", "params_json": "{not json"}
    )
    bad_name = json.dumps(
        {"refuse": False, "refuse_reason": "", "analysis": "python", "params_json": "{}"}
    )
    result, provider, _ = run_agent([bad_json, stats_json()], reader_logger, tmp_path, tag)
    assert result.status == "answered" and "not valid JSON" in provider.calls[1][0][-1].content
    result, provider, _ = run_agent([bad_name, stats_json()], reader_logger, tmp_path, tag)
    assert (
        result.status == "answered"
        and "analysis must be one of" in provider.calls[1][0][-1].content
    )


def test_an_analysis_the_data_cannot_answer_goes_back_once(
    secom_rows, reader_logger, tmp_path, tag
):
    too_few = SandboxResult(
        status="ok",
        payload={"ok": False, "error": {"code": "too_few_values", "message": "need 20 per group"}},
    )
    sandbox = FakeSandbox(by_analysis={"group_compare": too_few})
    script = [stats_json(), stats_json(refuse=True, reason="Too few values in one group.")]
    result, _, _ = run_agent(script, reader_logger, tmp_path, tag, sandbox)
    assert result.status == "refused" and result.corrections == 1 and result.calls == 2


def test_a_sandbox_failure_is_not_retried(secom_rows, reader_logger, tmp_path, tag):
    broken = SandboxResult(status="timeout", message="ran longer than 60 s")
    sandbox = FakeSandbox(by_analysis={"group_compare": broken})
    result, _, _ = run_agent([stats_json()], reader_logger, tmp_path, tag, sandbox)
    assert result.status == "failed" and result.calls == 1 and result.corrections == 0
    assert "timeout" in result.notes[0]


def test_an_unreachable_service_stops_the_agent():
    tools = AnalyticsMcpTools("http://127.0.0.1:9/mcp")  # nothing listens there
    _, gateway = make_gateway([stats_json()])
    with pytest.raises(AnalyticsMcpUnavailable, match="make analytics-mcp"):
        answer_stats("q", run=gateway.run("t"), tools=tools, question_id="q-unreachable")


def test_the_prompt_documents_every_analysis_and_parameter_of_the_menu():
    """The prompt text is written by hand; this fails when the menu and the prompt drift apart."""
    assert STATS_SCHEMA["properties"]["analysis"]["enum"] == [*menu.ANALYSES, "none"]
    for name, analysis in menu.ANALYSES.items():
        assert re.search(rf"^{name}\s", MENU_TEXT, re.M), name
        for field in analysis.model.model_fields:
            if field == "filters":
                continue
            assert field in MENU_TEXT, (name, field)
    for field in menu.Filters.model_fields:
        assert field in MENU_TEXT, field
    for op in ("<=", ">=", "==", "is_null", "not_null"):
        assert op in MENU_TEXT
    assert "refuse" in STATS_SYSTEM and "predict" in STATS_SYSTEM


# --- the graph ----------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def saver():
    with open_checkpointer() as s:
        yield s


class Env:
    def __init__(self, sql_server, docs_conn, saver, script, reader_logger, tmp_path, **kw):
        self.provider, self.gateway = make_gateway(script, paid=kw.pop("paid", False))
        self.sandbox = kw.pop("sandbox", None) or FakeSandbox(
            by_analysis={
                "group_compare": ok_outcome("group_compare", COMPARE_60),
                "feature_ranking": ok_outcome("feature_ranking", RANKING),
            }
        )
        self.run = self.gateway.run("graph-stats-test")
        from contextlib import contextmanager

        @contextmanager
        def connect():
            yield docs_conn

        tools = kw.pop("analytics", None)
        if tools is None:
            tools = analytics_tools(reader_logger, self.sandbox, tmp_path)
        self.deps = GraphDeps(
            self.run,
            SqlMcpTools(sql_server),
            DocsMcpTools(create_docs_server(connect, HashEmbedder(), "vector")),
            GraphSettings(_env_file=None, **kw),
            approval_mode="auto",
            analytics_tools=tools,
        )
        self.saver = saver
        self.graph = build_graph(self.deps, saver)
        self.thread = f"graph-stats-{uuid.uuid4().hex[:10]}"

    def ask(self, question):
        return ask(self.graph, question, self.thread)


@pytest.fixture
def make_env(sql_server, docs_conn, saver, reader_logger, tmp_path):
    made = []

    def make(script, **kw):
        env = Env(sql_server, docs_conn, saver, script, reader_logger, tmp_path, **kw)
        made.append(env)
        return env

    yield make
    for env in made:
        env.saver.delete_thread(env.thread)


def test_stats_route_from_question_to_cited_and_verified_answer(secom_rows, make_env, tag) -> None:
    env = make_env([plan("stats"), stats_json(), write(GOOD_DRAFT)])
    out = env.ask(f"Do passing and failing runs differ in feature 60? {tag}")
    assert out.status == "answered", out.state["verification"]
    assert out.state["route"] == "stats" and out.state["verification"]["ok"]
    stats = out.state["stats"]
    assert stats["status"] == "answered" and stats["analysis"] == "group_compare"
    (row,) = [r for r in calls_logged(tag) if r.tool == "group_compare"]
    assert row.status == "ok"
    assert f"[A1] analytics-mcp call #{stats['call_id']}: group_compare(" in out.state["sources"]
    assert "on SECOM, 60 run(s), extract " in out.state["sources"]
    assert set(out.stats) == {"planner", "stats_agent", "writer", "verifier"}
    assert out.stats["stats_agent"]["calls"] == 1 and out.stats["writer"]["calls"] == 1
    sources = {s["tag"]: s for s in thread_view("t", out.state)["source_items"]}
    assert sources["A1"]["kind"] == "stats" and sources["A1"]["cited"] is True
    assert sources["A1"]["result"]["p_value"] == 3.4e-10
    assert not out.state.get("sql") and env.deps.docs_tools.calls == 0  # only analytics-mcp ran
    evidence = env.provider.calls[2][0][1].content  # the writer's message
    assert "[A1] statistical analysis 'group_compare'" in evidence and "3.4e-10" in evidence


def test_the_planner_restatement_is_given_to_the_stats_agent(secom_rows, make_env, tag) -> None:
    env = make_env(
        [
            plan("stats", stats_question="Compare feature 60 by label"),
            stats_json(),
            write(GOOD_DRAFT),
        ]
    )
    env.ask(f"Is sensor 60 different for failures? {tag}")
    asked = env.provider.calls[1][0][1].content
    assert (
        "Is sensor 60 different for failures?" in asked and "Compare feature 60 by label" in asked
    )


def test_the_stats_agent_can_refuse_and_nothing_is_written(secom_rows, make_env, tag) -> None:
    refusal = stats_json(refuse=True, reason="This needs a trained model, which the menu lacks.")
    env = make_env([plan("stats"), refusal])
    out = env.ask(f"Train a model and predict the next lot. {tag}")
    assert out.status == "refused"
    assert out.state["answer"] == "This needs a trained model, which the menu lacks."
    assert "writer" not in out.stats and calls_logged(tag) == []


def test_the_planner_can_refuse_a_prediction_before_any_analysis(make_env, tag) -> None:
    env = make_env([plan("refuse", reason="Prediction is outside the fixed menu of analyses.")])
    out = env.ask(f"Predict whether the next lot fails. {tag}")
    assert out.status == "refused" and set(out.stats) == {"planner"}
    assert env.sandbox.jobs == []


def test_a_wrong_number_is_caught_and_the_writer_retries(secom_rows, make_env, tag) -> None:
    wrong = GOOD_DRAFT.replace("1463", "1500")
    env = make_env([plan("stats"), stats_json(), write(wrong), write(GOOD_DRAFT)])
    out = env.ask(f"Do passing and failing runs differ in feature 60? {tag}")
    assert out.status == "answered" and out.state["writer_attempts"] == 2
    first = out.state["draft_history"][0]
    assert not first["ok"] and {f["type"] for f in first["failures"]} == {
        "unsupported_number",
        "wrong_n",  # 1500 passing values is not a group size either
    }
    assert "1500 is in no evidence" in env.provider.calls[3][0][-1].content


def test_a_causal_claim_is_caught_and_the_writer_retries(secom_rows, make_env, tag) -> None:
    causal = GOOD_DRAFT.replace(
        "This is an association, not a cause.", "Feature 60 causes failures."
    )
    env = make_env([plan("stats"), stats_json(), write(causal), write(GOOD_DRAFT)])
    out = env.ask(f"Do passing and failing runs differ in feature 60? {tag}")
    assert out.status == "answered"
    assert {f["type"] for f in out.state["draft_history"][0]["failures"]} == {"causal_claim"}


def test_retries_used_up_gives_a_partial_answer_that_keeps_the_verified_sentences(
    secom_rows, make_env, tag
) -> None:
    bad = GOOD_DRAFT + " It is significant at the 1% level [A1]. Feature 60 causes failures."
    env = make_env([plan("stats"), stats_json(), write(bad), write(bad), write(bad)])
    out = env.ask(f"Do passing and failing runs differ in feature 60? {tag}")
    assert out.status == "partial" and "Feature 60 causes failures" not in out.state["answer"]
    assert "Feature 60 is higher in failing runs" in out.state["answer"]
    assert out.state["verification"]["redacted"] == ["causal_claim"]
    assert out.state["answer"].rstrip().endswith("(causal_claim).")


def test_a_failed_analysis_ends_the_question_without_a_writer_call(
    secom_rows, make_env, tag
) -> None:
    broken = SandboxResult(status="error", message="cannot run docker: boom")
    env = make_env([plan("stats"), stats_json()], sandbox=FakeSandbox(outcome=broken))
    out = env.ask(f"Do passing and failing runs differ in feature 60? {tag}")
    assert out.status == "failed" and "boom" in out.state["answer"]
    assert "writer" not in out.stats


def test_without_analytics_tools_the_stats_route_stops_with_a_clear_reason(
    secom_rows, make_env, saver, sql_server, docs_conn, reader_logger, tmp_path, tag
) -> None:
    env = make_env([plan("stats")])
    env.deps.analytics_tools = None
    out = env.ask(f"Rank the features. {tag}")
    assert out.status == "failed" and "analytics-mcp is not configured" in out.state["answer"]


def test_the_plan_cost_of_a_paid_stats_route_includes_the_stats_agent(
    secom_rows, make_env, tag
) -> None:
    env = make_env([plan("stats"), stats_json(), write(GOOD_DRAFT)], paid=True)
    question = f"Do passing and failing runs differ in feature 60? {tag}"
    out = env.ask(question)
    price = env.deps.run.gateway.pricing.worst_case_usd
    stats_in = estimate_tokens([Message.system(STATS_SYSTEM), Message.user(question)])
    expected = 2 * price("fake", "fake-1", "standard", stats_in, 400)  # 1 call + 1 correction
    expected += 3 * price(
        "fake", "fake-1", "standard", 18000 // 3 + 700, 1500
    )  # writer + 2 retries
    assert out.state["plan"]["planned_worst_case_usd"] == pytest.approx(expected, abs=2e-6)
    assert out.status == "answered"


def test_feature_ranking_answer_with_a_plot_artifact(secom_rows, make_env, tag, tmp_path) -> None:
    png = b"\x89PNG\r\n\x1a\nfake"
    sandbox = FakeSandbox(
        by_analysis={"plot": ok_outcome("plot", {"kind": "ranking_bar", "data": RANKING}, png)}
    )
    script = [
        plan("stats"),
        stats_json("plot", {"kind": "ranking_bar", "top_k": 2}),
        write(
            "The ranking of 4 tested features is plotted [A1]. Feature 60 leads with p = 3.4e-10 "
            "in 1463 passing and 97 failing runs [A1]."
        ),
    ]
    env = make_env(script, sandbox=sandbox)
    out = env.ask(f"Plot the top two features. {tag}")
    assert out.status == "answered", out.state["verification"]
    (art,) = out.state["stats"]["artifacts"]
    assert re.fullmatch(r"[0-9a-f]{64}", art["id"]) and (tmp_path / f"{art['id']}.png").exists()
    assert f"plot artifact {art['id']}" in out.state["sources"]


def test_a_wrong_n_is_caught_and_the_writer_retries(secom_rows, make_env, tag) -> None:
    """The live stats-01 case: the total number of runs given as the n of the test."""
    wrong = GOOD_DRAFT.replace(
        "The Mann-Whitney test used 1463 passing and 97 failing values",
        "The Mann-Whitney test (n=1567) used all runs",
    )
    env = make_env([plan("stats"), stats_json(), write(wrong), write(GOOD_DRAFT)])
    out = env.ask(f"Do passing and failing runs differ in feature 60? {tag}")
    assert out.status == "answered" and out.state["writer_attempts"] == 2
    first = out.state["draft_history"][0]
    assert not first["ok"] and "wrong_n" in {f["type"] for f in first["failures"]}
    assert "wrong_n: n = 1567 is not a number of values the test used" in (
        env.provider.calls[3][0][-1].content
    )
