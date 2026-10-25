"""SQL agent tests: the fake provider writes the "model" answers; sql-mcp is the real server (in
process, over MCP) on the real test database. No paid call is possible here.
"""

import json
import uuid
from collections.abc import Iterator

import pytest
from sqlalchemy import create_engine, text

from rflens.agents.sql_agent import MAX_CORRECTIONS, answer_question, tables_in
from rflens.agents.sql_prompt import ANSWER_SCHEMA, EXAMPLES, RULES, build_system_prompt
from rflens.agents.sql_tools import SCHEMA_QUESTION, SqlMcpTools
from rflens.config import LlmSettings, get_sql_mcp_settings
from rflens.db import get_engine
from rflens.eval.golden import load_golden, problems
from rflens.llm.base import CallCapExceeded, Pricing, ProviderError, Rate, Usage
from rflens.llm.fake import FakeProvider
from rflens.llm.gateway import Gateway
from rflens.llm.guards import RateLimiter
from rflens.llm.ledger import Ledger
from rflens.mcp.sql_guard import ALLOWED_RELATIONS, check_query
from rflens.mcp.sql_server import create_sql_server

PRICING = Pricing(
    {"fake": {"fake-1": {"standard": Rate(1.0, 1.0)}}}, usd_to_inr=85.7
)  # $1 per 1M tokens


def ans(sql="", refuse=False, reason="", tables=(), assumptions=()) -> str:
    return json.dumps(
        {
            "refuse": refuse,
            "refuse_reason": reason,
            "sql": sql,
            "tables_used": list(tables),
            "assumptions": list(assumptions),
        }
    )


BAD_TABLE = ans("SELECT * FROM uls_en")  # rejected by sql-mcp: not on the allow-list
BAD_COLUMN = ans("SELECT no_such_column FROM uls_hd")  # a Postgres error
GOOD = ans("SELECT 41 + 1 AS answer")


@pytest.fixture(scope="module")
def server():
    reader = create_engine(get_sql_mcp_settings().reader_url())
    logger = create_engine(get_sql_mcp_settings().logger_url())
    cfg = get_sql_mcp_settings()
    yield create_sql_server(reader.connect, logger.connect, cfg)
    reader.dispose()
    logger.dispose()


@pytest.fixture
def tools(server) -> SqlMcpTools:
    return SqlMcpTools(server)


@pytest.fixture
def qtext() -> Iterator[str]:
    """A unique question text; its sql_query_log rows (and the schema-context rows) are removed."""
    value = f"agent-test {uuid.uuid4().hex}"
    yield value
    with get_engine().begin() as c:
        c.execute(text("DELETE FROM sql_query_log WHERE question = :q"), {"q": value})
        c.execute(text("DELETE FROM sql_query_log WHERE question = :q"), {"q": SCHEMA_QUESTION})
        c.execute(text("DELETE FROM llm_calls"))


class Rig:
    def __init__(self, provider: FakeProvider, **settings_kw):
        settings_kw.setdefault("llm_fallback", "")
        self.settings = LlmSettings(_env_file=None, llm_provider=provider.name, **settings_kw)
        self.provider = provider
        self.gateway = Gateway(
            self.settings,
            {provider.name: provider},
            Ledger(get_engine(), self.settings.llm_budget_usd, 85.7),
            PRICING,
            rate_limiter=RateLimiter(10_000),
            sleep=lambda s: None,
            emit=lambda line: None,
        )

    def run(self):
        return self.gateway.run("agent-test")


def log_rows(question: str) -> list:
    with get_engine().connect() as c:
        return list(
            c.execute(
                text(
                    "SELECT tool, status, rejection_reason, sql_text FROM sql_query_log "
                    "WHERE question = :q ORDER BY id"
                ),
                {"q": question},
            )
        )


def ask(rig: Rig, tools: SqlMcpTools, question: str, qid="q1", **kw):
    with rig.run() as run:
        return answer_question(question, run=run, tools=tools, question_id=qid, **kw)


# --- the answered path --------------------------------------------------------------------------


def test_answered_path_runs_the_sql_through_sql_mcp_and_links_it_to_the_question(
    tools, qtext
) -> None:
    rig = Rig(FakeProvider(script=[GOOD]))
    r = ask(rig, tools, qtext)
    assert r.status == "answered" and r.sql == "SELECT 41 + 1 AS answer"
    assert r.columns == ["answer"] and r.rows == [[42]] and r.row_count == 1
    assert (r.calls, r.corrections, r.cost_usd) == (1, 0, 0)
    assert r.run_id.startswith("agent-test-")
    (row,) = log_rows(qtext)  # sql_query_log has the SQL next to the question text
    assert (row.tool, row.status, row.sql_text) == ("run_query", "ok", "SELECT 41 + 1 AS answer")


def test_the_prompt_carries_rules_examples_schema_comments_and_the_answer_schema(
    tools, qtext
) -> None:
    provider = FakeProvider(script=[GOOD])
    ask(Rig(provider), tools, qtext)
    messages, schema, _max_tokens = provider.calls[0]
    system = messages[0].content
    assert [m.role for m in messages] == ["system", "user"] and messages[1].content == qtext
    assert schema == ANSWER_SCHEMA
    assert RULES in system and "Never guess or invent units" in system
    assert system.count("\nQuestion: ") == len(EXAMPLES) == 6
    for relation in ALLOWED_RELATIONS:
        assert f"== {relation} (" in system
    assert "== uls_en" not in system  # not exposed by sql-mcp, so not in the context
    assert "ASSUMED MHz" in system and "UNKNOWN" in system  # unit caveats from the comments
    assert "open question a" in system


def test_tables_used_comes_from_the_sql_not_from_the_model(tools, qtext) -> None:
    sql = "WITH a AS (SELECT run_id FROM secom_runs) SELECT count(*) AS n FROM a"
    r = ask(Rig(FakeProvider(script=[ans(sql, tables=["wrong_table"])])), tools, qtext)
    assert r.tables_used == ["secom_runs"]
    assert tables_in("not sql at all (", ["b", "a", "a"]) == ["a", "b"]  # unparsable: model's list


def test_assumptions_become_notes_and_an_empty_result_is_still_answered(tools, qtext) -> None:
    sql = ans("SELECT 1 AS n WHERE false", assumptions=["The ERP unit is unknown.", ""])
    r = ask(Rig(FakeProvider(script=[sql])), tools, qtext)
    assert r.status == "answered" and r.rows == [] and r.row_count == 0
    assert r.notes == ["The ERP unit is unknown."]


def test_a_cut_off_result_is_noted(tools, qtext) -> None:
    series = "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r WHERE n < 500) "
    r = ask(Rig(FakeProvider(script=[ans(series + "SELECT n FROM r")])), tools, qtext)
    assert r.status == "answered" and r.row_count == 100 and r.truncated
    assert any("row limit" in n for n in r.notes)


def test_a_result_exactly_at_its_limit_is_not_reported_as_cut_off(tools, qtext) -> None:
    sql = "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r WHERE n < 3) "
    exact = ask(Rig(FakeProvider(script=[ans(sql + "SELECT n FROM r LIMIT 3")])), tools, qtext)
    assert exact.row_count == 3 and exact.truncated is False
    assert not any("row limit" in n for n in exact.notes)
    cut = ask(
        Rig(FakeProvider(script=[ans(sql.replace("< 3", "< 4") + "SELECT n FROM r LIMIT 3")])),
        tools,
        qtext,
    )
    assert cut.row_count == 3 and cut.truncated is True
    assert any("row limit" in n for n in cut.notes)


def test_the_schema_context_is_built_once_per_run(server, qtext) -> None:
    tools = SqlMcpTools(server)
    rig = Rig(FakeProvider(script=[GOOD, GOOD.replace("42", "43").replace("41 + 1", "41 + 2")]))
    with rig.run() as run:
        answer_question(qtext, run=run, tools=tools, question_id="a")
        answer_question(qtext + "2", run=run, tools=tools, question_id="b")
    with get_engine().connect() as c:
        listed = c.execute(
            text("SELECT count(*) FROM sql_query_log WHERE tool='list_tables' AND question = :q"),
            {"q": SCHEMA_QUESTION},
        ).scalar_one()
    assert listed == 1
    with get_engine().begin() as c:
        c.execute(text("DELETE FROM sql_query_log WHERE question = :q"), {"q": qtext + "2"})


# --- corrections --------------------------------------------------------------------------------


def test_a_sql_mcp_rejection_is_sent_back_once_and_the_corrected_sql_answers(tools, qtext) -> None:
    provider = FakeProvider(script=[BAD_TABLE, GOOD])
    r = ask(Rig(provider), tools, qtext)
    assert r.status == "answered" and r.rows == [[42]]
    assert (r.calls, r.corrections) == (2, 1)
    second = provider.calls[1][0]
    assert [m.role for m in second] == ["system", "user", "assistant", "user"]
    assert second[2].content == BAD_TABLE  # what the model said
    assert "rejected (table_not_allowed)" in second[3].content  # what sql-mcp said
    assert "Fix it" in second[3].content
    assert [(r_.status, r_.tool) for r_ in log_rows(qtext)] == [
        ("rejected", "run_query"),
        ("ok", "run_query"),
    ]


def test_a_postgres_error_is_sent_back_too(tools, qtext) -> None:
    provider = FakeProvider(script=[BAD_COLUMN, GOOD])
    r = ask(Rig(provider), tools, qtext)
    assert r.status == "answered" and r.corrections == 1
    assert "no_such_column" in provider.calls[1][0][-1].content
    assert [x.status for x in log_rows(qtext)] == ["error", "ok"]


def test_at_most_two_corrections_then_the_question_fails(tools, qtext) -> None:
    provider = FakeProvider(script=[BAD_TABLE, BAD_COLUMN, BAD_TABLE, GOOD, GOOD])
    r = ask(Rig(provider), tools, qtext)
    assert MAX_CORRECTIONS == 2
    assert r.status == "failed" and r.corrections == 2
    assert len(provider.calls) == 3  # one answer and two corrections, never a fourth
    assert any("gave up after 2 correction" in n for n in r.notes)
    assert r.sql == "SELECT * FROM uls_en"  # the last SQL it tried, for the report
    assert len(log_rows(qtext)) == 3


def test_the_per_question_cap_is_the_hard_limit(tools, qtext) -> None:
    provider = FakeProvider(script=[BAD_TABLE] * 5)
    r = ask(Rig(provider, llm_max_calls_per_question=2), tools, qtext)
    assert r.status == "failed" and len(provider.calls) == 2
    assert any("per-question call cap" in n for n in r.notes) and r.calls == 2


def test_the_run_cap_is_not_swallowed(tools, qtext) -> None:
    provider = FakeProvider(script=[BAD_TABLE] * 5)
    rig = Rig(provider, llm_max_calls_per_run=1)
    with rig.run() as run, pytest.raises(CallCapExceeded) as err:
        answer_question(qtext, run=run, tools=tools, question_id="q1")
    assert err.value.scope == "run" and len(provider.calls) == 1


def test_a_model_that_returns_broken_json_gets_a_correction(tools, qtext) -> None:
    provider = FakeProvider(script=["{not json", GOOD])
    r = ask(Rig(provider), tools, qtext)
    assert r.status == "answered" and r.corrections == 1
    assert "not valid JSON" in provider.calls[1][0][-1].content


@pytest.mark.parametrize(
    ("first", "complaint"),
    [
        (ans(""), "sql is empty"),
        (ans(refuse=True), "no refuse_reason"),
        ('{"sql": "SELECT 1"}', "required fields"),
        ("[1, 2]", "required fields"),
    ],
)
def test_unusable_answers_are_corrected(tools, qtext, first: str, complaint: str) -> None:
    provider = FakeProvider(script=[first, GOOD])
    r = ask(Rig(provider), tools, qtext)
    assert r.status == "answered" and r.corrections == 1
    assert complaint in provider.calls[1][0][-1].content
    assert log_rows(qtext)[0].tool == "run_query" and len(log_rows(qtext)) == 1  # first never ran


def test_a_provider_failure_ends_the_question_as_failed(tools, qtext) -> None:
    r = ask(Rig(FakeProvider(script=[ProviderError("refused", 400)])), tools, qtext)
    assert r.status == "failed" and any("model call failed" in n for n in r.notes)
    assert log_rows(qtext) == []


# --- refusal ------------------------------------------------------------------------------------


def test_refusal_runs_no_sql_and_reports_the_reason(tools, qtext) -> None:
    reply = ans(refuse=True, reason="This asks to modify data.", assumptions=["read-only"])
    provider = FakeProvider(script=[reply])
    r = ask(Rig(provider), tools, qtext)
    assert r.status == "refused" and r.sql is None and r.rows == []
    assert r.notes == ["This asks to modify data.", "read-only"]
    assert (r.calls, r.corrections) == (1, 0) and len(provider.calls) == 1
    assert log_rows(qtext) == []  # nothing reached run_query


def test_a_refusal_after_a_rejected_attempt_is_still_a_refusal(tools, qtext) -> None:
    provider = FakeProvider(
        script=[ans("DELETE FROM uls_hd"), ans(refuse=True, reason="read-only")]
    )
    r = ask(Rig(provider), tools, qtext)
    assert r.status == "refused" and r.corrections == 1
    assert "rejected (not_select)" in provider.calls[1][0][-1].content
    assert [x.status for x in log_rows(qtext)] == ["rejected"]


# --- accounting ---------------------------------------------------------------------------------


def test_calls_tokens_and_cost_are_counted_per_question_within_one_run(tools, qtext) -> None:
    provider = FakeProvider(
        is_paid=True, script=[BAD_TABLE, GOOD, GOOD.replace("41 + 1", "40 + 2")],
        usage=Usage(10, 5, 2),
    )  # fmt: skip
    rig = Rig(provider)
    with rig.run() as run:
        first = answer_question(qtext, run=run, tools=tools, question_id="a", max_output_tokens=20)
        second = answer_question(qtext, run=run, tools=tools, question_id="b", max_output_tokens=20)
    assert (first.calls, first.input_tokens, first.output_tokens, first.thinking_tokens) == (
        2, 20, 10, 4,
    )  # fmt: skip
    assert first.cost_usd == pytest.approx(34e-6)  # 2 calls x 17 tokens at $1 per 1M
    assert (second.calls, second.cost_usd) == (1, pytest.approx(17e-6))
    assert run.cost_usd == pytest.approx(51e-6)


# --- the prompt and the golden file -------------------------------------------------------------


def test_examples_are_valid_distinct_from_the_golden_set_and_balanced() -> None:
    golden = load_golden()
    assert sum(1 for _, a in EXAMPLES if a["refuse"]) == 2 and len(EXAMPLES) == 6
    prompt = build_system_prompt("schema")
    for q in golden:
        assert q.question not in prompt, f"golden question {q.id} appears in the prompt"
        if q.reference_sql:
            assert q.reference_sql not in prompt
    golden_sql = {" ".join(q.reference_sql.split()).lower() for q in golden if q.reference_sql}
    for _question, answer in EXAMPLES:
        if not answer["refuse"]:
            check_query(answer["sql"])  # passes the sql-mcp guard
            assert " ".join(answer["sql"].split()).lower() not in golden_sql
            assert tables_in(answer["sql"], []) == sorted(answer["tables_used"])


def test_the_example_sql_runs_through_sql_mcp(tools, qtext) -> None:
    for question, answer in EXAMPLES:
        if not answer["refuse"]:
            outcome = tools.run_query(answer["sql"], qtext)
            assert outcome.ok, (question, outcome.error)


def test_golden_file_structure_and_contents(tools, qtext) -> None:
    golden = load_golden()
    assert problems(golden) == []
    by_type = {t: [q for q in golden if q.type == t] for t in ("sql", "refuse")}
    assert (len(by_type["sql"]), len(by_type["refuse"]), len(golden)) == (19, 6, 52)
    by_id = {q.id: q for q in golden}
    assert by_id["sql-10"].question == (
        "How many cellular licences and how many spectrum leases expired in 2020?"
    )
    assert by_id["sql-10"].reference_result.rows == [[2, 26]]  # licences, leases: two columns
    assert by_id["sql-05"].reference_result.rows == [[1732]]
    assert "NOT is_lease" in by_id["sql-05"].reference_sql
    assert (
        "spectrum leases" in by_id["sql-05"].notes and "'A' assumed active" in by_id["sql-05"].notes
    )
    assert by_id["sql-06"].question == (
        "Which 3 licensees, counted by FCC Registration Number, hold the most active cellular "
        "licences (excluding spectrum leases)?"
    )
    top3 = by_id["sql-06"].reference_result.rows
    assert [r[0] for r in top3] == ["0003290673", "0003291192", "0014980726"]
    assert top3[2][1] == 283 and "278" in by_id["sql-06"].notes  # rank 4 is lower: no tie
    assert by_id["sql-15"].reference_result.rows == [["AST & Science, LLC", 1439]]
    assert "446" in by_id["sql-15"].notes  # rank 2 is lower
    # held-out questions for the FRN comment fix (migration 0011)
    assert by_id["sql-16"].reference_result.rows == [[154]]  # by name it would be 166
    assert (
        "NOT is_lease" in by_id["sql-16"].reference_sql
        and "DISTINCT licensee_frn" in by_id["sql-16"].reference_sql
    )
    assert by_id["sql-17"].reference_result.rows == [["0002210086", 11]]
    assert "licensee_frn IS NOT NULL" in by_id["sql-17"].reference_sql
    assert "Rank 1 has 11, rank 2 has 5" in by_id["sql-17"].notes
    # human-written questions are reviewed; assistant drafts wait for review
    assert {q.status for q in golden if q.author == "human"} == {"reviewed"}
    assert {q.status for q in golden if q.author == "assistant"} == {"draft"}
    assert next(q for q in golden if q.id == "sql-08").question == (
        "How many licensed cellular transmitter locations are in Texas?"
    )
    top8 = next(q for q in golden if q.id == "sql-03").reference_result
    assert len(top8.rows) == 8 and top8.rows[7][1] == 1341  # rank 8; rank 9 (1,018) is outside
    assert [q.question for q in golden[:3]] == [
        "How many SECOM production runs failed?",
        "How many SECOM runs failed in August 2008?",
        "Which 8 SECOM sensor features have the most missing values?",
    ]
    for q in by_type["sql"]:
        check_query(q.reference_sql)  # one SELECT over allow-listed relations
        ref = q.reference_result
        assert ref.rows and all(len(row) == len(ref.columns) for row in ref.rows), q.id
        assert tools.run_query(q.reference_sql, qtext).ok, q.id  # valid against the schema
    for q in by_type["refuse"]:
        assert q.expected_refusal and q.reference_sql is None
    ranked = {q.id for q in golden if q.ordered and q.type == "sql"}
    assert ranked == {
        "sql-03",
        "sql-06",
        "sql-07",
        "sql-11",
        "sql-13",
        "sql-15",
        "sql-17",
        "sql-18",
    }
    erp = next(q for q in golden if q.id == "sql-09")
    assert erp.must_mention == ["unknown"] and "UNKNOWN" in erp.notes
    assert all(
        "nterpretation" in q.notes or "ERP unit" in q.notes for q in by_type["sql"]
    )  # every sql question states how it was read
