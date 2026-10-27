"""The sql / stats boundary of the planner: the rules (graph/route_rules.py), the examples and text
in the planner prompt, and the graph with a fake provider that routes wrongly on purpose.

A fake provider cannot show that a real model routes well; it shows that the rules decide the
boundary when the model gets it wrong (as it did in run eval-e2e-aecd54a7), and that they leave
every other route alone. The real model is measured by `make eval-e2e`."""

# ruff: noqa: F811
# (the fixtures imported from analytics_support, api_support and test_stats are used by name)

import json

import pytest

from analytics_support import secom_rows  # noqa: F401  (fixture)
from api_support import docs_conn, sql_server  # noqa: F401  (fixtures)
from rflens.eval.golden import load_golden
from rflens.graph.prompts import (
    PLANNER_EXAMPLES,
    PLANNER_SYSTEM,
    STATS_WRITER_RULES,
    WRITER_SYSTEM,
    writer_system,
)
from rflens.graph.route_rules import boundary_route
from test_stats import (  # noqa: F401  (fixtures and helpers)
    GOOD_DRAFT,
    make_env,
    plan,
    reader_logger,
    saver,
    stats_json,
    tag,
    write,
)

GOLDEN = {q.id: q for q in load_golden()}


def sql_json(sql="SELECT 41 + 1 AS n"):
    return json.dumps(
        {"refuse": False, "refuse_reason": "", "sql": sql, "tables_used": [], "assumptions": []}
    )


# --- the rules -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "question",
    [
        "Which 8 SECOM sensor features have the most missing values?",  # the shape of sql-03
        "Which 5 SECOM features have the fewest missing values?",
        "What is the average value of SECOM feature 12 for failing runs versus passing runs?",
        "How many SECOM runs have no value for feature 33?",
        "What is the highest value of sensor 7 among failing runs?",
        "List the top 3 SECOM features by number of missing readings.",
        "What is the total number of SECOM production runs in July 2008?",
        "What is the minimum and maximum of feature 99?",
    ],
)
def test_simple_aggregates_over_secom_are_sql(question) -> None:
    assert boundary_route(question) == "sql"


@pytest.mark.parametrize(
    "question",
    [
        "Which three sensor features are most strongly associated with failing runs?",  # stats-01
        "Which 5 SECOM features show the strongest association with failing runs?",
        "Do passing and failing runs differ in feature 12, and how large is the difference?",
        "Is feature 210 significantly different between passing and failing runs?",
        "What is the effect size of feature 3 for pass versus fail?",
        "How is sensor 45 distributed? Give the quartiles and a histogram.",
        "Which features are most correlated with feature 15?",
        "Are there outliers in SECOM feature 88?",
        "Plot feature 60 for passing and failing runs.",
        "What is the average of feature 60, and is the pass versus fail difference significant?",
    ],
)
def test_tests_effects_associations_and_distributions_are_stats(question) -> None:
    assert boundary_route(question) == "stats"


@pytest.mark.parametrize(
    "question",
    [
        "How many active cellular licences does each company hold?",
        "What is the distribution of licence status codes in the ULS data?",  # not SECOM
        "Which licensees are most associated with band 5 frequencies?",
        "What is the average ERP of cellular transmitters?",
        "Which SECOM features matter?",  # neither an aggregate nor a statistics word
    ],
)
def test_other_questions_are_left_to_the_planner(question) -> None:
    assert boundary_route(question) is None


def test_the_rules_never_contradict_a_golden_question() -> None:
    for q in GOLDEN.values():
        decided = boundary_route(q.question)
        if q.type == "sql":
            assert decided in (None, "sql"), q.id
        elif q.type == "stats" and not q.expected_refusal:
            assert decided == "stats", q.id
        elif q.type in ("doc", "mixed", "refuse") or q.expected_refusal:
            assert decided is None, q.id  # the rules are only applied to a sql / stats route
    assert boundary_route(GOLDEN["sql-03"].question) == "sql"
    assert boundary_route(GOLDEN["sql-04"].question) == "sql"
    for qid in ("stats-01", "stats-02", "stats-03"):
        assert boundary_route(GOLDEN[qid].question) == "stats"


# --- the planner prompt ------------------------------------------------------------------------


def test_the_planner_prompt_states_the_boundary_and_gives_six_examples() -> None:
    assert len(PLANNER_EXAMPLES) == 6 and {r for _, r in PLANNER_EXAMPLES} == {"sql", "stats"}
    assert sum(r == "sql" for _, r in PLANNER_EXAMPLES) == 3
    for question, route in PLANNER_EXAMPLES:
        assert f'- "{question}" -> {route}' in PLANNER_SYSTEM
        assert boundary_route(question) == route, question  # the rules agree with every example
    golden = {q.question.lower() for q in GOLDEN.values()}
    assert not golden & {q.lower() for q, _ in PLANNER_EXAMPLES}
    for needle in (
        'Simple aggregates over any table are "sql"',
        "counts of\n  missing values",
        "statistical tests\n  and effect sizes",
        'an average, a count, a maximum\n  or a missing-value count is "sql" even for SECOM',
    ):
        assert needle in PLANNER_SYSTEM, needle


# --- the writer prompt -------------------------------------------------------------------------


def test_the_citation_rules_are_the_same_for_every_route() -> None:
    answered = {"status": "answered"}
    base = writer_system(None)
    assert base == WRITER_SYSTEM and writer_system({"status": "failed"}) == WRITER_SYSTEM
    with_stats = writer_system(answered)
    assert with_stats.startswith(WRITER_SYSTEM.split("\nAnswer with a JSON object")[0])
    assert STATS_WRITER_RULES in with_stats and with_stats.endswith('{"answer": "..."} only.')
    for text in (base, with_stats):
        assert "Write one tag per bracket: [D1] [D2], never [D1, D2]" in text
        assert "Cite every sentence that states a number" in text
    assert "[A1]" not in base and "never the number of runs of the dataset" in with_stats


# --- the graph with a planner that routes wrongly ----------------------------------------------


def test_a_secom_aggregate_sent_to_stats_by_the_model_runs_as_sql(
    secom_rows, make_env, tag
) -> None:
    question = f"Which 5 SECOM sensor features have the fewest missing values? {tag}"
    env = make_env([plan("stats"), sql_json(), write("The answer is 42 [S1].")])
    out = env.ask(question)
    assert out.state["route"] == "sql" and out.status == "answered"
    assert out.state["plan"]["route_rule"] == "planner said stats, the rules say sql"
    assert "stats_agent" not in out.stats and env.sandbox.jobs == []
    assert out.state["sql"]["status"] == "answered" and not out.state.get("stats")


def test_a_statistical_question_sent_to_sql_by_the_model_runs_as_stats(
    secom_rows, make_env, tag
) -> None:
    question = f"Which sensor features are most strongly associated with failing runs? {tag}"
    env = make_env([plan("sql"), stats_json("group_compare"), write(GOOD_DRAFT)])
    out = env.ask(question)
    assert out.state["route"] == "stats" and out.status == "answered"
    assert out.state["plan"]["route_rule"] == "planner said sql, the rules say stats"
    assert "sql_agent" not in out.stats and len(env.sandbox.jobs) == 1


def test_a_correct_model_route_is_not_touched(secom_rows, make_env, tag) -> None:
    env = make_env([plan("sql"), sql_json(), write("The answer is 42 [S1].")])
    out = env.ask(f"What is the average of SECOM feature 12 among passing runs? {tag}")
    assert out.state["route"] == "sql" and out.state["plan"]["route_rule"] is None


def test_other_routes_and_non_secom_questions_are_never_rerouted(make_env, tag) -> None:
    env = make_env([plan("sql"), sql_json(), write("The answer is 42 [S1].")])
    out = env.ask(f"What is the distribution of licence status codes in the ULS data? {tag}")
    assert out.state["route"] == "sql" and out.state["plan"]["route_rule"] is None
    refusal = make_env([plan("refuse", reason="Prediction is outside the menu.")])
    out = refusal.ask(f"Is feature 3 significantly different? Predict the next lot too. {tag}")
    assert out.state["route"] == "refuse" and out.status == "refused"


def test_the_writer_gets_the_stats_rules_only_when_there_is_an_analysis(
    secom_rows, make_env, tag
) -> None:
    env = make_env([plan("stats"), stats_json(), write(GOOD_DRAFT)])
    env.ask(f"Do passing and failing runs differ in feature 60? {tag}")
    writer_prompt = env.provider.calls[2][0][0].content
    assert STATS_WRITER_RULES in writer_prompt
    plain = make_env([plan("sql"), sql_json(), write("The answer is 42 [S1].")])
    plain.ask(f"What is the average of SECOM feature 12 among failing runs? {tag}")
    assert plain.provider.calls[2][0][0].content == WRITER_SYSTEM
