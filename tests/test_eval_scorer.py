"""Execution-accuracy scorer and the eval-sql planning helpers (no database, no LLM)."""

from decimal import Decimal

import pytest

from rflens.agents.sql_agent import AgentResult
from rflens.config import LlmSettings
from rflens.eval.eval_sql import CALLS_PER_QUESTION, build_plan, eval_settings, main, score
from rflens.eval.golden import GoldenQuestion
from rflens.eval.scorer import ResultSet, compare, values_equal
from rflens.llm.base import Pricing, Rate


def rs(rows, columns=None) -> ResultSet:
    width = len(rows[0]) if rows else 1
    return ResultSet(columns or [f"c{i}" for i in range(width)], rows)


# --- ordering -----------------------------------------------------------------------------------


def test_unordered_ignores_row_order_and_ordered_does_not() -> None:
    a, b = rs([[1, "x"], [2, "y"], [3, "z"]]), rs([[3, "z"], [1, "x"], [2, "y"]])
    assert compare(a, b).match
    verdict = compare(a, b, ordered=True)
    assert not verdict.match and "reference column 1" in verdict.reason
    assert compare(a, a, ordered=True).match


def test_unordered_is_a_multiset_comparison() -> None:
    assert compare(rs([[1], [1]]), rs([[1], [2]])).match is False
    assert compare(rs([[1], [1], [2]]), rs([[2], [1], [1]])).match
    assert compare(rs([[1], [2]]), rs([[1], [1]])).match is False  # same size, other content


def test_the_row_count_must_match_exactly() -> None:
    assert "row(s)" in compare(rs([[1]]), rs([[1], [2]])).reason
    assert "row(s)" in compare(rs([[1], [2], [3]]), rs([[1], [2]])).reason
    assert not compare(rs([[1], [2], [2]]), rs([[1], [2]])).match


def test_column_names_are_ignored() -> None:
    assert compare(ResultSet(["a"], [[5]]), ResultSet(["total"], [[5]])).match


# --- relaxed column match: extra agent columns allowed, reference columns required ----------

REF = ResultSet(["country", "grantees"], [["US", 19212], ["China", 12962], ["Taiwan", 3717]])


def test_an_extra_agent_column_passes() -> None:
    extra_end = ResultSet(
        ["c", "n", "x"], [["US", 19212, 1], ["China", 12962, 2], ["Taiwan", 3717, 3]]
    )
    extra_mid = ResultSet(
        ["c", "x", "n"], [["US", "a", 19212], ["China", "b", 12962], ["Taiwan", "c", 3717]]
    )
    extra_front = ResultSet(
        ["rank", "c", "n"], [[1, "US", 19212], [2, "China", 12962], [3, "Taiwan", 3717]]
    )
    for agent in (extra_end, extra_mid, extra_front):
        assert compare(agent, REF).match and compare(agent, REF, ordered=True).match


def test_reference_columns_may_come_in_another_order() -> None:
    swapped = ResultSet(["n", "c"], [[19212, "US"], [12962, "China"], [3717, "Taiwan"]])
    assert compare(swapped, REF).match and compare(swapped, REF, ordered=True).match


def test_a_missing_reference_column_fails() -> None:
    only_names = ResultSet(["c"], [["US"], ["China"], ["Taiwan"]])
    verdict = compare(only_names, REF)
    assert not verdict.match and "reference needs 2" in verdict.reason
    wrong_second = ResultSet(["c", "x"], [["US", 1], ["China", 2], ["Taiwan", 3]])
    verdict = compare(wrong_second, REF)
    assert not verdict.match and "reference column 2" in verdict.reason
    assert not compare(rs([[1, 2]]), rs([[1, 2, 3]])).match


def test_a_wrong_value_in_a_reference_column_fails_even_with_extra_columns() -> None:
    wrong = ResultSet(["c", "n", "x"], [["US", 19212, 1], ["China", 12962, 2], ["Taiwan", 3718, 3]])
    assert not compare(wrong, REF).match


def test_right_values_in_the_wrong_row_order_fail_only_when_ordered() -> None:
    shuffled = ResultSet(
        ["c", "n", "x"], [["China", 12962, 2], ["US", 19212, 1], ["Taiwan", 3717, 3]]
    )
    assert compare(shuffled, REF, ordered=False).match
    verdict = compare(shuffled, REF, ordered=True)
    assert not verdict.match and "reference column" in verdict.reason
    # one column in the right order is not enough: every matched column must follow the order
    one_column_sorted = ResultSet(["n", "c"], [[19212, "China"], [12962, "US"], [3717, "Taiwan"]])
    assert not compare(one_column_sorted, REF, ordered=True).match


def test_columns_that_agree_separately_but_pair_up_differently_do_not_pass() -> None:
    mixed = ResultSet(["c", "n"], [["US", 12962], ["China", 19212], ["Taiwan", 3717]])
    assert not compare(mixed, REF).match  # same names, same numbers, wrong pairs
    assert not compare(mixed, REF, ordered=True).match


def test_duplicate_columns_are_matched_to_distinct_agent_columns() -> None:
    reference = ResultSet(["a", "b"], [[1, 1], [2, 2]])
    assert compare(ResultSet(["x", "y"], [[1, 1], [2, 2]]), reference).match
    assert not compare(ResultSet(["x"], [[1], [2]]), reference).match  # one column cannot be both
    assert compare(ResultSet(["x", "y", "z"], [[1, 5, 1], [2, 6, 2]]), reference).match


def test_extra_columns_with_tolerance_nulls_and_ties() -> None:
    reference = ResultSet(["v", "w"], [[1000.0, None], [2.5, 3]])
    agent = ResultSet(["extra", "w", "v"], [["p", None, 1000.0000005], ["q", 3.0, 2.5]])
    assert compare(agent, reference).match
    tied = ResultSet(["k", "n"], [["a", 5], ["b", 5], ["c", 4]])
    assert compare(
        ResultSet(["n", "k", "z"], [[5, "a", 0], [5, "b", 0], [4, "c", 0]]), tied, True
    ).match
    assert not compare(ResultSet(["n", "k"], [[5, "b"], [5, "a"], [4, "c"]]), tied, True).match


# --- numbers ------------------------------------------------------------------------------------


def test_floats_use_a_relative_tolerance_of_1e_6() -> None:
    assert values_equal(1000.0, 1000.0 + 0.0009)  # 9e-7 relative
    assert not values_equal(1000.0, 1000.0 + 0.002)  # 2e-6 relative
    assert values_equal(8.515123076923071, 8.5151230769)
    assert not values_equal(2.563464, 2.5635)
    assert values_equal(0.0, 0.0) and values_equal(0.0, 1e-15)
    assert not values_equal(0.0, 1e-3)


def test_int_float_and_decimal_are_the_same_number() -> None:
    assert values_equal(104, 104.0) and values_equal(Decimal("1054.900"), 1054.9)
    assert compare(rs([[104]]), rs([[104.0]])).match


def test_booleans_strings_and_numbers_do_not_mix() -> None:
    assert not values_equal(True, 1) and not values_equal(1, True)
    assert values_equal(True, True) and not values_equal(True, False)
    assert not values_equal(5, "5") and not values_equal("5", 5)
    assert values_equal("Japan", "Japan") and not values_equal("Japan", "japan")
    assert values_equal("2026-10-18", "2026-10-18")


# --- NULLs and empty results --------------------------------------------------------------------


def test_nulls_equal_only_nulls() -> None:
    assert values_equal(None, None)
    assert not values_equal(None, 0) and not values_equal(0, None)
    assert not values_equal(None, "") and not values_equal("", None)
    assert compare(rs([[None, 1], [2, None]]), rs([[2, None], [None, 1]])).match
    assert not compare(rs([[None]]), rs([[0]])).match


def test_empty_results() -> None:
    empty2 = ResultSet(["a", "b"], [])
    assert compare(empty2, ResultSet(["x", "y"], [])).match
    assert compare(empty2, ResultSet(["x"], [])).match  # an extra column is allowed
    assert not compare(ResultSet(["x"], []), empty2).match  # a missing reference column is not
    assert not compare(empty2, ResultSet(["x", "y"], [[1, 2]])).match
    assert not compare(ResultSet(["x", "y"], [[1, 2]]), empty2, ordered=True).match


# --- score(): sql, refuse, must_mention ---------------------------------------------------------


def question(**kw) -> GoldenQuestion:
    base = {"id": "q", "type": "sql", "question": "?", "status": "draft"}
    return GoldenQuestion(**{**base, **kw})


def result(**kw) -> AgentResult:
    return AgentResult(**{"question_id": "q", "question": "?", "status": "answered", **kw})


def test_score_sql_compares_the_result_set() -> None:
    q = question(reference_result=ResultSet(["n"], [[104]]))
    assert score(q, result(columns=["failed"], rows=[[104]]))[0]
    ok, reason = score(q, result(columns=["failed"], rows=[[105]]))
    assert not ok and "reference column 1" in reason
    ok, reason = score(q, result(status="failed", notes=["gave up"]))
    assert not ok and "gave up" in reason
    assert not score(q, result(status="refused", notes=["no"]))[0]


def test_score_ordered_question() -> None:
    ref = ResultSet(["a"], [[1], [2]])
    assert score(question(reference_result=ref), result(columns=["a"], rows=[[2], [1]]))[0]
    assert not score(
        question(reference_result=ref, ordered=True), result(columns=["a"], rows=[[2], [1]])
    )[0]


def test_score_must_mention_requires_the_unit_to_be_reported() -> None:
    q = question(reference_result=ResultSet(["m"], [[1054.9]]), must_mention=["unknown"])
    answer = {"columns": ["m"], "rows": [[1054.9]]}
    assert score(q, result(**answer, notes=["The ERP unit is UNKNOWN."]))[0]
    ok, reason = score(q, result(**answer, notes=["Value is in watts."]))
    assert not ok and "unknown" in reason
    assert not score(q, result(**answer))[0]


def test_score_refuse_needs_a_refusal() -> None:
    q = question(type="refuse", expected_refusal="modifies data")
    assert score(q, result(status="refused", notes=["read-only"]))[0]
    ok, reason = score(q, result(status="answered", columns=["n"], rows=[[1]]))
    assert not ok and "modifies data" in reason
    assert not score(q, result(status="failed"))[0]


# --- eval-sql planning --------------------------------------------------------------------------


def test_eval_settings_cap_the_run_at_questions_x_4_and_turn_fallback_off() -> None:
    base = LlmSettings(_env_file=None)
    tuned = eval_settings(base, 17)
    assert tuned.llm_max_calls_per_run == 68 and tuned.llm_fallback == ""
    assert tuned.llm_max_calls_per_question == base.llm_max_calls_per_question == 4
    assert base.llm_max_calls_per_run == 30  # the original settings are untouched
    assert CALLS_PER_QUESTION == 4


def test_worst_case_plan_arithmetic() -> None:
    pricing = Pricing(
        {"gemini": {"m": {"standard": Rate(1_000_000, 2_000_000)}}}
    )  # $1 / $2 per token
    settings = LlmSettings(_env_file=None, llm_model="m")
    qs = [question(id="a", question="x" * 30), question(id="b", question="y" * 3)]
    plan = build_plan(qs, "s" * 297, settings, 100, pricing)
    # prompt: ceil((297 + 30) / 3) + 4 * 2 messages = 117 tokens
    assert plan.prompt_tokens == 117 and (plan.questions, plan.max_calls) == (2, 8)
    # call k sends prompt + k * (100 + 150) tokens and may use 100 output tokens
    per_question = sum((117 + k * 250) * 1 + 100 * 2 for k in range(4))
    assert plan.worst_per_question_usd == per_question
    assert plan.worst_total_usd == 2 * per_question


def test_main_stops_cleanly_when_sql_mcp_is_not_running(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setenv("SQL_MCP_URL", "http://127.0.0.1:9/mcp")  # nothing listens on port 9
    assert main(["--dry-run"]) == 2
    assert "make sql-mcp" in capsys.readouterr().err
