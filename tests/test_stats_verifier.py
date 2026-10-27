"""The verifier's statistics rules: [A1] as a number source, and unsupported_stat_claim,
causal_claim and missing_n. Pure functions; no database."""

import copy

import pytest

from analytics_support import COMPARE_60, RANKING
from rflens.graph.verifier import NUM_RE, parse_number, stats_evidence, verify

QUESTION = "Do passing and failing runs differ in feature 60?"


def stats(result=None, **extra):
    base = {
        "status": "answered",
        "analysis": "group_compare",
        "params": {"feature": 60, "test": "mann_whitney", "min_n": 20},
        "call_id": 7,
        "result": copy.deepcopy(result or COMPARE_60),
        "n_runs_used": 1567,
        "n_runs_in_extract": 1567,
    }
    return base | extra


def check(draft, st=None, question=QUESTION):
    return verify(draft, question=question, route="stats", sql=None, chunks=[], stats=st or stats())


def kinds(result):
    return [f["type"] for f in result["failures"]]


def test_scientific_notation_is_one_number_with_its_own_precision() -> None:
    assert [m.group(0) for m in NUM_RE.finditer("p = 3.4e-10 and 1,234 and 2E5")] == [
        "3.4e-10",
        "1,234",
        "2E5",
    ]
    assert parse_number("3.4e-10") == (3.4e-10, 11)
    assert parse_number("1.5E+3") == (1500.0, 0)
    assert parse_number("12.35") == (12.35, 2)
    assert parse_number("1,234") == (1234.0, 0)


def test_a_correct_cited_answer_passes() -> None:
    draft = (
        "Feature 60 is higher in failing runs [A1]. The Mann-Whitney test used 1463 passing and "
        "97 failing values and gives p = 3.4e-10 with an effect size of 0.38 [A1]."
    )
    result = check(draft)
    assert result["ok"], result["failures"]
    sources = {n["number"]: n["source"] for e in result["evidence_map"] for n in e["numbers"]}
    assert sources["1463"] == "stats" and sources["97"] == "stats" and sources["0.38"] == "stats"


@pytest.mark.parametrize(
    ("number", "why"),
    [
        ("1500", "not a leaf"),
        ("0.45", "an effect size that is not in the result"),
        ("2.1e-10", "a p-value that is not in the result"),
        ("3.4e-12", "the right digits, the wrong exponent"),
    ],
)
def test_a_number_that_is_in_no_result_is_unsupported(number, why) -> None:
    result = check(f"The statistic is {number} for feature 60 [A1].")
    assert "unsupported_number" in kinds(result), why


def test_rounded_percent_and_derived_numbers_are_accepted() -> None:
    result = check(
        "Failing runs have a median of 1.5 against 0.9 for passing runs, a difference of 0.6 "
        "[A1]. In total 1560 values were used [A1]. The share of passing values among them is "
        "93.8% [A1]."
    )
    assert result["ok"], result["failures"]


def test_a_p_bound_is_accepted_only_if_the_result_satisfies_it() -> None:
    ok = check("The result has p < 0.001 for 1463 passing and 97 failing values [A1].")
    assert ok["ok"], ok["failures"]
    bad = check("The result has p < 0.0000001 for 1463 passing and 97 failing values [A1].")
    assert "unsupported_number" in kinds(bad)  # 3.4e-10 is below it, but 0.0000001 is a 1e-07 bound
    weak = stats(copy.deepcopy(COMPARE_60) | {"p_value": 0.0004})  # below 0.001, not below 0.0001
    assert check("The result has p < 0.001 with 1463 passing and 97 failing values [A1].", weak)[
        "ok"
    ]
    too_strong = check(
        "The result has p < 0.0001 with 1463 passing and 97 failing values [A1].", weak
    )
    assert "unsupported_number" in kinds(too_strong)


def test_a_sourced_claim_needs_the_a1_tag() -> None:
    result = check("The Mann-Whitney test used 1463 passing values.")
    assert set(kinds(result)) == {"missing_citation"}  # the answer and the sentence both lack [A1]


def test_citing_a1_without_an_analysis_is_dangling() -> None:
    result = verify("It is higher [A1].", question=QUESTION, route="stats", sql=None, chunks=[])
    assert "dangling_citation" in kinds(result)
    failed = verify(
        "It is higher [A1].",
        question=QUESTION,
        route="stats",
        sql=None,
        chunks=[],
        stats={"status": "refused"},
    )
    assert "dangling_citation" in kinds(failed)


def test_an_answer_with_an_analysis_must_cite_it() -> None:
    assert "missing_citation" in kinds(check("Feature 60 differs between the groups."))


def test_feature_values_have_no_unit() -> None:
    result = check("The median of feature 60 in failing runs is 1.5 dBm [A1].")
    assert [f["type"] for f in result["failures"]] == ["invented_unit"]
    assert "no known unit" in result["failures"][0]["detail"]
    in_question = check("Is the median 1.5 dBm? Median 1.5 dBm [A1].", question="Is it 1.5 dBm?")
    assert "invented_unit" not in kinds(in_question)


# --- the three rules of the statistics ---------------------------------------------------------


def test_significant_needs_a_p_value_below_alpha_in_the_result() -> None:
    draft = (
        "Feature 60 differs significantly between the groups, with p = 3.4e-10 for 1463 passing "
        "and 97 failing values [A1]."
    )
    assert check(draft)["ok"]
    weak = copy.deepcopy(COMPARE_60) | {"p_value": 0.3, "significant_at_alpha": False}
    result = check(draft.replace("3.4e-10", "0.3"), stats(weak))
    assert "unsupported_stat_claim" in kinds(result)
    assert "all are 0.05 or more" in next(f["detail"] for f in result["failures"])


def test_significant_without_any_test_in_the_result_is_unsupported() -> None:
    describe = stats({"feature": 60, "n": 1560, "n_missing": 7, "mean": 2.96}, analysis="describe")
    result = check("The mean of 2.96 is significant [A1].", describe)
    assert kinds(result) == ["unsupported_stat_claim"]
    assert "the analysis has none" in result["failures"][0]["detail"]


def test_significant_must_come_with_its_p_value() -> None:
    result = check("Feature 60 differs significantly between the groups [A1].")
    assert kinds(result) == ["unsupported_stat_claim"]
    assert "must come with" in result["failures"][0]["detail"]


def test_not_significant_needs_a_p_value_at_or_above_alpha() -> None:
    weak = copy.deepcopy(COMPARE_60) | {"p_value": 0.3, "significant_at_alpha": False}
    draft = "The difference is not significant (p = 0.3, 1463 and 97 values) [A1]."
    assert check(draft, stats(weak))["ok"]
    result = check(draft.replace("0.3", "3.4e-10"), stats())
    assert "unsupported_stat_claim" in kinds(result)


@pytest.mark.parametrize(
    "sentence",
    [
        "Feature 60 causes the failures [A1].",
        "Failures are due to feature 60 [A1].",
        "A high value leads to a failing run [A1].",
        "This results in more failures [A1].",
        "Feature 60 drives failures [A1].",
    ],
)
def test_cause_words_are_refused(sentence) -> None:
    result = check(sentence)
    assert "causal_claim" in kinds(result)
    assert next(f for f in result["failures"] if f["type"] == "causal_claim")["sentence"] == 0


def test_a_denial_of_causation_is_allowed() -> None:
    result = check("This is an association and does not show that feature 60 causes failures [A1].")
    assert "causal_claim" not in kinds(result)


def test_a_p_value_needs_the_n_behind_it() -> None:
    result = check("Feature 60 differs, with p = 3.4e-10 [A1].")
    assert kinds(result) == ["missing_n"]
    assert result["failures"][0]["sentence"] == 0  # so redaction can drop that sentence
    fixed = check("Feature 60 differs, with p = 3.4e-10 for 1463 passing values [A1].")
    assert fixed["ok"], fixed["failures"]
    bare = check("The test statistic is 3.4e-10 [A1].")  # a number equal to a p-value, no n
    assert "missing_n" in kinds(bare)


def test_a_ranking_is_checked_against_its_rows_and_group_sizes() -> None:
    st = stats(RANKING, analysis="feature_ranking", params={"top_k": 2, "min_n": 20})
    draft = (
        "The two strongest associations are feature 60 (p = 3.4e-10, 1463 passing and 97 failing "
        "values) and feature 104 (q = 0.012, 1460 passing and 104 failing values) [A1]."
    )
    result = check(draft, st, question="Which two features are most associated with failing runs?")
    assert result["ok"], result["failures"]
    wrong = check(draft.replace("104 (q", "205 (q"), st)
    assert "unsupported_number" in kinds(wrong)


def test_stats_evidence_exposes_its_leaves_and_bounds() -> None:
    ev = stats_evidence(stats())
    assert {1463.0, 97.0, 0.384, 3.4e-10, 60.0, 1567.0} <= ev["values"]
    assert ev["p_values"] == [3.4e-10]
    # the n behind the test: both group sizes and their total, not the runs of the extract (1567)
    assert ev["n_values"] == ev["sample_sizes"] == {1463.0, 97.0, 1560.0}
    assert 0.001 in ev["derived"] and 0.05 in ev["derived"]  # p = 3.4e-10 is below both bounds
    assert 0.0000001 not in ev["derived"]  # bounds go down to 1e-06 only
    assert 1463 + 97 in ev["derived"] and abs(1.5 - 0.9) in {round(v, 10) for v in ev["derived"]}


# --- wrong_n: an n stated for a test is one of the test's own sample sizes ------------------

RANKING_STATS = stats(RANKING, analysis="feature_ranking", params={"top_k": 2, "min_n": 20})
RANKING_Q = "Which two features are most associated with failing runs?"


def test_the_stats_01_answer_with_the_total_runs_as_n_is_wrong_n() -> None:
    """Live: "a Mann-Whitney U test (n=1567, ...)" -- 1567 is the runs of the extract, while the
    tests rested on 1,463 / 97 and 1,460 / 104 values."""
    draft = (
        "Features 60 and 104 are most associated with failing runs [A1]. "
        "This used a Mann-Whitney U test (n=1567, with missing values dropped pairwise) [A1]."
    )
    result = check(draft, RANKING_STATS, RANKING_Q)
    wrong = [f for f in result["failures"] if f["type"] == "wrong_n"]
    assert [f["sentence"] for f in wrong] == [1] and "n = 1567" in wrong[0]["detail"]
    assert kinds(result) == ["wrong_n"]  # 1567 itself is a number of the result: no other failure


@pytest.mark.parametrize(
    "sentence",
    [
        "The Mann-Whitney test used 1463 passing and 97 failing values [A1].",
        "The Mann-Whitney test has n=1463 for pass and n=97 for fail [A1].",
        "The Mann-Whitney test (n = 1,560 in total) found a difference [A1].",  # 1463 + 97
        "The Mann-Whitney test of 1460 pass and 104 fail values found a difference [A1].",
        "The Spearman correlation rests on n of 97 pairs [A1].",
    ],
)
def test_the_groups_sizes_and_their_total_are_accepted(sentence) -> None:
    result = check(sentence, RANKING_STATS, RANKING_Q)
    assert "wrong_n" not in kinds(result), result["failures"]


@pytest.mark.parametrize(
    ("sentence", "stated"),
    [
        ("The Mann-Whitney test used 1500 passing and 97 failing values [A1].", "1500"),
        ("The Mann-Whitney test has n = 1,567 [A1].", "1,567"),
        ("The effect size comes from 1567 runs [A1].", "1567"),
        ("The p-value rests on n: 99 values [A1].", "99"),
        (
            "With 104 failing and 1567 passing runs the test is significant, p = 3.4e-10 [A1].",
            "1567",
        ),
    ],
)
def test_another_number_as_n_is_wrong_n(sentence, stated) -> None:
    result = check(sentence, RANKING_STATS, RANKING_Q)
    wrong = [f for f in result["failures"] if f["type"] == "wrong_n"]
    assert wrong and all(f["sentence"] == 0 for f in wrong) and stated in wrong[0]["detail"]


def test_a_number_of_runs_outside_a_sentence_about_a_test_is_not_checked() -> None:
    result = check("The dataset has 1567 runs [A1].", RANKING_STATS, RANKING_Q)
    assert result["ok"], result["failures"]


def test_the_runs_of_the_extract_no_longer_count_as_the_n_behind_a_p_value() -> None:
    result = check("Feature 60 differs, with p = 3.4e-10 over all runs (1567) [A1].", RANKING_STATS)
    assert "missing_n" in kinds(result)  # 1567 is not a sample size of the test


def test_wrong_n_is_a_prose_failure_so_it_goes_to_the_writer() -> None:
    from rflens.graph.nodes import SQL_FAILURES

    assert "wrong_n" not in SQL_FAILURES
