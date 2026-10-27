"""The four stats golden questions: structure, independent references, cutoff rule, scoring."""

import copy
import datetime
from pathlib import Path

import pytest
import yaml

from rflens.eval.e2e_score import _at, _same, score_question
from rflens.eval.golden import (
    CUTOFF_RATIO,
    GoldenQuestion,
    load_golden,
    load_raw,
    problems,
    refresh_stats,
    stats_reference,
)
from rflens.eval.results_log import E2E_COLUMNS, E2E_TITLE, parse_table, refresh_counts

GOLDEN = {q.id: q for q in load_golden() if q.type == "stats"}
REFERENCE = {
    "scipy": "1.18.1",
    "extract_sha256": "0" * 64,
    "n_runs": 1567,
    "ranking": [
        {
            "feature": 60,
            "n_pass": 1456,
            "n_fail": 104,
            "p_value": 5.7e-11,
            "q_value": 2.6e-8,
            "rank_biserial": 0.3,
        },
        {
            "feature": 104,
            "n_pass": 1461,
            "n_fail": 104,
            "p_value": 9.6e-9,
            "q_value": 2.2e-6,
            "rank_biserial": 0.3,
        },
        {
            "feature": 248,
            "n_pass": 781,
            "n_fail": 71,
            "p_value": 1.07e-6,
            "q_value": 1.4e-4,
            "rank_biserial": 0.3,
        },
        {
            "feature": 520,
            "n_pass": 781,
            "n_fail": 71,
            "p_value": 1.19e-6,
            "q_value": 1.4e-4,
            "rank_biserial": 0.3,
        },
        {
            "feature": 511,
            "n_pass": 1461,
            "n_fail": 104,
            "p_value": 3.5e-6,
            "q_value": 3.2e-4,
            "rank_biserial": 0.3,
        },
        {
            "feature": 478,
            "n_pass": 1457,
            "n_fail": 104,
            "p_value": 3.5e-5,
            "q_value": 2.7e-3,
            "rank_biserial": 0.3,
        },
    ],
    "compare60": {
        "feature": 60,
        "n_pass": 1456,
        "n_fail": 104,
        "median_pass": 0.72,
        "median_fail": 5.52,
        "mean_pass": 2.56,
        "mean_fail": 8.5,
        "p_value": 5.7e-11,
        "rank_biserial": 0.384,
    },
    "dist104": {
        "feature": 104,
        "n": 1565,
        "n_missing": 2,
        "min": -0.03,
        "q1": -0.0118,
        "median": -0.0101,
        "q3": -0.0082,
        "max": 0.02,
        "counts": [1] * 20,
    },
}


def test_there_are_four_stats_questions_three_with_references_and_one_refusal() -> None:
    assert list(GOLDEN) == ["stats-01", "stats-02", "stats-03", "stats-04"]
    # stats-02 asks for what its reference holds: both medians and the effect size
    assert GOLDEN["stats-02"].question.endswith("(the median of each group and the effect size)?")
    for q in GOLDEN.values():
        assert q.author == "assistant" and q.status == "reviewed" and q.notes.strip()
    refusal = GOLDEN["stats-04"]
    assert refusal.expected_refusal and refusal.reference_analysis is None
    assert "predict" in refusal.question and "classifier" in refusal.question
    kinds = {
        q.id: q.reference_analysis["analysis"] for q in GOLDEN.values() if q.reference_analysis
    }
    assert kinds == {
        "stats-01": "feature_ranking",
        "stats-02": "group_compare",
        "stats-03": "distribution",
    }


def test_every_reference_says_how_it_was_computed_and_by_what() -> None:
    for qid in ("stats-01", "stats-02", "stats-03"):
        q = GOLDEN[qid]
        via = q.reference_stats["via"]
        assert "analytics_runner/reference.py" in via and "shares no code" in via
        assert "independently of the analytics code" in q.notes
        assert (
            q.reference_stats["scipy"] == "1.18.1"
            and len(q.reference_stats["extract_sha256"]) == 64
        )
        assert q.reference_stats["checks"] and q.reference_stats["answer_rows"]


def test_the_ranked_question_has_a_clear_cutoff_and_a_clear_order() -> None:
    q = GOLDEN["stats-01"]
    n = q.reference_analysis["params"]["top_k"]
    cutoff = q.reference_stats["cutoff"]
    assert cutoff["top_k"] == n and cutoff["ratio"] >= CUTOFF_RATIO
    assert cutoff[f"p_rank_{n + 1}"] / cutoff[f"p_rank_{n}"] == pytest.approx(cutoff["ratio"])
    p = {
        c["path"]: c["value"] for c in q.reference_stats["checks"] if c["path"].endswith("p_value")
    }
    ps = [p[f"top.{i}.p_value"] for i in range(n)]
    assert ps == sorted(ps) and all(b / a >= 2 for a, b in zip(ps, ps[1:], strict=False))
    assert q.reference_stats["ordered"] is True
    assert (
        "near-tied" in q.notes and "248 and 520" in q.notes
    )  # the reason the cutoff is where it is


def test_the_builder_refuses_a_near_tie_at_the_cutoff_or_inside_the_top() -> None:
    with pytest.raises(SystemExit, match="near-tie at the cutoff"):
        stats_reference("stats-01", {"top_k": 3}, REFERENCE)  # 248 (1.07e-6) vs 520 (1.19e-6)
    with pytest.raises(SystemExit, match="ranks 3 and 4 are near-tied"):
        stats_reference("stats-01", {"top_k": 4}, REFERENCE)  # 248 and 520 both inside the top 4
    cutoff_only = copy.deepcopy(REFERENCE)
    cutoff_only["ranking"][1]["p_value"] = 2.0e-10  # rank 2 is only 3.5 times rank 1
    with pytest.raises(SystemExit, match=r"near-tie at the cutoff.*ratio 3\.5"):
        stats_reference("stats-01", {"top_k": 1}, cutoff_only)
    built = stats_reference("stats-01", {"top_k": 2}, REFERENCE)
    assert built["answer_rows"] == [[60], [104]] and built["cutoff"]["ratio"] > 100
    close_pair = copy.deepcopy(REFERENCE)
    close_pair["ranking"][1]["p_value"] = (
        6.0e-11  # ranks 1 and 2 almost equal: order is a coin flip
    )
    with pytest.raises(SystemExit, match="near-tied"):
        stats_reference("stats-01", {"top_k": 2}, close_pair)


def test_the_other_references_use_the_independent_values() -> None:
    cmp_ = {c["path"]: c["value"] for c in stats_reference("stats-02", {}, REFERENCE)["checks"]}
    assert cmp_["groups.pass.n"] == 1456 and cmp_["groups.fail.n"] == 104
    assert cmp_["higher_in"] == "fail" and cmp_["rank_biserial"] == 0.384
    # the group sizes are checked in the result; the text must give the medians that are asked for
    built02 = stats_reference("stats-02", {}, REFERENCE)
    assert built02["answer_rows"] == [[0.72, 5.52]]
    stats02 = GOLDEN["stats-02"]
    assert stats02.reference_stats["answer_rows"] == [[0.7222500000000001, 5.5222999999999995]]
    assert "not for the sample sizes" in stats02.notes
    assert {"path": "groups.pass.n", "value": 1456} in stats02.reference_stats["checks"]
    dist = stats_reference("stats-03", {}, REFERENCE)
    # n (1565) is a check on the result, not a required value of the text: the question does not ask
    assert dist["answer_rows"] == [[2, -0.0118, -0.0101, -0.0082]]
    assert {"path": "n", "value": 1565} in dist["checks"]
    stats03 = GOLDEN["stats-03"]
    assert stats03.reference_stats["answer_rows"] == [[2, -0.0118, -0.0101, -0.0082]]
    assert "does not ask for the sample size" in stats03.notes
    assert {"path": "counts", "value": [1] * 20} in dist["checks"]


def test_refresh_writes_references_into_a_copy_and_leaves_the_rest(tmp_path: Path) -> None:
    raw = load_raw()
    keep = [
        q for q in raw["questions"] if q["id"] in ("stats-01", "stats-02", "stats-03", "sql-01")
    ]
    for q in keep:
        q.pop("reference_stats", None)
    path = tmp_path / "golden.yaml"
    path.write_text(yaml.safe_dump({"questions": keep}), encoding="utf-8")
    lines = refresh_stats(path, REFERENCE, datetime.date(2026, 10, 25))
    assert [ln.split(":")[0] for ln in lines] == ["stats-01", "stats-02", "stats-03"]
    written = {q["id"]: q for q in yaml.safe_load(path.read_text(encoding="utf-8"))["questions"]}
    assert written["stats-02"]["reference_stats"]["checked_on"] == "2026-10-25"
    assert "reference_stats" not in written["sql-01"]


def test_problems_flags_a_stats_question_without_references() -> None:
    bad = GoldenQuestion(
        id="stats-9",
        type="stats",
        question="q?",
        status="draft",
        author="assistant",
        notes="",
        reference_analysis={"analysis": "describe", "params": {"feature": 9999}},
    )
    found = problems([bad])
    assert any("no notes" in f for f in found)
    assert any("does not validate" in f for f in found)
    assert any("no reference_stats.checks" in f for f in found)
    assert any("answer_rows" in f for f in found)
    refused_with_ref = GoldenQuestion(
        id="stats-8",
        type="stats",
        question="q?",
        status="draft",
        author="assistant",
        notes="n",
        expected_refusal="x",
        reference_analysis={"analysis": "describe", "params": {"feature": 1}},
    )
    assert any("no reference_analysis/stats" in f for f in problems([refused_with_ref]))


# --- scoring ------------------------------------------------------------------------------------


def state_for(qid: str, **over) -> dict:
    q = GOLDEN[qid]
    ref = q.reference_stats
    result: dict = {}
    for check in ref["checks"]:
        node = result
        parts = check["path"].split(".")
        for i, part in enumerate(parts):
            last = i == len(parts) - 1
            nxt = parts[i + 1] if not last else None
            if isinstance(node, list):
                idx = int(part)
                while len(node) <= idx:
                    node.append({})
                if last:
                    node[idx] = check["value"]
                else:
                    node = node[idx]
            else:
                if last:
                    node[part] = check["value"]
                else:
                    node = node.setdefault(part, [] if nxt.isdigit() else {})
    values = " ".join(str(v) for row in ref["answer_rows"] for v in row)
    base = {
        "route": "stats",
        "status": "answered",
        "answer": f"The analysis found {values} [A1].",
        "verification": {"ok": True},
        "stats": {
            "status": "answered",
            "analysis": q.reference_analysis["analysis"],
            "result": result,
        },
        "node_stats": {},
        "draft_history": [],
    }
    return base | over


@pytest.mark.parametrize("qid", ["stats-01", "stats-02", "stats-03"])
def test_a_matching_answer_scores_correct(qid) -> None:
    score = score_question(GOLDEN[qid], state_for(qid))
    assert score.correct, score.reason


def test_a_wrong_analysis_number_citation_or_answer_value_is_a_miss() -> None:
    q = GOLDEN["stats-02"]
    good = state_for("stats-02")
    wrong_analysis = copy.deepcopy(good)
    wrong_analysis["stats"]["analysis"] = "describe"
    assert "the reference is group_compare" in score_question(q, wrong_analysis).reason
    wrong_number = copy.deepcopy(good)
    wrong_number["stats"]["result"]["groups"]["pass"]["n"] = 1455
    assert "groups.pass.n is 1455" in score_question(q, wrong_number).reason
    uncited = copy.deepcopy(good) | {"answer": good["answer"].replace("[A1]", "")}
    assert "does not cite the analysis [A1]" in score_question(q, uncited).reason
    missing_value = copy.deepcopy(good) | {"answer": "The groups differ [A1]."}
    assert "lacks reference value(s)" in score_question(q, missing_value).reason
    assert score_question(q, wrong_number).cause == "wrong analysis"
    refused = copy.deepcopy(good) | {"status": "refused", "stats": {"status": "refused"}}
    assert "no answered analysis" in score_question(q, refused).reason


def test_a_ranked_answer_must_name_the_features_in_order() -> None:
    q = GOLDEN["stats-01"]
    swapped = state_for("stats-01") | {"answer": "Features 104 and then 60 [A1]."}
    assert not score_question(q, swapped).correct
    ordered = state_for("stats-01") | {"answer": "Feature 60 first, then feature 104 [A1]."}
    assert score_question(q, ordered).correct


def test_the_refusal_question_needs_a_refusal_and_no_analysis() -> None:
    q = GOLDEN["stats-04"]
    refused = {"status": "refused", "answer": "Prediction is outside the menu.", "route": "refuse"}
    assert score_question(q, refused).correct
    ran_anyway = refused | {"stats": {"status": "answered", "analysis": "feature_ranking"}}
    assert "an analysis ran" in score_question(q, ran_anyway).reason
    answered = {"status": "answered", "answer": "Here is a ranking.", "route": "stats"}
    assert "expected a refusal" in score_question(q, answered).reason
    stats_agent_refused = refused | {
        "route": "stats",
        "stats": {"status": "refused", "notes": ["x"]},
    }
    assert score_question(q, stats_agent_refused).correct


def test_paths_and_comparison_rules() -> None:
    assert _at({"top": [{"feature": 60}]}, "top.0.feature") == 60
    assert _same(5.6786377604731255e-11, 5.678637760473e-11)  # 1e-6 relative
    assert not _same(5.7e-11, 5.6e-11) and not _same(True, 1) and _same("fail", "fail")
    assert _same([1, 2], [1, 2]) and not _same([1, 2], [1, 3]) and not _same([1], [1, 1])
    assert _same(0.0, 0.0) and not _same(None, 1.0)


def test_results_gain_a_stats_column_and_old_rows_keep_their_cells(tmp_path: Path) -> None:
    path = tmp_path / "RESULTS.md"
    refresh_counts(path, load_golden())
    old = path.read_text(encoding="utf-8")
    header = "| " + " | ".join(E2E_COLUMNS) + " |"
    assert header in old and "| mixed | stats | refuse |" in old
    # a file written before the column existed: header, separator and one row without it
    cols = [c for c in E2E_COLUMNS if c != "stats"]
    legacy = old.replace(header, "| " + " | ".join(cols) + " |")
    legacy = legacy.replace("|---|" * 0 + "|" + "---|" * len(E2E_COLUMNS), "|" + "---|" * len(cols))
    row = [
        "d",
        "r",
        "c",
        "m",
        "1/1",
        "1/1",
        "1/1 / 1/1",
        "0/0",
        "0/0",
        "1/1",
        "0/0",
        "3",
        "$0",
        "n",
    ]
    legacy = legacy.rstrip("\n") + "\n| " + " | ".join(row) + " |\n"
    path.write_text(legacy, encoding="utf-8")
    refresh_counts(path, load_golden())
    (parsed,) = parse_table(path, E2E_TITLE)
    assert len(parsed) == len(E2E_COLUMNS)
    assert parsed[:8] == row[:8] and parsed[8] == "-" and parsed[9:] == row[8:]
    refresh_counts(path, load_golden())  # idempotent
    assert parse_table(path, E2E_TITLE) == [parsed]


def test_only_picks_questions_by_id_and_rejects_unknown_ids(capsys) -> None:
    from rflens.eval.eval_e2e import main

    assert main(["--only", "stats-01,nope-9", "--dry-run"]) == 2
    assert "unknown question id(s): ['nope-9']" in capsys.readouterr().err
