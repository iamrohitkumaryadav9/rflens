"""make eval-e2e: the scorer (pure, on synthetic final states), the report, the RESULTS.md row and
the runner on the real graph with the fake provider. No paid call is possible here.
"""

# ruff: noqa: F811
# (the fixtures imported from test_graph are used by name)

import pytest

from rflens.eval.e2e_score import (
    chunk_hits,
    cited_tags,
    classify_miss,
    contains_values,
    score_question,
)
from rflens.eval.eval_e2e import (
    CALLS_PER_QUESTION,
    eval_settings,
    evaluate,
    render_report,
    result_cells,
    summarize,
    worst_case_usd,
)
from rflens.eval.golden import GoldenQuestion
from rflens.eval.results_log import E2E_COLUMNS, E2E_TITLE, append_e2e_row, parse_table
from rflens.eval.scorer import ResultSet
from rflens.llm.base import Pricing, Rate
from test_graph import (  # noqa: F401
    docs_conn,
    make_env,
    plan,
    saver,
    sql_answer,
    sql_server,
    tag,
    write,
)


def gq(qid="sql-x", kind="sql", rows=None, columns=None, **kw) -> GoldenQuestion:
    defaults = {
        "id": qid,
        "type": kind,
        "question": f"question {qid}?",
        "status": "reviewed",
        "author": "human",
        "spec": None,
        "reference_result": ResultSet(columns or ["n"], rows if rows is not None else [[42]]),
    }
    if kind == "refuse":
        defaults.update(reference_result=None, expected_refusal="out of scope")
    if kind in ("doc", "mixed"):
        defaults.update(
            spec="36.101", expected_clauses=["5.5"], confirmed_chunks=["36.101@1/5.5/1"]
        )
    if kind == "doc":
        defaults["reference_result"] = None
    defaults.update(kw)
    return GoldenQuestion(**defaults)


def state(
    answer="The answer is 42 [S1].",
    status="answered",
    ok=True,
    route="sql",
    columns=("n",),
    rows=((42,),),
    chunks=(),
    **kw,
) -> dict:
    s = {
        "status": status,
        "answer": answer,
        "route": route,
        "verification": {"ok": ok, "failures": []},
        "sql": {
            "status": "answered",
            "columns": list(columns),
            "rows": [list(r) for r in rows],
            "query_id": 11,
        },
        "chunks": [dict(c) for c in chunks],
        "doc_hits": [],
        "draft_history": [{"attempt": 1, "draft": answer, "ok": ok, "failures": []}],
        "node_stats": {
            "planner": {"calls": 1, "cost_usd": 0.001},
            "writer": {"calls": 1, "cost_usd": 0.002},
        },
        "sources": "[S1] sql-mcp query #11",
    }
    s.update(kw)
    return s


def chunk(ref="D1", clause="5.5", spec="36.101", table=None) -> dict:
    return {
        "ref": ref,
        "chunk_id": f"{spec}@1/{clause}/1",
        "spec": spec,
        "clause": clause,
        "table_number": table,
    }


# --- values in the answer text ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "rows", "ordered", "ok"),
    [
        ("There are 1,234 of them.", [[1234]], False, True),  # thousands separator
        ("It is 2.56 on average.", [[2.563464217]], False, True),  # rounded at its own precision
        ("It is 2.9 on average.", [[2.563464217]], False, False),
        ("Cellco Partnership leads with 422.", [["cellco partnership", 422]], False, True),
        ("No grantees registered then.", [[0]], False, True),  # zero may be written as a word
        ("Eight grantees.", [[0]], False, False),
        ("The unit is unknown.", [[None]], False, True),  # NULL needs no text
        ("A has 5 and B has 3.", [["A", 5], ["B", 3]], True, True),
        ("B has 3 and A has 5.", [["A", 5], ["B", 3]], True, False),  # not in reference order
        ("B has 3 and A has 5.", [["A", 5], ["B", 3]], False, True),  # order only if ranked
        ("A has 5.", [["A", 5], ["B", 3]], True, False),  # a value is missing
    ],
)
def test_contains_values(text, rows, ordered, ok) -> None:
    assert contains_values(text, ResultSet(["a"] * len(rows[0]), rows), ordered)[0] is ok


def test_the_reason_names_missing_and_misordered_values() -> None:
    ref = ResultSet(["a", "b"], [["A", 5], ["B", 3]])
    assert "lacks reference value(s) ['3']" in contains_values("A 5, B", ref, True)[1]
    assert "not in reference order" in contains_values("B 3, A 5", ref, True)[1]


# --- scoring one question ---------------------------------------------------------------------


def test_a_sql_answer_is_correct_when_cited_matching_and_containing_the_values() -> None:
    s = score_question(gq(), state())
    assert s.correct and s.sql_ok and s.verified and s.cited == ["S1"] and s.cause == ""
    assert (s.calls, s.cost_usd) == (2, 0.003)


@pytest.mark.parametrize(
    ("kwargs", "why"),
    [
        ({"answer": "The answer is 42."}, "does not cite the SQL result"),
        ({"rows": ((43,),), "answer": "The answer is 43 [S1]."}, "differs from the reference"),
        ({"answer": "The count is large [S1]."}, "lacks reference value"),
        ({"sql": None}, "no SQL result"),
    ],
)
def test_sql_misses(kwargs, why) -> None:
    s = score_question(gq(), state(**kwargs))
    assert not s.correct and why in s.reason


def test_must_mention_is_checked_in_the_answer_text() -> None:
    q = gq(must_mention=["unknown"])
    assert not score_question(q, state(answer="The ERP is 42 [S1].")).correct
    assert score_question(q, state(answer="The ERP is 42 [S1]; its unit is UNKNOWN.")).correct


def test_a_ranked_result_must_appear_in_reference_order() -> None:
    q = gq(rows=[["A", 5], ["B", 3]], columns=["n", "c"], ordered=True)
    good = state(
        answer="A [S1] has 5, then B has 3.", columns=("n", "c"), rows=(("A", 5), ("B", 3))
    )
    swapped = {**good, "answer": "B has 3, then A has 5 [S1]."}
    assert score_question(q, good).correct
    assert "reference order" in score_question(q, swapped).reason


def test_doc_correct_needs_a_cited_hit_and_a_verified_answered_state() -> None:
    q = gq("doc-x", "doc")
    ok = state(answer="Band 5 [D1].", route="doc", chunks=[chunk()], sql=None)
    assert score_question(q, ok).correct and score_question(q, ok).doc_hit
    assert not score_question(q, {**ok, "status": "partial"}).correct  # redacted
    assert not score_question(q, {**ok, "verification": {"ok": False, "failures": []}}).correct
    uncited = {**ok, "answer": "Band 5."}  # the right chunk was retrieved but not cited
    assert "no cited chunk" in score_question(q, uncited).reason
    wrong = {**ok, "chunks": [chunk(clause="6.2")]}
    assert not score_question(q, wrong).correct


def test_chunk_hit_rule_is_exact_clause_or_listed_table_of_the_right_spec() -> None:
    q = gq("doc-x", "doc", expected_clauses=["6.2.1"], expected_tables=["7.3.1-1"])
    assert chunk_hits(q, chunk(clause="6.2.1")) and chunk_hits(
        q, chunk(clause="x", table="7.3.1-1")
    )
    assert not chunk_hits(q, chunk(clause="6.2.1.1"))  # a sub-clause does not count
    assert not chunk_hits(q, chunk(clause="6.2.1", spec="38.101-1"))  # wrong spec
    assert not chunk_hits(q, chunk(clause="x", table="7.3.1-2"))


def test_evidence_recall_is_reported_apart_from_correctness() -> None:
    q = gq("doc-x", "doc")
    hit = {"chunk_id": "x", "spec": "36.101", "clause": "5.5", "table_number": None}
    cases = [
        # (search hits, chunks the writer saw, expected recall_search, recall_evidence)
        ([hit], [chunk()], True, True),
        (
            [hit],
            [chunk(clause="6.2")],
            True,
            False,
        ),  # retrieved at rank 4-5, not given to the writer
        ([], [chunk(clause="6.2")], False, False),
    ]
    for hits, chunks, search, evidence in cases:
        s = score_question(
            q, state(answer="x [D1].", chunks=chunks, doc_hits=hits, route="doc", sql=None)
        )
        assert (s.recall_search, s.recall_evidence) == (search, evidence)


def test_mixed_needs_both_parts_and_must_mention() -> None:
    q = gq("mixed-x", "mixed", must_mention=["unknown"])
    both = state(
        answer="Band 5 [D1]. The count is 42 [S1], unit unknown.", route="mixed", chunks=[chunk()]
    )
    s = score_question(q, both)
    assert s.correct and s.doc_hit and s.sql_ok
    no_doc = {**both, "chunks": [chunk(clause="6.2")]}
    s = score_question(q, no_doc)
    assert not s.correct and s.sql_ok and s.doc_hit is False
    no_mention = {**both, "answer": "Band 5 [D1]. The count is 42 [S1]."}
    s = score_question(q, no_mention)
    assert not s.correct and s.doc_hit and s.sql_ok is False and "does not mention" in s.reason


def test_refuse_needs_a_refusal_with_a_reason_and_no_sql_or_search() -> None:
    q = gq("refuse-x", "refuse")
    refused = {
        "status": "refused",
        "answer": "This asks to change data.",
        "route": "refuse",
        "draft_history": [],
        "node_stats": {"planner": {"calls": 1, "cost_usd": 0.0003}},
    }
    assert score_question(q, refused).correct
    assert "status is answered" in score_question(q, {**refused, "status": "answered"}).reason
    assert "without a reason" in score_question(q, {**refused, "answer": " "}).reason
    sql_ran = {**refused, "sql": {"query_id": 5}}
    assert "a SQL query ran" in score_question(q, sql_ran).reason
    searched = {**refused, "chunks": [chunk()]}
    assert "a document search ran" in score_question(q, searched).reason
    refused_by_sql_agent = {
        **refused,
        "route": "sql",
        "sql": {"status": "refused", "query_id": None},
    }
    assert score_question(q, refused_by_sql_agent).correct  # the SQL agent's refusal runs no query


def test_retries_and_failure_types_come_from_the_draft_history() -> None:
    drafts = [
        {"attempt": 1, "draft": "a", "ok": False, "failures": [{"type": "unsupported_number"}]},
        {"attempt": 2, "draft": "b", "ok": False, "failures": [{"type": "edge_side_mismatch"}]},
        {"attempt": 3, "draft": "c", "ok": True, "failures": []},
    ]
    s = score_question(gq(), state(draft_history=drafts, sql_retries=1))
    assert (s.writer_retries, s.sql_retries) == (
        1,
        1,
    )  # 3 drafts = 1 + 1 SQL retry + 1 writer retry
    assert s.failure_types == ["edge_side_mismatch", "unsupported_number"]


def test_first_guess_at_the_cause_of_a_miss() -> None:
    sql_q, doc_q, refuse_q = gq(), gq("doc-x", "doc"), gq("refuse-x", "refuse")
    wrong = state(rows=((43,),), answer="43 [S1]")
    assert score_question(sql_q, wrong).cause == "wrong SQL"
    assert score_question(sql_q, {**wrong, "route": "doc"}).cause == "wrong route"
    redacted = state(status="partial", answer="x", ok=False)
    assert score_question(sql_q, redacted).cause == "writer/verifier redaction"
    format_only = state(answer="The count is forty-two [S1].")  # right result, value not written
    assert score_question(sql_q, format_only).cause == "scorer issue"
    no_evidence = state(answer="x [D1].", route="doc", chunks=[chunk(clause="6.2")], sql=None)
    assert score_question(doc_q, no_evidence).cause == "retrieval miss"
    redacted_doc = state(
        answer="x", route="doc", status="partial", ok=False, sql=None, chunks=[chunk()]
    )
    assert score_question(doc_q, redacted_doc).cause == "writer/verifier redaction"
    answered = {"status": "answered", "answer": "42", "draft_history": [], "node_stats": {}}
    assert score_question(refuse_q, answered).cause == "refusal error"
    assert classify_miss(sql_q, {**wrong, "halt": "cap"}, False, None, None) == "other"


def test_cited_tags() -> None:
    assert cited_tags("a [S1] b [D2] [D1] [S1]") == ["D1", "D2", "S1"]


# --- report and RESULTS.md --------------------------------------------------------------------


def sample_scores():
    return [
        score_question(gq("sql-1"), state()),
        score_question(gq("sql-2", author="assistant"), state(rows=((9,),), answer="9 [S1]")),
        score_question(
            gq("doc-1", "doc"),
            state(
                answer="x [D1].",
                route="doc",
                chunks=[chunk()],
                sql=None,
                doc_hits=[{"spec": "36.101", "clause": "5.5", "chunk_id": "c"}],
            ),
        ),
        score_question(
            gq("doc-2", "doc", author="assistant"),
            state(
                answer="x [D1].",
                route="doc",
                chunks=[chunk(clause="6.2")],
                sql=None,
                doc_hits=[{"spec": "36.101", "clause": "5.5", "chunk_id": "c"}],
            ),
        ),
    ]


def test_summary_counts_per_type_author_and_doc_recall() -> None:
    sm = summarize(sample_scores())
    assert sm["type"] == {"sql": "1/2", "doc": "1/2", "mixed": "0/0", "refuse": "0/0"}
    assert sm["author"] == {"human": "2/2", "assistant": "0/2"}
    assert (sm["doc_answer"], sm["doc_recall_search"], sm["doc_recall_evidence"]) == (
        "1/2",
        "2/2",
        "1/2",
    )  # doc-2: found by the search, not given to the writer
    assert sm["status"] == {"answered": 4} and sm["causes"] == {"wrong SQL": 1, "retrieval miss": 1}
    assert result_cells(sample_scores())["doc"] == "1/2 / 2/2"


def test_the_report_shows_every_miss_with_answer_sources_reference_and_cause() -> None:
    scores = sample_scores()
    text = render_report(scores, {q.qid: gq(q.qid, q.type) for q in scores})
    assert "MISS sql-2 [sql, assistant]" in text and "MISS doc-2 [doc, assistant]" in text
    assert "MISS sql-1" not in text
    assert (
        "cause (first guess): wrong SQL" in text and "cause (first guess): retrieval miss" in text
    )
    assert "answer: 9 [S1]" in text and "[S1] sql-mcp query #11" in text
    assert "reference: result ['n'] [[42]]" in text


def test_the_e2e_row_is_appended_to_a_results_file_that_has_no_such_table_yet(tmp_path) -> None:
    path = tmp_path / "RESULTS.md"
    from rflens.eval.results_log import refresh_counts

    refresh_counts(path, [gq()])
    text = path.read_text(encoding="utf-8")
    path.write_text(text[: text.index(E2E_TITLE)].rstrip() + "\n", encoding="utf-8")  # an old file
    row = append_e2e_row(
        run_id="eval-e2e-1",
        commit="abc",
        model="m",
        tier="standard",
        questions_run=4,
        questions_total=51,
        cells=result_cells(sample_scores()),
        calls=8,
        cost_usd=0.0123,
        note="a | b",
        path=path,
        questions=[gq()],
    )
    (parsed,) = parse_table(path, E2E_TITLE)
    assert len(parsed) == len(E2E_COLUMNS) and parsed[1] == "eval-e2e-1"
    assert parsed[4] == "4/51" and parsed[5] == "1/2" and parsed[6] == "1/2 / 2/2"
    assert parsed[9] == "2/2" and parsed[10] == "0/2" and parsed[-1] == "a / b"
    assert row.startswith("| ")


def test_worst_case_is_ten_calls_per_question_at_the_dearest_call_kind() -> None:
    pricing = Pricing({"gemini": {"gemini-m": {"standard": Rate(0.25, 1.5)}}}, usd_to_inr=85.7)
    from rflens.config import LlmSettings

    settings = LlmSettings(_env_file=None, llm_model="gemini-m", llm_service_tier="standard")
    questions = [gq("a"), gq("b")]
    per_q, total = worst_case_usd(questions, "x" * 4000, settings, 800, 700, 9000, pricing)
    assert total == pytest.approx(2 * per_q) and per_q > 0
    one_call = per_q / CALLS_PER_QUESTION
    sql_call = pricing.worst_case_usd("gemini", "gemini-m", "standard", 1000 + 3000 + 8, 800)
    assert one_call >= sql_call * 0.9  # at least about a SQL call
    capped = eval_settings(settings, 51)
    assert capped.llm_max_calls_per_run == 510 and capped.llm_fallback == ""


# --- the runner on the real graph (fake provider) ---------------------------------------------


def test_evaluate_runs_questions_through_the_graph_and_scores_the_final_state(
    make_env, tag
) -> None:
    sql_q = gq("sql-t", reference_sql="x", question=f"What is forty one plus one? {tag}")
    doc_q = gq("doc-t", "doc", question=f"What is the uplink range of LTE band 5? {tag}")
    refuse_q = gq("refuse-t", "refuse", question=f"Delete every licence. {tag}")
    script = [
        plan("sql"),
        sql_answer("SELECT 42 AS n"),
        write("The answer is 42 [S1]."),
        plan("doc", "36.101", doc_query="uplink range"),
        write("Band 5 uplink spans 824 MHz to 849 MHz [D1]."),
        plan("refuse", reason="This asks to change data."),
    ]
    env = make_env(script)
    lines: list[str] = []
    scores, stopped = evaluate(
        [sql_q, doc_q, refuse_q],
        run=env.run,
        graph=env.graph,
        run_id="e2e-test",
        stop_usd=1.0,
        emit=lines.append,
    )
    try:
        assert stopped is None and [s.qid for s in scores] == ["sql-t", "doc-t", "refuse-t"]
        assert [s.correct for s in scores] == [True, True, True], [s.reason for s in scores]
        doc = scores[1]
        assert doc.doc_hit and doc.recall_search and doc.recall_evidence and doc.cited == ["D1"]
        assert scores[2].status == "refused" and scores[2].calls == 1
        assert len(lines) == 3 and "sql-t" in lines[0] and "yes" in lines[0]
        # one thread per question, derived from the run id and the question id
        threads = {s.qid: f"e2e-test-{s.qid}" for s in scores}
        for thread in threads.values():
            assert env.graph.get_state({"configurable": {"thread_id": thread}}).values
    finally:
        for s in scores:
            env.saver.delete_thread(f"e2e-test-{s.qid}")


def test_the_cost_stop_finishes_the_current_question_and_stops(make_env, tag) -> None:
    qs = [gq(f"sql-{i}", question=f"What is forty one plus one? {tag} {i}") for i in range(3)]
    one = [plan("sql"), sql_answer("SELECT 42 AS n"), write("The answer is 42 [S1].")]
    env = make_env(one * 3, paid=True)  # a paid fake: its calls cost a little
    scores, stopped = evaluate(
        qs, run=env.run, graph=env.graph, run_id="e2e-stop", stop_usd=1e-9, emit=lambda line: None
    )
    try:
        assert len(scores) == 1 and scores[0].correct  # the first question was finished
        assert stopped and "cost stop" in stopped and "before sql-1" in stopped
        assert scores[0].cost_usd > 0
    finally:
        for s in scores:
            env.saver.delete_thread(f"e2e-stop-{s.qid}")


def test_a_run_level_stop_ends_the_command_with_the_results_so_far(make_env, tag) -> None:
    from rflens.llm.base import PaidCallBlocked

    qs = [gq(f"sql-{i}", question=f"What is forty one plus one? {tag} {i}") for i in range(2)]
    script = [
        plan("sql"),
        sql_answer("SELECT 42 AS n"),
        write("The answer is 42 [S1]."),
        PaidCallBlocked("no"),
    ]
    env = make_env(script)
    scores, stopped = evaluate(
        qs, run=env.run, graph=env.graph, run_id="e2e-halt", stop_usd=1.0, emit=lambda line: None
    )
    try:
        assert len(scores) == 1 and "sql-1: PaidCallBlocked" in stopped
    finally:
        for s in scores:
            env.saver.delete_thread(f"e2e-halt-{s.qid}")
        env.saver.delete_thread("e2e-halt-sql-1")


# --- scorer v2: rows are ordered, values within a row and tied rows are not ----------------------


def test_the_order_of_values_within_one_row_is_never_checked() -> None:
    ref = ResultSet(["call_sign", "licensee", "n"], [["KNKN259", "UNION TELEPHONE COMPANY", 1564]])
    for text in (
        "KNKN259 has 1564 records; its licensee is UNION TELEPHONE COMPANY.",
        "1564 records belong to UNION TELEPHONE COMPANY, call sign KNKN259.",
    ):
        assert contains_values(text, ref, ordered=True)[0], text


def test_rows_tied_on_the_ranking_value_are_an_unordered_group() -> None:
    ref = ResultSet(
        ["idx", "n"], [[158, 1429], [159, 1429], [293, 1429], [294, 1429], [86, 1341], [221, 1341]]
    )
    ok = "Features 294, 158, 293 and 159 each have 1429, then 221 and 86 have 1341."
    assert contains_values(ok, ref, ordered=True)[0]  # any order inside a tie group
    swapped = "Features 86 and 221 have 1341, then 158, 159, 293 and 294 have 1429."
    good, why = contains_values(swapped, ref, ordered=True)
    assert not good and "not in reference order" in why  # groups must keep their order
    assert contains_values(swapped, ref, ordered=False)[0]  # unranked questions don't care


def test_ranking_groups_use_the_monotone_column_with_ties() -> None:
    from rflens.eval.e2e_score import ranking_groups

    rows = [[158, 1429], [159, 1429], [293, 1429], [294, 1429], [86, 1341], [221, 1341]]
    assert ranking_groups(rows) == [[0, 1, 2, 3], [4, 5]]
    # ids ascending and a constant count: the id is the only ranking column, so no ties
    assert ranking_groups([[1, 5], [2, 5], [3, 5]]) == [[0], [1], [2]]
    # strings only, or numbers that are not monotone: every row stands alone
    assert ranking_groups([["a", "x"], ["b", "y"]]) == [[0], [1]]
    assert ranking_groups([[5], [9], [2]]) == [[0], [1], [2]]
    assert ranking_groups([[422], [315], [315], [42]]) == [[0], [1, 2], [3]]
    assert ranking_groups([]) == []


def test_without_a_ranking_column_rows_are_checked_one_by_one() -> None:
    ref = ResultSet(
        ["name", "k"], [["Alpha", 5], ["Beta", 5]]
    )  # constant number: no ranking column
    assert contains_values("Alpha 5, Beta 5", ref, ordered=True)[0]
    assert not contains_values("Beta 5, Alpha 5", ref, ordered=True)[0]


def test_must_mention_stays_strict_a_paraphrase_is_a_miss() -> None:
    q = gq("sql-19", rows=[[0]], columns=["grantees"], must_mention=["2021"])
    paraphrase = state(
        answer="There were 0 grantees registered in 2022; the snapshot ends before 2022 [S1].",
        rows=((0,),),
    )
    s = score_question(q, paraphrase)
    assert not s.correct and "does not mention ['2021']" in s.reason
    ok = state(answer="0 grantees in 2022; the data ends in 2021 [S1].", rows=((0,),))
    assert score_question(q, ok).correct


def test_the_real_shapes_of_the_two_fixed_misses_now_score_as_correct() -> None:
    sql_07 = gq(
        "sql-07",
        rows=[["KNKN259", "UNION TELEPHONE COMPANY", 1564]],
        columns=["call_sign", "licensee_name", "n"],
        ordered=True,
    )
    answer = (
        "The call sign with the most frequency records is KNKN259, with 1564 records [S1]. "
        "Its licensee is UNION TELEPHONE COMPANY [S1]."
    )
    s7 = state(
        answer=answer,
        columns=("call_sign", "licensee_name", "n"),
        rows=(("KNKN259", "UNION TELEPHONE COMPANY", 1564),),
    )
    assert score_question(sql_07, s7).correct


# --- rescoring saved threads -------------------------------------------------------------------


def test_rescore_reads_the_saved_threads_and_reports_old_against_new(make_env, tag) -> None:
    from rflens.eval.eval_e2e import render_rescore, rescore_run, thread_state

    q = gq("sql-t", reference_sql="x", question=f"What is forty one plus one? {tag}")
    env = make_env([plan("sql"), sql_answer("SELECT 42 AS n"), write("The answer is 42 [S1].")])
    scores, _ = evaluate(
        [q],
        run=env.run,
        graph=env.graph,
        run_id="e2e-rescore",
        stop_usd=1.0,
        emit=lambda line: None,
    )
    try:
        assert thread_state(env.saver, "e2e-rescore-sql-t")["status"] == "answered"
        again = rescore_run("e2e-rescore", [q, gq("sql-never-run")], env.saver)
        assert [s.qid for s in again] == ["sql-t"] and again[0].correct == scores[0].correct
        text = render_rescore("e2e-rescore", {"sql-t": {"correct": False}}, again)
        assert "sql-t" in text and "<- changed" in text and "old 0/1 correct, new 1/1" in text
    finally:
        env.saver.delete_thread("e2e-rescore-sql-t")


# --- a loop fails one question; run-level caps still stop the run ------------------------------


def test_a_loop_in_one_question_does_not_stop_the_run(make_env, tag) -> None:
    from rflens.llm.base import LoopDetected

    qs = [gq(f"sql-{i}", question=f"What is forty one plus one? {tag} {i}") for i in range(2)]
    # the first question's writer call raises before it reaches the provider: its answer is unused
    first = [plan("sql"), sql_answer("SELECT 42 AS n")]
    second = [plan("sql"), sql_answer("SELECT 42 AS n"), write("The answer is 42 [S1].")]
    env = make_env(first + second)
    real = env.run.generate

    def generate(messages, **kw):
        if kw["question_id"].endswith("sql-0") and kw["purpose"] == "writer":
            raise LoopDetected("repeat_prompt", env.run.run_id, "same prompt", kw["question_id"])
        return real(messages, **kw)

    env.run.generate = generate
    scores, stopped = evaluate(
        qs, run=env.run, graph=env.graph, run_id="e2e-loop", stop_usd=1.0, emit=lambda line: None
    )
    try:
        assert stopped is None and [s.qid for s in scores] == ["sql-0", "sql-1"]
        assert scores[0].status == "failed" and "loop_detected" in scores[0].failure_types
        assert not scores[0].correct and scores[0].cause == "other"
        assert scores[1].correct  # the next question ran normally
    finally:
        for s in scores:
            env.saver.delete_thread(f"e2e-loop-{s.qid}")


def test_the_run_call_cap_still_stops_the_run(make_env, tag) -> None:
    qs = [gq(f"sql-{i}", question=f"What is forty one plus one? {tag} {i}") for i in range(2)]
    one = [plan("sql"), sql_answer("SELECT 42 AS n"), write("The answer is 42 [S1].")]
    env = make_env(one * 2, llm_kw={"llm_max_calls_per_run": 4})
    scores, stopped = evaluate(
        qs, run=env.run, graph=env.graph, run_id="e2e-cap", stop_usd=1.0, emit=lambda line: None
    )
    try:
        assert len(scores) == 1 and "sql-1: CallCapExceeded" in stopped
    finally:
        for s in scores:
            env.saver.delete_thread(f"e2e-cap-{s.qid}")
        env.saver.delete_thread("e2e-cap-sql-1")
