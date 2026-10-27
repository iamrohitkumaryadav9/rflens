"""The grown golden set: authors, doc targets, mixed questions and the assistant questions.

Doc and mixed targets cannot be checked against the 3GPP corpus here (the corpus is in the
developer database, never in the repo or CI), so these tests check what can be known offline: the
shape, the chunk-id format, and consistency between a question's target and its confirmed chunk.
"""

import re

import pytest

from rflens.eval.golden import GoldenQuestion, load_golden, problems
from rflens.mcp.docs_server import CHUNK_ID_RE
from rflens.mcp.sql_guard import check_query

GOLDEN = load_golden()
BY_ID = {q.id: q for q in GOLDEN}


def of(kind: str, author: str | None = None) -> list[GoldenQuestion]:
    return [q for q in GOLDEN if q.type == kind and author in (None, q.author)]


def test_the_file_validates_and_counts_per_type_and_author() -> None:
    assert problems(GOLDEN) == []
    counts = {
        (t, a): len(of(t, a))
        for t in ("sql", "doc", "mixed", "stats", "refuse")
        for a in ("human", "assistant")
    }
    assert counts == {
        ("sql", "human"): 17,
        ("sql", "assistant"): 2,
        ("doc", "human"): 10,
        ("doc", "assistant"): 12,
        ("mixed", "human"): 0,
        ("mixed", "assistant"): 4,
        ("stats", "human"): 0,
        ("stats", "assistant"): 4,
        ("refuse", "human"): 3,
        ("refuse", "assistant"): 3,
    }
    assert len(GOLDEN) == 55 and len({q.id for q in GOLDEN}) == 55


def test_authors_and_statuses_are_consistent() -> None:
    human = [q for q in GOLDEN if q.author == "human"]
    assistant = [q for q in GOLDEN if q.author == "assistant"]
    assert {q.status for q in human} == {"reviewed"}
    assert {q.status for q in assistant} == {"reviewed"}  # the owner reviewed all 25 drafts
    # the questions that existed before the assistant's are human; mixed-03 was dropped in review
    assert {q.id for q in assistant} == (
        {f"doc-{i}" for i in range(11, 23)}
        | {f"mixed-{i:02d}" for i in (1, 2, 4, 5)}
        | {"sql-18", "sql-19", "refuse-04", "refuse-05", "refuse-06"}
        | {f"stats-{i:02d}" for i in range(1, 5)}
    )


# --- doc questions ----------------------------------------------------------------------------


def test_the_ten_smoke_questions_moved_in_as_doc_01_to_10() -> None:
    smoke = of("doc", "human")
    assert [q.id for q in smoke] == [f"doc-{i:02d}" for i in range(1, 11)]
    assert [q.spec for q in smoke] == ["38.101-1"] * 8 + ["36.101"] * 2
    assert smoke[1].question.startswith("What is the maximum output power and tolerance")
    assert smoke[1].expected_clauses == ["6.2.1"] and smoke[1].expected_tables == ["6.2.1-1"]
    assert all("smoke set" in q.notes for q in smoke)


def test_assistant_doc_questions_have_one_confirmed_target_each() -> None:
    drafts = of("doc", "assistant")
    assert len(drafts) == 12 and sum(q.spec == "38.101-1" for q in drafts) == 6
    for q in drafts:
        assert len(q.expected_clauses) + len(q.expected_tables) == 1, q.id
        assert q.confirmed_chunks, q.id
        for chunk_id in q.confirmed_chunks:
            assert CHUNK_ID_RE.fullmatch(chunk_id), chunk_id
            assert chunk_id.startswith(q.spec + "@"), q.id
            if q.expected_clauses:  # clause targets: the confirmed chunk is in that clause
                assert chunk_id.split("/")[1] == q.expected_clauses[0], q.id
    assert len({q.question for q in drafts}) == 12


def test_assistant_doc_questions_cover_transmitter_receiver_and_bands_with_mixed_phrasing() -> None:
    drafts = of("doc", "assistant")
    per_spec = {s: [q for q in drafts if q.spec == s] for s in ("36.101", "38.101-1")}
    for questions in per_spec.values():
        text = " ".join(q.notes for q in questions)
        assert "Transmitter" in text and "Receiver" in text and "Band definition" in text
    lookups = [q for q in drafts if "table lookup with a specific band" in q.notes]
    assert len(lookups) >= 2
    assert all(re.search(r"\bn?5\b", q.question) for q in lookups)
    identifier = [q for q in drafts if re.search(r"NS_\d+|\bn?\d{1,2}\b|clause 6\.\d", q.question)]
    plain = [q for q in drafts if q not in identifier]
    assert len(identifier) >= 5 and len(plain) >= 4  # both styles are well represented


def test_doc_18_says_single_carrier_so_the_ca_clause_is_not_an_answer() -> None:
    q = BY_ID["doc-18"]
    assert "single carrier" in q.question and "NS_07" in q.question
    assert "not category M1 or M2" in q.question  # so clause 6.2.4E is excluded by the wording
    assert "excluded by the wording" in q.notes and "does not exclude" not in q.notes
    assert q.expected_tables == ["6.2.4-2"] and q.confirmed_chunks == ["36.101@19.6.0/6.2.4/4"]
    assert "6.2.4A.7" in q.notes and "CA_NS_07" in q.notes  # the excluded clause is named


def test_mixed_03_was_dropped_in_review() -> None:
    assert "mixed-03" not in BY_ID  # unnatural question, the answer rested on a single record


def test_notes_quote_no_spec_text() -> None:
    """3GPP specs are copyrighted: notes name clauses and tables, they do not quote paragraphs."""
    for q in of("doc") + of("mixed"):
        assert len(q.notes) < 900, q.id
        assert "shall" not in q.notes.lower(), q.id


# --- mixed questions --------------------------------------------------------------------------


def test_mixed_questions_pair_one_spec_target_with_a_sql_lookup() -> None:
    mixed = of("mixed")
    assert [q.id for q in mixed] == ["mixed-01", "mixed-02", "mixed-04", "mixed-05"]
    for q in mixed:
        assert q.expected_clauses or q.expected_tables, q.id
        assert len(q.expected_clauses) + len(q.expected_tables) == 1, q.id
        assert q.confirmed_chunks and q.reference_sql and q.reference_result.rows, q.id
        check_query(q.reference_sql)  # passes the sql-mcp guard
        assert "TS 36.101" in q.question or "TS 38.101-1" in q.question, q.id
        assert "ASSUMED MHz" in q.notes and "Not scored" not in q.notes, q.id
        assert "uls_fr rows" not in q.notes, q.id


def test_mixed_sql_uses_the_band_edges_from_the_spec_tables() -> None:
    assert "824" in BY_ID["mixed-01"].reference_sql and "849" in BY_ID["mixed-01"].reference_sql
    assert "869" in BY_ID["mixed-02"].reference_sql and "894" in BY_ID["mixed-02"].reference_sql
    assert "869" in BY_ID["mixed-04"].reference_sql and BY_ID["mixed-04"].must_mention == [
        "unknown"
    ]
    assert "880" in BY_ID["mixed-05"].reference_sql and "915" in BY_ID["mixed-05"].reference_sql
    assert BY_ID["mixed-01"].spec == "36.101" and BY_ID["mixed-02"].spec == "38.101-1"
    assert BY_ID["mixed-02"].expected_tables == ["5.2-1"]
    assert BY_ID["mixed-01"].expected_clauses == ["5.5"]


def test_mixed_reference_results() -> None:
    top5 = BY_ID["mixed-01"]
    assert top5.reference_result.columns == ["active_licences"] and top5.ordered
    # counts only, like sql-18; rank 5 (42) is above rank 6 (27); the companies are in the notes
    assert top5.reference_result.rows == [[422], [315], [283], [278], [42]]
    assert "licensee_name" not in top5.reference_sql and "LIMIT 5" in top5.reference_sql
    for name in ("Cellco Partnership", "CommNet Cellular Inc.", "0003290673", "0017329715"):
        assert name in top5.notes, name
    assert BY_ID["mixed-02"].reference_result.rows == [[116525]]
    assert BY_ID["mixed-04"].reference_result.rows == [["KNKA683", 1787.3]]
    assert BY_ID["mixed-05"].reference_result.rows == [[88057]]


# --- the two new sql drafts and the refuse drafts ---------------------------------------------


def test_sql_18_reference_groups_by_frn_though_the_question_does_not_say_so() -> None:
    q = BY_ID["sql-18"]
    assert (
        q.question
        == "Which company holds the most active cellular licences, excluding spectrum leases?"
    )
    assert "FRN" not in q.question and "Registration" not in q.question
    assert "GROUP BY licensee_frn" in q.reference_sql and "NOT is_lease" in q.reference_sql
    assert q.reference_result.rows == [[422]] and q.ordered
    assert "418" in q.notes and "315" in q.notes  # the by-name value, and rank 2


def test_mixed_05_wording_after_review() -> None:
    q = BY_ID["mixed-05"]
    assert "fall inside the uplink range of NR band n8" in q.question
    assert q.question.endswith("(These cellular frequencies would overlap n8 uplink use.)")
    assert q.reference_result.rows == [[88057]]


def test_sql_19_needs_the_snapshot_cutoff_that_the_question_does_not_mention() -> None:
    q = BY_ID["sql-19"]
    assert q.question == "How many FCC equipment grantees were registered in 2022?"
    assert "2021" not in q.question and q.must_mention == ["2021"]
    assert q.reference_result.rows == [[0]] and "2,779" in q.notes


@pytest.mark.parametrize(
    ("qid", "word"),
    [("refuse-04", "DROP"), ("refuse-05", "address"), ("refuse-06", "prediction")],
)
def test_assistant_refuse_questions(qid: str, word: str) -> None:
    q = BY_ID[qid]
    assert q.author == "assistant" and q.status == "reviewed" and q.reference_sql is None
    assert word.lower() in (q.expected_refusal + " " + q.notes).lower()
    if qid == "refuse-04":
        assert q.question.startswith("Could you please drop") and "table" in q.question
    if qid == "refuse-05":
        assert "home address" in q.question and "KNKA209" in q.question
    if qid == "refuse-06":
        assert "will be active" in q.question


def test_doc_03_and_doc_04_say_nr_and_record_why() -> None:
    for qid in ("doc-03", "doc-04"):
        q = BY_ID[qid]
        assert " NR " in q.question and q.spec == "38.101-1" and q.status == "reviewed"
        assert "names no spec" in q.notes and "exist in both" in q.notes
    assert BY_ID["doc-03"].expected_clauses == ["6.2.2"]
    assert BY_ID["doc-04"].expected_clauses == ["6.2.3.2"]
    assert "NS_04" in BY_ID["doc-04"].question
    # doc-12 stays as written
    assert BY_ID["doc-12"].question == (
        "What additional maximum power reduction applies when network signalling value NS_37 "
        "is used?"
    )
