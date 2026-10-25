"""Verifier unit tests: pure functions, no database, no LLM."""

import pytest

from rflens.graph.safety import TracingEnabledError, assert_tracing_off
from rflens.graph.spec_rules import infer_spec, merge_with_hint
from rflens.graph.verifier import sql_band_edges, verify

CHUNK_TEXT = (
    "TS 36.101 V19.6.0 clause 5.5\n\nOperating band 5 uplink 824 MHz to 849 MHz, "
    'downlink 869 MHz to 894 MHz. The note says "the band is paired" for this table.'
)
CHUNK = {
    "ref": "D1",
    "chunk_id": "36.101@19.6.0/5.5/2",
    "spec": "36.101",
    "version": "19.6.0",
    "clause": "5.5",
    "clause_title": "Operating bands",
    "table_number": "5.5-1",
    "text": CHUNK_TEXT,
}


def sql_result(rows=None, columns=None, executed="SELECT 1", flags=None):
    rows = [[104, 1567]] if rows is None else rows
    return {
        "status": "answered",
        "columns": columns or ["failed_runs", "total_runs"],
        "rows": rows,
        "row_count": len(rows),
        "executed_sql": executed,
        "query_id": 7,
        "column_flags": flags or {},
    }


def check(draft, *, sql=None, chunks=(), route="sql", question="How many runs failed?"):
    return verify(draft, question=question, route=route, sql=sql, chunks=list(chunks))


def kinds(result):
    return sorted({f["type"] for f in result["failures"]})


def test_exact_number_passes_and_evidence_map_names_the_source():
    r = check("104 runs failed [S1].", sql=sql_result())
    assert r["ok"], r["failures"]
    assert r["evidence_map"][0]["numbers"] == [{"number": "104", "source": "sql", "match": "exact"}]


def test_invented_number_fails():
    r = check("987 runs failed [S1].", sql=sql_result())
    assert kinds(r) == ["unsupported_number"]


def test_rounded_numbers_pass_at_their_own_precision():
    sql = sql_result(rows=[[12.346, 1234567]], columns=["share", "n"])
    for text in ("The share is 12.35 [S1].", "The share is 12.3 [S1].", "The share is 12 [S1]."):
        assert check(text, sql=sql)["ok"], text
    r = check("The share is 12.9 [S1].", sql=sql)
    assert kinds(r) == ["unsupported_number"]


def test_thousands_separators_equal_the_plain_number():
    sql = sql_result(rows=[[1234567]], columns=["n"])
    r = check("There are 1,234,567 rows [S1].", sql=sql)
    assert r["ok"], r["failures"]
    assert check("There are 1234567 rows [S1].", sql=sql)["ok"]


def test_derived_values_pass_when_computed_from_one_column():
    sql = sql_result(rows=[[104], [1463]], columns=["n"])
    for text in (
        "In total 1567 runs [S1].",  # sum
        "That is 1359 more [S1].",  # difference
        "The failed share is 6.6% of the runs [S1].",  # 104 / 1567 * 100
    ):
        assert check(text, sql=sql)["ok"], text
    assert kinds(check("It is 777777 in total [S1].", sql=sql)) == ["unsupported_number"]


def test_numbers_of_the_question_and_the_sql_need_no_citation():
    sql = sql_result(rows=[["A"], ["B"]], columns=["name"], executed="SELECT name FROM t LIMIT 5")
    r = check(
        "These are the top 5 names. They are A and B [S1].",
        sql=sql,
        question="Which 5 names come first?",
    )
    assert r["ok"], r["failures"]


def test_invented_unit():
    r = check("The value is 104 MHz [S1].", sql=sql_result())
    assert kinds(r) == ["invented_unit"]


def test_unit_from_a_cited_chunk_is_allowed():
    r = check("Band 5 uplink spans 824 MHz to 849 MHz [D1].", chunks=[CHUNK], route="doc")
    assert r["ok"], r["failures"]


def test_unknown_unit_column_allows_no_unit():
    flags = {"power_erp": {"unknown": True, "assumed": False, "units": []}}
    sql = sql_result(rows=[[140.8]], columns=["power_erp"], flags=flags)
    assert check("The ERP is 140.8 [S1].", sql=sql)["ok"]
    assert kinds(check("The ERP is 140.8 dBm [S1].", sql=sql)) == ["invented_unit"]


def test_assumed_column_requires_assumption_wording():
    flags = {"frequency_assigned": {"unknown": False, "assumed": True, "units": ["mhz"]}}
    sql = sql_result(rows=[[835.0]], columns=["frequency_assigned"], flags=flags)
    assert kinds(check("The frequency is 835 MHz [S1].", sql=sql)) == ["missing_assumption_wording"]
    ok = check("The frequency is 835 MHz (unit assumed) [S1].", sql=sql)
    assert ok["ok"], ok["failures"]


def test_unsupported_quote():
    bad = check('The spec says "the band is unpaired" [D1].', chunks=[CHUNK], route="doc")
    assert kinds(bad) == ["unsupported_quote"]
    good = check('The spec says "the band is paired" [D1].', chunks=[CHUNK], route="doc")
    assert good["ok"], good["failures"]


def test_missing_citation_on_a_sourced_number():
    r = check("104 runs failed.", sql=sql_result())
    assert "missing_citation" in kinds(r)


def test_draft_without_any_citation_fails_when_there_is_evidence():
    r = check("The runs failed.", sql=sql_result())
    assert kinds(r) == ["missing_citation"]


def test_dangling_citations():
    r = check("104 runs failed [S1] and more [D3] [S2].", sql=sql_result(), chunks=[CHUNK])
    assert [f["detail"] for f in r["failures"] if f["type"] == "dangling_citation"] == [
        "[D3] does not exist",
        "[S2] does not exist",
    ]


def test_empty_draft_fails():
    assert kinds(check("   ", sql=sql_result())) == ["empty_answer"]


MIXED_SQL = (
    "SELECT count(*) FROM v_uls_frequencies f WHERE f.frequency_assigned >= 824 "
    "AND COALESCE(f.frequency_upper_band, f.frequency_assigned) <= 849 AND f.license_status = 'A'"
)


def test_band_edges_are_read_from_where_clauses():
    assert sql_band_edges(MIXED_SQL) == [
        ("frequency_assigned", 824.0),
        ("frequency_upper_band", 849.0),
    ]
    assert sql_band_edges("SELECT 824 AS x FROM t") == []
    assert sql_band_edges("SELECT * FROM t WHERE f.frequency BETWEEN 1710 AND 1785") == [
        ("frequency", 1710.0),
        ("frequency", 1785.0),
    ]
    assert sql_band_edges("SELECT * FROM t WHERE n > 100") == []  # not a frequency column
    assert sql_band_edges("not sql at all (") == []


def _mixed(executed, draft="The count is 41 [S1]. Band 5 uplink is 824 MHz to 849 MHz [D1]."):
    sql = sql_result(rows=[[41]], columns=["n"], executed=executed)
    return check(draft, sql=sql, chunks=[CHUNK], route="mixed")


def test_sql_literals_present_in_a_cited_chunk_pass():
    r = _mixed(MIXED_SQL)
    assert r["ok"], r["failures"]


def test_invented_band_edge_in_the_sql_fails():
    wrong = MIXED_SQL.replace("824", "814")
    r = _mixed(wrong)
    assert kinds(r) == ["unsupported_sql_literal"]
    assert "814" in r["failures"][0]["detail"]


def test_band_edge_in_an_uncited_chunk_does_not_count():
    r = _mixed(MIXED_SQL, draft="The count is 41 [S1].")
    assert "unsupported_sql_literal" in kinds(r)


def test_sql_literal_check_applies_to_mixed_questions_only():
    sql = sql_result(rows=[[41]], columns=["n"], executed=MIXED_SQL.replace("824", "814"))
    assert check("The count is 41 [S1].", sql=sql, route="sql")["ok"]


def test_spec_rules():
    assert infer_spec("What is the power class in band n78?").spec == "38.101-1"
    assert infer_spec("LTE band 5 uplink").spec == "36.101"
    assert infer_spec("Compare NR n78 with LTE band 42").source == "ambiguous"
    assert infer_spec("What is band 5?").source == "none"
    rules = infer_spec("NR n78 power")
    assert merge_with_hint(rules, "36.101") == ("38.101-1", "rule", True)
    assert merge_with_hint(rules, "38.101-1") == ("38.101-1", "rule", False)
    assert merge_with_hint(infer_spec("band 5"), "36.101") == ("36.101", "llm-hint", False)
    assert merge_with_hint(infer_spec("band 5"), "none") == (None, "none", False)


def test_tracing_switches_are_refused():
    assert_tracing_off({"LANGSMITH_TRACING": "false"})
    assert_tracing_off({})
    for name in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2"):
        with pytest.raises(TracingEnabledError):
            assert_tracing_off({name: "true"})


# --- a band row whose dash is missing in the source (36.101 band 41 is one) -----------------------

MISSING_DASH_ROW = "77 | 1111 MHz |  | 1999 MHz | 1111 MHz |  | 1999 MHz | TDD"  # invented numbers


def test_both_edges_of_a_row_with_a_missing_dash_are_separate_numbers() -> None:
    from rflens.graph.verifier import numbers_in

    assert {1111.0, 1999.0} <= numbers_in(MISSING_DASH_ROW)
    chunk = {**CHUNK, "text": f"TS 99.101 V1 clause 1\n\n{MISSING_DASH_ROW}\n"}
    ok = check("Band 77 spans 1111 MHz to 1999 MHz [D1].", chunks=[chunk], route="doc")
    assert ok["ok"], ok["failures"]


def test_band_41_in_the_real_chunk_text_yields_2496_and_2690() -> None:
    """The actual 3GPP chunk text lives only in the gitignored data/ folder, so this runs where
    the spec chunks exist (the developer's machine), not in CI."""
    import json

    from rflens.graph.verifier import numbers_in
    from rflens.ingest.spec_bands import INTERIM

    path = INTERIM / "36.101" / "chunks.jsonl"
    if not path.exists():
        pytest.skip(
            "data/interim/3gpp/36.101/chunks.jsonl is not present (3GPP text is not committed)"
        )
    chunk = next(
        json.loads(line)
        for line in path.open(encoding="utf-8")
        if json.loads(line)["chunk_id"] == "36.101@19.6.0/5.5/2"
    )
    row = next(line for line in chunk["text"].splitlines() if line.startswith("41 |"))
    assert {2496.0, 2690.0} <= numbers_in(row)
    result = check(
        "Band 41 spans 2496 MHz to 2690 MHz [D1].",
        chunks=[{**CHUNK, "text": chunk["text"]}],
        route="doc",
    )
    assert result["ok"], result["failures"]


# --- edge_side_mismatch: the SQL must use the side the question asks for ------------------------

BAND_JOIN_SQL = (
    "SELECT count(*) FROM v_uls_frequencies f JOIN spec_operating_bands b "
    "ON b.spec = '36.101' AND b.band = '5' WHERE f.frequency_assigned >= b.{lo} "
    "AND f.frequency_assigned <= b.{hi}"
)


def side_check(question, lo, hi, sql_question=""):
    sql = sql_result(rows=[[41]], columns=["n"], executed=BAND_JOIN_SQL.format(lo=lo, hi=hi))
    return verify(
        "The count is 41 [S1].",
        question=question,
        route="mixed",
        sql=sql,
        chunks=[],
        sql_question=sql_question,
    )


def test_uplink_question_with_downlink_columns_fails() -> None:
    r = side_check(
        "How many records are in the LTE band 5 uplink range?", "dl_low_mhz", "dl_high_mhz"
    )
    assert kinds(r) == ["edge_side_mismatch"]
    assert "uplink" in r["failures"][0]["detail"] and "downlink" in r["failures"][0]["detail"]


def test_downlink_question_with_uplink_columns_fails() -> None:
    r = side_check("How many records are in the band 5 downlink?", "ul_low_mhz", "ul_high_mhz")
    assert kinds(r) == ["edge_side_mismatch"]


def test_the_matching_side_passes_and_the_joined_band_is_reported() -> None:
    r = side_check("How many records are in the LTE band 5 UL range?", "ul_low_mhz", "ul_high_mhz")
    assert r["ok"], r["failures"]
    assert r["band_joins"] == [{"spec": "36.101", "band": "5"}]


def test_a_question_about_both_directions_needs_both_sides() -> None:
    both = "How many records are in neither the uplink nor the downlink of band 5?"
    assert kinds(side_check(both, "ul_low_mhz", "ul_high_mhz")) == ["edge_side_mismatch"]
    assert kinds(side_check(both, "dl_low_mhz", "dl_high_mhz")) == ["edge_side_mismatch"]
    sql = sql_result(
        rows=[[41]],
        columns=["n"],
        executed="SELECT count(*) FROM spec_operating_bands b WHERE b.ul_low_mhz < 1 "
        "AND b.dl_high_mhz > 1 AND b.band = '5'",
    )
    r = verify("The count is 41 [S1].", question=both, route="mixed", sql=sql, chunks=[])
    assert r["ok"], r["failures"]


def test_the_restated_question_counts_for_the_direction_too() -> None:
    r = side_check(
        "How many records are inside band 5?", "dl_low_mhz", "dl_high_mhz", "in the uplink"
    )
    assert kinds(r) == ["edge_side_mismatch"]


def test_no_direction_word_or_no_band_table_means_no_side_check() -> None:
    assert side_check("How many records are inside band 5?", "dl_low_mhz", "dl_high_mhz")["ok"]
    plain = sql_result(
        rows=[[41]], columns=["n"], executed="SELECT count(*) FROM v_uls_frequencies"
    )
    r = verify("The count is 41 [S1].", question="uplink?", route="mixed", sql=plain, chunks=[])
    assert r["ok"]
    sql_route = verify(
        "The count is 41 [S1].",
        question="uplink?",
        route="sql",
        sql=sql_result(
            rows=[[41]],
            columns=["n"],
            executed=BAND_JOIN_SQL.format(lo="dl_low_mhz", hi="dl_high_mhz"),
        ),
        chunks=[],
    )
    assert sql_route["ok"]


# --- frn_name_grouping: companies are identified by licensee_frn ----------------------------


def frn_check(executed: str, route: str = "sql"):
    sql = sql_result(rows=[["x", 1]], columns=["licensee", "n"], executed=executed)
    return verify("There is one row [S1].", question="Who?", route=route, sql=sql, chunks=[])


GOOD_GROUPING = (
    "SELECT licensee_frn, min(licensee_name) AS licensee, count(*) AS n FROM v_uls_licences "
    "GROUP BY licensee_frn ORDER BY n DESC LIMIT 5"
)


def test_grouping_by_frn_alone_passes() -> None:
    assert frn_check(GOOD_GROUPING)["ok"]


def test_grouping_by_name_together_with_frn_fails() -> None:
    r = frn_check(
        GOOD_GROUPING.replace("GROUP BY licensee_frn", "GROUP BY licensee_frn, licensee_name")
    )
    assert kinds(r) == ["frn_name_grouping"]
    assert "together with licensee_frn" in r["failures"][0]["detail"]


def test_grouping_by_name_alone_fails_when_the_frn_is_available() -> None:
    r = frn_check("SELECT licensee_name, count(*) AS n FROM v_uls_licences GROUP BY licensee_name")
    assert kinds(r) == ["frn_name_grouping"]
    assert "although licensee_frn is available" in r["failures"][0]["detail"]


def test_ordinals_aliases_and_qualified_names_are_resolved() -> None:
    assert kinds(
        frn_check("SELECT l.licensee_name, count(*) FROM v_uls_licences l GROUP BY 1")
    ) == ["frn_name_grouping"]
    assert kinds(
        frn_check(
            "SELECT l.licensee_name AS company, count(*) FROM v_uls_licences l GROUP BY company"
        )
    ) == ["frn_name_grouping"]
    assert kinds(
        frn_check(
            "SELECT l.licensee_frn, l.licensee_name, count(*) FROM v_uls_licences l GROUP BY 1, 2"
        )
    ) == ["frn_name_grouping"]
    assert frn_check(
        "SELECT l.licensee_frn, min(l.licensee_name), count(*) FROM v_uls_licences l GROUP BY 1"
    )["ok"]


def test_a_subquery_or_cte_is_checked_too() -> None:
    sql = (
        "WITH c AS (SELECT licensee_name, count(*) AS n FROM v_uls_licences "
        "GROUP BY licensee_name) "
        "SELECT * FROM c"
    )
    assert kinds(frn_check(sql)) == ["frn_name_grouping"]


def test_other_groupings_and_other_relations_are_not_touched() -> None:
    assert frn_check("SELECT license_status, count(*) FROM v_uls_licences GROUP BY license_status")[
        "ok"
    ]
    assert frn_check("SELECT firm_name, count(*) FROM fcc_test_firms GROUP BY firm_name")["ok"]
    assert frn_check("SELECT licensee_name FROM v_uls_licences LIMIT 3")["ok"]  # no grouping
    assert frn_check("not sql at all (")["ok"]


def test_the_rule_applies_to_mixed_questions_too() -> None:
    sql = "SELECT licensee_name, count(*) FROM v_uls_licences GROUP BY licensee_name"
    assert kinds(frn_check(sql, route="mixed")) == ["frn_name_grouping"]


# --- missing_band_citation ------------------------------------------------------------------------

BAND_CHUNK = {
    **CHUNK,
    "band_lines": ["TS 36.101 band 5: uplink 824-849 MHz, downlink 869-894 MHz, FDD"],
}


def test_an_answer_must_cite_a_chunk_that_supplied_band_rows() -> None:
    sql = sql_result(
        rows=[[41]], columns=["n"], executed=BAND_JOIN_SQL.format(lo="ul_low_mhz", hi="ul_high_mhz")
    )
    cites_sql_only = verify(
        "The uplink count is 41 [S1].",
        question="How many in the uplink of band 5?",
        route="mixed",
        sql=sql,
        chunks=[BAND_CHUNK],
    )
    assert kinds(cites_sql_only) == ["missing_band_citation"]
    assert "[D1]" in cites_sql_only["failures"][0]["detail"]
    cites_band = verify(
        "The uplink count is 41 [S1]; band 5 uplink is 824 MHz to 849 MHz [D1].",
        question="How many in the uplink of band 5?",
        route="mixed",
        sql=sql,
        chunks=[BAND_CHUNK],
    )
    assert cites_band["ok"], cites_band["failures"]


def test_only_a_band_chunk_counts_and_only_when_band_rows_were_supplied() -> None:
    other = {**CHUNK, "ref": "D2", "chunk_id": "36.101@19.6.0/6.2.2/1"}
    sql = sql_result(rows=[[41]], columns=["n"])
    cited_other = verify(
        "The count is 41 [S1]. Another clause [D2].",
        question="q",
        route="mixed",
        sql=sql,
        chunks=[BAND_CHUNK, other],
    )
    assert kinds(cited_other) == ["missing_band_citation"]  # D2 holds no band row
    no_bands = verify(
        "The count is 41 [S1] and a clause [D1].",
        question="q",
        route="mixed",
        sql=sql,
        chunks=[CHUNK],
    )
    assert no_bands["ok"], no_bands["failures"]  # a search-based mixed answer needs no band cite
