"""Clause lookup and the variant-clause rule (retrieval/variants.py, retrieval/hybrid.py), on the
real test database with the hashing embedder and synthetic chunks in a rolled-back transaction.
"""

# ruff: noqa: F811
# (the conn fixture imported from test_docs_mcp is used by name)

import pytest

from rflens.config import RetrievalSettings
from rflens.retrieval.embedder import HashEmbedder
from rflens.retrieval.hybrid import search
from rflens.retrieval.index import embed_pending, load_chunks
from rflens.retrieval.variants import (
    TOPICS,
    clause_letters,
    extract_refs,
    is_downranked,
    mentions_topic,
)
from test_docs_mcp import LTE, NR, conn, simple_row  # noqa: F401

# --- the pure rules ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "clauses", "tables"),
    [
        ("What does clause 6.6.2.1 require?", ["6.6.2.1"], []),
        ("see Clause 5.5 and subclause 6.2.4A.7.", ["5.5", "6.2.4A.7"], []),
        ("What is in Table 7.3.1-1 for band 5?", [], ["7.3.1-1"]),
        ("Table 7.3.2-1a and table 6.2.1-1", [], ["7.3.2-1a", "6.2.1-1"]),
        ("clause 6.6.2.1 and again clause 6.6.2.1", ["6.6.2.1"], []),
        ("a 10 MHz channel, 15 kHz spacing, power class 3, 6.2 dB", [], []),  # bare numbers
        ("NS_07 in band n78", [], []),
        ("section 6.2.3 and Section 7.3", ["6.2.3", "7.3"], []),
    ],
)
def test_extract_refs(text, clauses, tables) -> None:
    assert extract_refs(text) == (clauses, tables)


@pytest.mark.parametrize(
    ("clause", "letters"),
    [
        ("6.2.1", ()),
        ("5.5H", ("H",)),
        ("6.2.4A.7", ("A",)),
        ("5.2E.1F", ("E", "F")),
        ("7.3E.2F.2", ("E", "F")),
        ("A.3.9.2", ()),  # an annex clause is not a variant
        ("D.1", ()),
        ("", ()),
    ],
)
def test_clause_letters(clause, letters) -> None:
    assert clause_letters(clause) == letters


def test_a_variant_clause_is_down_ranked_unless_its_topic_is_mentioned() -> None:
    plain = "what is the output power of the UE"
    assert not is_downranked("36.101", "6.2.2", plain)  # a base clause is never down-ranked
    assert is_downranked("36.101", "6.2.4A.7", plain)  # carrier aggregation not mentioned
    assert not is_downranked("36.101", "6.2.4A.7", "A-MPR with carrier aggregation")
    assert not is_downranked("36.101", "6.2.4A.7", "the CA_NS_07 requirement")
    assert is_downranked("36.101", "5.5H", "operating bands of band 5")
    assert not is_downranked("36.101", "5.5H", "operating bands for the broadcast mode")
    assert is_downranked("36.101", "6.3.2F", "minimum output power")
    assert not is_downranked("36.101", "6.3.2F", "minimum output power of an NB1 UE")
    assert not is_downranked("36.101", "6.2.3E", "category M1 maximum power reduction")
    # the letters mean different things in the two specs
    assert is_downranked("38.101-1", "6.2D.1", "carrier aggregation")  # D is UL MIMO in NR
    assert not is_downranked("38.101-1", "6.2D.1", "UL MIMO output power")
    assert not is_downranked("36.101", "6.6.1D", "ProSe occupied bandwidth")  # D is ProSe in LTE


def test_a_clause_with_two_letters_needs_both_topics() -> None:
    clause = "5.2E.1F"  # NR: sidelink / V2X, with additional requirements
    assert is_downranked("38.101-1", clause, "sidelink operating bands")
    assert is_downranked("38.101-1", clause, "additional requirements")
    assert not is_downranked("38.101-1", clause, "additional requirements for sidelink")


def test_every_letter_found_in_the_clause_titles_has_a_topic() -> None:
    lte = set("ABCDEFGHIJK")  # from the titles of the loaded TS 36.101 / TS 38.101-1
    nr = set("ABCDEFGHIJKLM")
    assert {letter for spec, letter in TOPICS if spec == "36.101"} == lte
    assert {letter for spec, letter in TOPICS if spec == "38.101-1"} == nr
    assert not mentions_topic("36.101", "Z", "carrier aggregation")  # unknown letter: never excused


def test_the_rules_are_on_by_default() -> None:
    s = RetrievalSettings(_env_file=None)
    assert s.clause_lookup and s.variant_downrank


# --- in the search -----------------------------------------------------------------------------


def load(conn, source, rows) -> None:
    ids = load_chunks(conn, source, rows, prune=False)
    embed_pending(conn, HashEmbedder(), ids, progress=lambda s: None)


QUERY = "uplink transmit power limit widget"


def clauses_of(hits) -> list[str]:
    return [h.clause for h in hits]


def test_variant_clauses_move_behind_the_base_clause_unless_the_topic_is_named(conn) -> None:
    load(
        conn,
        NR,
        [
            # the variant repeats the query word for word, so it is the nearest chunk
            simple_row("38.101-1", 1, QUERY, clause="6.2.1A"),
            simple_row(
                "38.101-1", 2, QUERY + " extra words about a base requirement", clause="6.2.1"
            ),
            simple_row("38.101-1", 3, QUERY + " sidelink V2X", clause="6.2.1E"),
        ],
    )
    embedder = HashEmbedder()
    kwargs = {"mode": "vector", "top_k": 3, "spec": "38.101-1", "clause_lookup": False}
    off = search(conn, embedder, QUERY, variant_downrank=False, **kwargs)
    assert clauses_of(off)[0] == "6.2.1A"  # without the rule the variant is first
    on = search(conn, embedder, QUERY, variant_downrank=True, **kwargs)
    assert clauses_of(on)[0] == "6.2.1" and set(clauses_of(on)) == {"6.2.1", "6.2.1A", "6.2.1E"}
    # the question names carrier aggregation: 6.2.1A keeps its place, 6.2.1E is still behind
    ca = search(
        conn, embedder, QUERY, variant_downrank=True, context="with carrier aggregation", **kwargs
    )
    assert clauses_of(ca)[:2] == ["6.2.1A", "6.2.1"]
    assert clauses_of(ca)[2] == "6.2.1E"


def test_a_named_clause_or_table_is_fetched_first_and_marked_as_a_lookup(conn) -> None:
    rows = [simple_row("36.101", i, QUERY + f" filler {i}", clause=f"6.1.{i}") for i in range(1, 4)]
    target = simple_row("36.101", 9, "nothing like the query at all", clause="6.6.2.1")
    table = simple_row("36.101", 10, "another unrelated chunk", clause="7.3.1")
    table["kind"], table["table_number"] = "table", "7.3.1-1"
    load(conn, LTE, [*rows, target, table])
    embedder = HashEmbedder()
    plain = search(conn, embedder, QUERY, mode="vector", top_k=2, clause_lookup=True)
    assert "6.6.2.1" not in clauses_of(plain)  # no reference in the query: nothing is looked up
    named = search(
        conn, embedder, f"clause 6.6.2.1 {QUERY}", mode="vector", top_k=5, clause_lookup=True
    )
    assert named[0].clause == "6.6.2.1" and named[0].via == "lookup"
    assert all(h.via == "search" for h in named[1:]) and len(named) == 5
    by_table = search(
        conn, embedder, f"Table 7.3.1-1 {QUERY}", mode="vector", top_k=5, clause_lookup=True
    )
    assert by_table[0].table_number == "7.3.1-1" and by_table[0].via == "lookup"
    off = search(
        conn, embedder, f"clause 6.6.2.1 {QUERY}", mode="vector", top_k=5, clause_lookup=False
    )
    assert all(h.via == "search" for h in off)


def test_a_named_clause_is_looked_up_in_both_specs_unless_the_spec_is_known(conn) -> None:
    load(conn, LTE, [simple_row("36.101", 1, "lte text", clause="6.2.3")])
    load(conn, NR, [simple_row("38.101-1", 1, "nr text", clause="6.2.3")])
    embedder = HashEmbedder()
    both = search(conn, embedder, "clause 6.2.3 power", mode="vector", top_k=5, clause_lookup=True)
    assert {h.spec for h in both if h.via == "lookup"} == {"36.101", "38.101-1"}
    one = search(
        conn,
        embedder,
        "clause 6.2.3 power",
        mode="vector",
        top_k=5,
        spec="38.101-1",
        clause_lookup=True,
    )
    assert {h.spec for h in one} == {"38.101-1"}


def test_a_lookup_of_a_clause_that_does_not_exist_changes_nothing(conn) -> None:
    load(conn, LTE, [simple_row("36.101", 1, QUERY, clause="6.2.3")])
    embedder = HashEmbedder()
    hits = search(
        conn, embedder, f"clause 9.9.9 {QUERY}", mode="vector", top_k=5, clause_lookup=True
    )
    assert [h.via for h in hits] == ["search"] and hits[0].clause == "6.2.3"
