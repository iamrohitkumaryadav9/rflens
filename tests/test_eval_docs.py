"""eval-docs: recall@5 through docs-mcp. The server is the real docs-mcp (in process, over MCP)
with the hashing embedder and synthetic chunks in a rolled-back transaction of the test database.
"""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager

import pytest
from sqlalchemy import Connection

from rflens.db import get_engine
from rflens.eval.eval_docs import (
    CONFIGS,
    DocsEval,
    DocsMcpError,
    DocsMcpUnavailable,
    DocsSearchClient,
    group_cells,
    misses,
    rank_of_hit,
    render,
    run_eval,
)
from rflens.ingest.spec_fetch import SpecSource
from rflens.mcp.docs_server import create_docs_server
from rflens.retrieval.embedder import EmbedError, HashEmbedder
from rflens.retrieval.eval_smoke import Question
from rflens.retrieval.index import embed_pending, load_chunks

NR = SpecSource("38.101-1", "38_series", "t-nr.zip", "19.0.0", 19, "0" * 64)
LTE = SpecSource("36.101", "36_series", "t-lte.zip", "19.0.0", 19, "1" * 64)


def chunk(spec: str, i: int, body: str, clause: str, table: str | None = None) -> dict:
    return {
        "chunk_id": f"{spec}@19.0.0/{clause}/{i}",
        "spec": spec,
        "version": "19.0.0",
        "release": 19,
        "clause": clause,
        "clause_title": "Title",
        "parent_path": [],
        "parent_titles": [],
        "kind": "table" if table else "text",
        "position": i,
        "table_caption": None,
        "table_number": table,
        "table_part": None,
        "source_file": "x.md",
        "text": body,
        "raw_text": body,
        "media": [],
        "image_placeholders": 0,
        "forced_split": False,
        "char_count": len(body),
    }


@pytest.fixture
def conn() -> Iterator[Connection]:
    with get_engine().connect() as connection:
        transaction = connection.begin()
        try:
            yield connection
        finally:
            transaction.rollback()


def make_client(conn: Connection) -> DocsSearchClient:
    embedder = HashEmbedder()

    @contextmanager
    def connect() -> Iterator[Connection]:
        yield conn

    return DocsSearchClient(create_docs_server(connect, embedder, "vector"))


def load(conn: Connection, source: SpecSource, rows: list[dict]) -> None:
    ids = load_chunks(conn, source, rows, prune=False)
    embed_pending(conn, HashEmbedder(), ids, progress=lambda s: None)


def q(qid, spec, text, clauses=(), tables=(), author="human") -> Question:
    return Question(qid, spec, text, tuple(clauses), tuple(tables), author)


# --- the hit rule -----------------------------------------------------------------------------


def result(spec="38.101-1", clause="6.2.1", table=None) -> dict:
    return {"spec": spec, "clause": clause, "table_number": table}


def test_a_hit_needs_the_right_spec_and_an_exact_clause_or_a_listed_table() -> None:
    question = q("a", "38.101-1", "x", clauses=["6.2.3"], tables=["6.2.1-1"])
    assert rank_of_hit(question, [result(clause="6.2.3")]) == 1
    assert rank_of_hit(question, [result(clause="6.2.3.5")]) is None  # a sub-clause does not count
    assert rank_of_hit(question, [result(), result(clause="9", table="6.2.1-1")]) == 2
    assert rank_of_hit(question, [result(spec="36.101", clause="6.2.3")]) is None  # wrong spec
    assert rank_of_hit(question, [result(table="6.2.1-2")]) is None
    assert rank_of_hit(question, []) is None
    only_table = q("b", "36.101", "x", tables=["7.3.1-1"])
    assert rank_of_hit(only_table, [result(spec="36.101", clause="7.3.1", table="7.3.1-1")]) == 1
    assert rank_of_hit(only_table, [result(spec="36.101", clause="7.3.1")]) is None  # no table no


# --- summary and report -----------------------------------------------------------------------


def fake_eval() -> DocsEval:
    questions = [
        q("d1", "38.101-1", "first human", ["1"], author="human"),
        q("d2", "36.101", "second human", ["2"], author="human"),
        q("d3", "38.101-1", "assistant one", ["3"], author="assistant"),
    ]
    ranks = {
        "d1": {c: (1 if c[0] != "keyword" else None) for c in CONFIGS},
        "d2": {c: (2 if c[1] else None) for c in CONFIGS},  # found only with the spec filter
        "d3": {c: None for c in CONFIGS},
    }
    return DocsEval(questions, ranks, 18)


def test_group_cells_count_hits_per_author_mode_and_filter() -> None:
    ev = fake_eval()
    assert group_cells(ev, "human") == {
        "vector": (1, 2, 2),  # no filter / filter / questions
        "keyword": (0, 1, 2),
        "hybrid": (1, 2, 2),
    }
    assert group_cells(ev, "assistant") == {m: (0, 0, 1) for m in ("vector", "keyword", "hybrid")}
    assert [x.id for x in misses(ev, ("vector", False))] == ["d2", "d3"]
    assert [x.id for x in misses(ev, ("hybrid", True))] == ["d3"]


def test_render_lists_each_question_and_recall_per_group() -> None:
    text = render(fake_eval())
    assert "d1" in text and "d3" in text and "-" in text
    assert "human (2)" in text and "assistant (1)" in text
    assert "1/2 / 2/2" in text  # human vector: 1 without the filter, 2 with it
    assert "0/1 / 0/1" in text


# --- through the real docs-mcp ---------------------------------------------------------------


def test_run_eval_searches_every_mode_with_and_without_the_filter(conn: Connection) -> None:
    load(conn, NR, [chunk("38.101-1", 0, "zeta widget alpha target text", "6.2.9")])
    flood = [chunk("36.101", i, "zeta widget alpha zeta widget alpha", "6.2.9") for i in range(8)]
    load(conn, LTE, flood)  # another spec's near-identical chunks crowd the unfiltered top 5
    question = q("t1", "38.101-1", "zeta widget alpha", ["6.2.9"])
    client = make_client(conn)
    ev = run_eval(client, [question])
    assert ev.searches == 6 and client.calls == 6  # 3 modes x (no filter, filter)
    for mode in ("vector", "keyword", "hybrid"):
        assert ev.ranks["t1"][(mode, True)] == 1  # with the filter only 38.101-1 is searched
    assert ev.ranks["t1"][("keyword", False)] is None  # the flood fills the unfiltered top 5
    assert group_cells(ev, "human")["keyword"] == (0, 1, 1)


def test_a_table_target_is_found_by_its_table_number(conn: Connection) -> None:
    load(conn, NR, [chunk("38.101-1", 0, "gamma delta row data", "5.2", table="5.2-1")])
    question = q("t2", "38.101-1", "gamma delta row data", tables=["5.2-1"])
    ev = run_eval(make_client(conn), [question])
    assert ev.ranks["t2"][("keyword", True)] == 1
    wrong = q("t3", "38.101-1", "gamma delta row data", tables=["5.2-2"])
    assert run_eval(make_client(conn), [wrong]).ranks["t3"][("keyword", True)] is None


def test_an_unreachable_docs_mcp_is_a_clear_error() -> None:
    client = DocsSearchClient("http://127.0.0.1:9/mcp")  # nothing listens on port 9
    with pytest.raises(DocsMcpUnavailable, match="make docs-mcp"):
        client.search("x", None, "keyword")


def test_a_tool_error_such_as_a_down_embedder_stops_the_eval(conn: Connection) -> None:
    class Down(HashEmbedder):
        def embed_query(self, text: str) -> list[float]:
            raise EmbedError("connection refused")

    @contextmanager
    def connect() -> Iterator[Connection]:
        yield conn

    client = DocsSearchClient(create_docs_server(connect, Down(), "vector"))
    with pytest.raises(DocsMcpError, match="embedding service is unavailable"):
        client.search("anything", None, "vector")
    assert client.search("anything", None, "keyword") == []  # keyword mode needs no embedder


def test_configs_are_the_six_mode_and_filter_pairs() -> None:
    assert len(CONFIGS) == 6
    assert {m for m, _ in CONFIGS} == {"vector", "keyword", "hybrid"}
    assert isinstance(group_cells(fake_eval(), "human"), Mapping)
