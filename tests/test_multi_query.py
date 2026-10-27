"""Multi-query retrieval: the doc agent searches with the planner's doc_query AND the user's
question, merged by reciprocal rank fusion before the variant rule and the cut (retrieval/hybrid.py
`also_query`), and the free measurement of the three ways (eval/eval_docs.py `--queries-from`).

Real test database, hashing embedder, synthetic chunks in a rolled-back transaction."""

# ruff: noqa: F811
# (the conn fixture imported from test_docs_mcp is used by name)

from rflens.config import GraphSettings
from rflens.eval.eval_docs import (
    SPEC_SETTINGS,
    WAYS,
    DocsSearchClient,
    QueryInput,
    merged_verdict,
    render_ways,
    run_ways_eval,
    ways_table,
)
from rflens.graph.docs_tools import DocsMcpTools
from rflens.retrieval.embedder import HashEmbedder
from rflens.retrieval.eval_smoke import Question
from rflens.retrieval.hybrid import search
from rflens.retrieval.index import embed_pending, load_chunks
from test_docs_mcp import LTE, NR, Env, conn, data, simple_row  # noqa: F401

Q1 = "alpha bravo charlie"
Q2 = "delta echo foxtrot"


def load(conn, source, rows) -> None:
    ids = load_chunks(conn, source, rows, prune=False)
    embed_pending(conn, HashEmbedder(), ids, progress=lambda s: None)


def clauses(hits) -> list[str]:
    return [h.clause for h in hits]


def two_topics(conn) -> None:
    load(
        conn,
        NR,
        [
            simple_row("38.101-1", 1, Q1, clause="6.1.1"),
            simple_row("38.101-1", 2, Q2, clause="6.1.2"),
            *[
                simple_row("38.101-1", 10 + i, f"unrelated filler words number {i}", clause="6.3.1")
                for i in range(6)
            ],
        ],
    )


def test_both_queries_contribute_to_the_merged_top(conn) -> None:
    two_topics(conn)
    embedder = HashEmbedder()
    kwargs = {"mode": "vector", "spec": "38.101-1", "clause_lookup": False}
    only_first = search(conn, embedder, Q1, top_k=1, **kwargs)
    assert clauses(only_first) == ["6.1.1"]
    merged = search(conn, embedder, Q1, top_k=2, also_query=Q2, **kwargs)
    assert set(clauses(merged)) == {"6.1.1", "6.1.2"}  # each is rank 1 of its own list
    reverse = search(conn, embedder, Q2, top_k=2, also_query=Q1, **kwargs)
    assert set(clauses(reverse)) == {"6.1.1", "6.1.2"}


def test_a_second_query_equal_to_the_first_or_empty_changes_nothing(conn) -> None:
    two_topics(conn)
    embedder = HashEmbedder()
    kwargs = {"mode": "vector", "spec": "38.101-1", "top_k": 4, "clause_lookup": False}
    plain = search(conn, embedder, Q1, **kwargs)
    for same in (Q1, f"  {Q1} ", "", "   ", None):
        again = search(conn, embedder, Q1, also_query=same, **kwargs)
        assert [(h.chunk_id, round(h.score, 9)) for h in again] == [
            (h.chunk_id, round(h.score, 9)) for h in plain
        ]


def test_a_chunk_found_by_both_queries_appears_once_and_the_cut_applies(conn) -> None:
    both = f"{Q1} {Q2}"
    load(
        conn,
        NR,
        [
            simple_row("38.101-1", 1, both, clause="6.1.1"),
            simple_row("38.101-1", 2, Q1, clause="6.1.2"),
            simple_row("38.101-1", 3, Q2, clause="6.1.3"),
        ],
    )
    hits = search(
        conn,
        HashEmbedder(),
        Q1,
        mode="vector",
        top_k=3,
        spec="38.101-1",
        also_query=Q2,
        clause_lookup=False,
    )
    assert len(hits) == len({h.chunk_id for h in hits}) == 3
    top1 = search(
        conn, HashEmbedder(), Q1, mode="vector", top_k=1, also_query=Q2, clause_lookup=False
    )
    assert len(top1) == 1


def test_the_variant_rule_acts_on_the_merged_list(conn) -> None:
    load(
        conn,
        NR,
        [
            simple_row("38.101-1", 1, Q1, clause="6.2.1A"),  # nearest to Q1, but a CA variant
            simple_row("38.101-1", 2, Q2, clause="6.2.1"),
            simple_row("38.101-1", 3, Q2 + " words", clause="6.2.2"),
        ],
    )
    embedder = HashEmbedder()
    kwargs = {
        "mode": "vector",
        "top_k": 3,
        "spec": "38.101-1",
        "also_query": Q2,
        "clause_lookup": False,
    }
    off = search(conn, embedder, Q1, variant_downrank=False, **kwargs)
    assert set(clauses(off)) == {"6.2.1A", "6.2.1", "6.2.2"}
    on = search(conn, embedder, Q1, variant_downrank=True, **kwargs)
    assert clauses(on)[-1] == "6.2.1A"  # the variant moved behind the base clauses
    ca = search(
        conn, embedder, Q1, variant_downrank=True, context="with carrier aggregation", **kwargs
    )
    assert "6.2.1A" in clauses(ca)[:2]  # the question's topic excuses it, as before


def test_clause_lookups_stay_first_and_a_reference_in_either_query_counts(conn) -> None:
    two_topics(conn)
    load(conn, NR, [simple_row("38.101-1", 99, "nothing alike here at all", clause="6.6.2.1")])
    embedder = HashEmbedder()
    in_first = search(
        conn,
        embedder,
        f"clause 6.6.2.1 {Q1}",
        mode="vector",
        top_k=3,
        spec="38.101-1",
        also_query=Q2,
        clause_lookup=True,
    )
    assert in_first[0].clause == "6.6.2.1" and in_first[0].via == "lookup"
    in_second = search(
        conn,
        embedder,
        Q1,
        mode="vector",
        top_k=3,
        spec="38.101-1",
        also_query=f"what does clause 6.6.2.1 say about {Q2}",
        clause_lookup=True,
    )
    assert in_second[0].clause == "6.6.2.1" and in_second[0].via == "lookup"
    assert all(h.via == "search" for h in in_second[1:])


def test_keyword_and_hybrid_modes_merge_too(conn) -> None:
    two_topics(conn)
    for mode in ("keyword", "hybrid"):
        merged = search(
            conn,
            HashEmbedder(),
            Q1,
            mode=mode,
            top_k=2,
            spec="38.101-1",
            also_query=Q2,
            clause_lookup=False,
        )
        assert set(clauses(merged)) >= {"6.1.1", "6.1.2"}, mode


# --- docs-mcp and the doc agent ------------------------------------------------------------


def test_docs_mcp_takes_also_query_and_caps_its_length(conn) -> None:
    env = Env(conn)
    env.load(
        NR,
        [simple_row("38.101-1", 1, Q1, clause="6.1.1"), simple_row("38.101-1", 2, Q2, "6.1.2")],
    )
    one = data(env.call("search_docs", {"query": Q1, "k": 1, "mode": "vector"}))
    assert [r["clause"] for r in one["results"]] == ["6.1.1"]
    two = data(env.call("search_docs", {"query": Q1, "also_query": Q2, "k": 2, "mode": "vector"}))
    assert {r["clause"] for r in two["results"]} == {"6.1.1", "6.1.2"}
    long = data(
        env.call("search_docs", {"query": Q1, "also_query": "x " * 600, "k": 2, "mode": "vector"})
    )
    assert len(long["results"]) == 2  # cut to the query limit, not an error


def test_the_doc_agent_searches_with_both_queries_unless_switched_off(conn) -> None:
    from rflens.graph import nodes

    calls: list[dict] = []

    class Recorder(DocsMcpTools):
        def __init__(self):
            super().__init__(target=None)

        def search(self, query, spec, mode, k, context=None, also_query=None):
            calls.append(
                {"query": query, "spec": spec, "context": context, "also_query": also_query}
            )
            return []

    for multi, expected in ((True, "the user question"), (False, None)):
        calls.clear()
        deps = nodes.GraphDeps(
            run=type(
                "R",
                (),
                {
                    "guard": type("G", (), {"max_calls_per_question": 0})(),
                    "seed_question": lambda *_: None,
                },
            )(),
            sql_tools=None,
            docs_tools=Recorder(),
            settings=GraphSettings(_env_file=None, orch_multi_query=multi),
        )
        agent = nodes.make_doc_agent(deps)
        out = agent(
            {
                "question_id": "q",
                "question": "the user question",
                "plan": {"doc_query": "the planner rewrite"},
                "spec": "38.101-1",
            }
        )
        assert out["halt"].startswith("no specification passages")
        (call,) = calls
        assert call["query"].startswith("the planner rewrite") and call["spec"] == "38.101-1"
        assert call["context"] == "the user question" and call["also_query"] == expected


def test_multi_query_is_on_by_default() -> None:
    assert GraphSettings(_env_file=None).orch_multi_query is True


# --- the free measurement of the three ways ------------------------------------------------


def question(qid, author, expected="6.1.1") -> Question:
    return Question(
        id=qid,
        spec="38.101-1",
        author=author,
        question="Q?",
        expected_clauses=[expected],
        expected_tables=[],
    )


class FakeClient(DocsSearchClient):
    """Answers by the query text, so each way can be given a hit or a miss."""

    def __init__(self, hits_by_query: dict[str, list[str]]):
        super().__init__(target=None)
        self.hits_by_query = hits_by_query
        self.log: list[tuple] = []

    def search(self, query, spec, mode, k=5, context=None, also_query=None):
        self.calls += 1
        self.log.append((query, spec, context, also_query))
        key = f"{query}|{also_query}" if also_query else query
        wanted = self.hits_by_query.get(key, [])
        return [{"spec": "38.101-1", "clause": c, "table_number": None} for c in wanted]


def test_the_three_ways_use_the_right_arguments_and_the_verdict_needs_both_groups() -> None:
    qs = [question("doc-01", "human"), question("doc-11", "assistant")]
    inputs = {
        "doc-01": QueryInput("rewrite one", "38.101-1"),
        "doc-11": QueryInput("rewrite two", None),
    }
    client = FakeClient(
        {
            "rewrite one": ["6.1.1"],  # the planner's wording hits for the human question
            "Q?": ["6.1.1"],  # the question alone hits for both
            "rewrite two": ["9.9"],  # ... and misses for the assistant question
            "rewrite one|Q?": ["6.1.1"],
            "rewrite two|Q?": ["6.1.1"],  # merged rescues the assistant question
        }
    )
    ranks = run_ways_eval(client, qs, inputs)
    assert client.calls == 2 * len(WAYS) * len(SPEC_SETTINGS)
    assert ranks["doc-11"][("doc_query", "agent spec")] is None
    assert ranks["doc-11"][("both", "agent spec")] == 1
    # arguments: the doc_query way passes the question as context, the question way passes nothing
    assert ("rewrite one", "38.101-1", "Q?", None) in client.log
    assert ("Q?", "38.101-1", None, None) in client.log
    assert ("rewrite one", "38.101-1", "Q?", "Q?") in client.log
    assert ("rewrite two", None, "Q?", "Q?") in client.log  # the agent had no spec for this one
    table = ways_table(qs, ranks)
    assert table[("both", "agent spec")] == {"human": (1, 1), "assistant": (1, 1)}
    assert table[("doc_query", "agent spec")] == {"human": (1, 1), "assistant": (0, 1)}
    assert merged_verdict(table, "agent spec") is True
    text = render_ways("eval-e2e-x", qs, ranks)
    assert "merged >= better single query for both groups (agent spec): YES" in text
    assert "doc-11" in text and "assistant" in text


def test_the_verdict_is_no_when_merging_loses_in_one_group() -> None:
    qs = [question("doc-01", "human"), question("doc-11", "assistant")]
    inputs = {i: QueryInput("rewrite", "38.101-1") for i in ("doc-01", "doc-11")}
    client = FakeClient({"rewrite": ["6.1.1"], "rewrite|Q?": []})  # merging hurts both groups
    ranks = run_ways_eval(client, qs, inputs)
    table = ways_table(qs, ranks)
    assert merged_verdict(table, "agent spec") is False
    assert "(agent spec): NO" in render_ways("r", qs, ranks)


def test_a_question_without_a_saved_doc_query_is_left_out() -> None:
    qs = [question("doc-01", "human"), question("doc-02", "human")]
    ranks = run_ways_eval(FakeClient({}), qs, {"doc-01": QueryInput("x", None)})
    assert list(ranks) == ["doc-01"]
