"""Retrieval tests: embedder protocol (mock HTTP), indexing and search against real Postgres with
a deterministic fake embedder. No Ollama and no 3GPP text (synthetic fixtures only)."""

import json
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path

import httpx
import pytest
from sqlalchemy import Connection, text

from rflens.db import get_engine
from rflens.ingest.spec_chunker import chunk_parts
from rflens.ingest.spec_fetch import SpecSource
from rflens.retrieval.embedder import (
    DOCUMENT_PREFIX,
    EMBED_DIM,
    QUERY_PREFIX,
    ContextLengthError,
    EmbedError,
    EmbedResult,
    HashEmbedder,
    OllamaEmbedder,
    vector_literal,
)
from rflens.retrieval.eval_smoke import (
    DEFAULT_FILE,
    Question,
    is_hit,
    load_questions,
    load_status,
    recall_at_k,
)
from rflens.retrieval.hybrid import Hit, keyword_search, rrf, search, vector_search
from rflens.retrieval.index import (
    embed_pending,
    load_chunks,
    related_chunks,
    select_rows,
    text_sha256,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "spec" / "synthetic"
SOURCE = SpecSource("99.101-1", "99_series", "99101-1-j00.zip", "19.0.0", 19, "0" * 64)
SPEC = "99.101-1"


def synthetic_rows(soft: int = 3000, hard: int = 6000, table_hard: int | None = None) -> list[dict]:
    parts = [(p.name, p.read_text("utf-8")) for p in sorted(FIXTURE_DIR.glob("*.md"))]
    chunks, _ = chunk_parts(parts, soft, hard, table_hard)
    return [json.loads(json.dumps(asdict(c))) for c in chunks]


def row(chunk_id: str, body: str, clause: str = "6.9", kind: str = "text") -> dict:
    """A minimal chunk row in the JSONL shape."""
    return {
        "chunk_id": chunk_id, "spec": SPEC, "version": "19.0.0", "release": 19, "clause": clause,
        "clause_title": "Title", "parent_path": [], "parent_titles": [], "kind": kind,
        "position": 0, "table_caption": None, "table_number": None, "table_part": None,
        "source_file": "x.md", "text": body, "raw_text": body, "media": [],
        "image_placeholders": 0, "forced_split": False, "char_count": len(body),
    }


@pytest.fixture
def conn() -> Iterator[Connection]:
    """Real Postgres connection; rolled back so real data is never touched."""
    with get_engine().connect() as connection:
        transaction = connection.begin()
        try:
            yield connection
        finally:
            transaction.rollback()


# --- embedder over a mock HTTP transport ---------------------------------------------------------


def vec() -> list[float]:
    return [0.5] * EMBED_DIM


def mock_embedder(handler, **kwargs) -> OllamaEmbedder:
    return OllamaEmbedder(
        "http://ollama.test", "nomic-embed-text:latest", transport=httpx.MockTransport(handler),
        sleep=kwargs.pop("sleep", lambda s: None), **kwargs,
    )


def ok_embed(request: httpx.Request) -> httpx.Response:
    n = len(json.loads(request.content)["input"])
    return httpx.Response(200, json={"embeddings": [vec()] * n, "prompt_eval_count": 10 * n})


def test_document_and_query_prefixes_and_no_truncation() -> None:
    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return ok_embed(request)

    emb = mock_embedder(handler)
    result = emb.embed_documents(["first chunk", "second chunk"])
    emb.embed_query("what is n78")
    assert DOCUMENT_PREFIX == "search_document: " and QUERY_PREFIX == "search_query: "
    # one request per chunk, so each chunk gets its own real token count
    assert [b["input"] for b in bodies] == [
        ["search_document: first chunk"],
        ["search_document: second chunk"],
        ["search_query: what is n78"],
    ]
    assert all(b["truncate"] is False for b in bodies)  # never truncate silently
    assert result.token_counts == [10, 10] and result.tokens == 20
    assert len(result.vectors[0]) == EMBED_DIM


def test_hash_embedder_uses_the_same_prefixes() -> None:
    emb = HashEmbedder()
    emb.embed_documents(["a b"])
    emb.embed_query("a")
    assert emb.inputs == ["search_document: a b", "search_query: a"]


def test_context_length_error_identifies_the_input() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if any("TOO-LONG" in s for s in json.loads(request.content)["input"]):
            error = {"error": "the input length exceeds the context length"}
            return httpx.Response(400, json=error)
        return ok_embed(request)

    emb = mock_embedder(handler)
    with pytest.raises(ContextLengthError) as err:
        emb.embed_documents(["fine", "TOO-LONG", "fine too"])
    assert err.value.index == 1


def test_transport_errors_are_retried_with_backoff_then_succeed() -> None:
    calls, sleeps = [], []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) < 3:
            return httpx.Response(503, text="busy")
        return ok_embed(request)

    emb = mock_embedder(handler, sleep=sleeps.append)
    assert len(emb.embed_documents(["x"]).vectors) == 1
    assert len(calls) == 3 and sleeps == [1, 3]


def test_three_failures_stop_with_an_error() -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        raise httpx.ConnectError("refused")

    emb = mock_embedder(handler)
    with pytest.raises(EmbedError, match="failed 3 times"):
        emb.embed_documents(["x"])
    assert len(calls) == 3


def test_wrong_vector_size_is_an_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"embeddings": [[0.1, 0.2]], "prompt_eval_count": 1})

    with pytest.raises(EmbedError, match="768"):
        mock_embedder(handler).embed_documents(["x"])


def test_model_id_is_tag_plus_digest() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        models = [{"name": "nomic-embed-text:latest", "digest": "0a109f422b47e3a30ba2" + "0" * 44}]
        return httpx.Response(200, json={"models": models})

    assert mock_embedder(handler).model_id == "nomic-embed-text:latest@0a109f422b47"

    def empty(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": []})

    with pytest.raises(EmbedError, match="not available"):
        _ = mock_embedder(empty).model_id


# --- pure helpers --------------------------------------------------------------------------------


def test_rrf_scores_and_order() -> None:
    scores = rrf({"vector": ["a", "b", "c"], "keyword": ["b", "d", "a"]}, k=60)
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["a"] == pytest.approx(1 / 61 + 1 / 63)
    assert scores["d"] == pytest.approx(1 / 62) and scores["c"] == pytest.approx(1 / 63)
    assert sorted(scores, key=scores.get, reverse=True) == ["b", "a", "d", "c"]
    assert rrf({"x": ["a"]}, k=1)["a"] == pytest.approx(0.5)  # k is configurable


def test_related_chunks_parses_the_notes_pointer() -> None:
    pointer = "Notes for this table are in chunk"
    body = f"TS 99.101-1 V19.0.0 clause 8: T\n\nrow\n\n{pointer} 99.101-1@19.0.0/8/0."
    assert related_chunks(body) == ["99.101-1@19.0.0/8/0"]
    assert related_chunks(f"{pointer} a@1/7.1~2/3. More text") == ["a@1/7.1~2/3"]
    assert related_chunks("no pointer here") == []


def test_vector_literal_is_pgvector_text() -> None:
    assert vector_literal([0.5, 1.0, -2.25]) == "[0.5,1,-2.25]"


# --- indexing against Postgres ------------------------------------------------------------------


def load_all(conn: Connection, rows: list[dict] | None = None, *, prune: bool = True) -> list[str]:
    return load_chunks(conn, SOURCE, rows or synthetic_rows(), prune=prune)


def test_load_and_embed_then_second_run_skips_everything(conn: Connection) -> None:
    ids = load_all(conn)
    emb = HashEmbedder()
    first = embed_pending(conn, emb, ids, batch_size=4, progress=lambda s: None)
    assert first.embedded == len(ids) and first.skipped == 0
    second = embed_pending(conn, emb, ids, batch_size=4, progress=lambda s: None)
    assert second.embedded == 0 and second.skipped == len(ids)
    dims = conn.execute(
        text("SELECT DISTINCT vector_dims(embedding) FROM chunks WHERE chunk_id = ANY(:ids)"),
        {"ids": ids},
    ).scalars().all()
    assert dims == [EMBED_DIM]


def test_a_different_model_counts_as_not_embedded(conn: Connection) -> None:
    ids = load_all(conn)
    embed_pending(conn, HashEmbedder(), ids, progress=lambda s: None)
    other = HashEmbedder(model_id="another-model@123")
    stats = embed_pending(conn, other, ids, progress=lambda s: None)
    assert stats.embedded == len(ids) and stats.skipped == 0
    models = conn.execute(
        text("SELECT DISTINCT embedding_model FROM chunks WHERE chunk_id = ANY(:ids)"), {"ids": ids}
    ).scalars().all()
    assert models == ["another-model@123"]


def test_changed_text_is_reembedded_and_unchanged_is_kept(conn: Connection) -> None:
    rows = synthetic_rows()
    ids = load_all(conn, rows)
    embed_pending(conn, HashEmbedder(), ids, progress=lambda s: None)
    changed = [dict(r) for r in rows]
    changed[0]["text"] += " plus a new sentence"
    load_all(conn, changed, prune=False)
    stats = embed_pending(conn, HashEmbedder(), ids, progress=lambda s: None)
    assert stats.embedded == 1 and stats.skipped == len(ids) - 1
    sha = conn.execute(
        text("SELECT text_sha256 FROM chunks WHERE chunk_id = :id"), {"id": ids[0]}
    ).scalar_one()
    assert sha == text_sha256(changed[0]["text"])


def test_full_load_prunes_removed_chunks_but_a_subset_load_does_not(conn: Connection) -> None:
    rows = synthetic_rows()
    ids = load_all(conn, rows)

    def stored() -> int:
        sql = text("SELECT count(*) FROM chunks WHERE chunk_id = ANY(:i)")
        return conn.execute(sql, {"i": ids}).scalar_one()

    load_all(conn, rows[:3], prune=False)
    assert stored() == len(ids)
    load_all(conn, rows[:3], prune=True)
    assert stored() == 3


def test_select_rows_clause_and_limit(tmp_path: Path) -> None:
    path = tmp_path / "chunks.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in synthetic_rows()), encoding="utf-8")
    only6 = select_rows(path, clause="6")
    assert only6 and all(r["clause"] == "6" or r["clause"].startswith("6.") for r in only6)
    assert not any(r["clause"].startswith("7") for r in only6)
    assert len(select_rows(path, clause="6", limit=2)) == 2
    assert [r["chunk_id"] for r in select_rows(path, limit=3)] == [
        r["chunk_id"] for r in synthetic_rows()[:3]
    ]


def test_context_error_names_the_chunk_and_keeps_earlier_batches(conn: Connection) -> None:
    ids = load_all(conn)

    class TooLong(HashEmbedder):
        def embed_documents(self, texts: list[str]) -> EmbedResult:
            if len(self.inputs) >= 2:  # second batch fails on its first input
                raise ContextLengthError(0)
            return super().embed_documents(texts)

    commits = []
    with pytest.raises(EmbedError, match="exceeds the embedding model's context") as err:
        embed_pending(
            conn, TooLong(), ids, batch_size=2, commit=lambda: commits.append(1),
            progress=lambda s: None,
        )
    assert ids[2] in str(err.value)  # the third chunk is the first of the failing batch
    assert len(commits) == 1  # the first batch was committed before the failure
    done = conn.execute(
        text("SELECT count(embedding) FROM chunks WHERE chunk_id = ANY(:i)"), {"i": ids}
    ).scalar_one()
    assert done == 2


def stored_tokens(conn: Connection, chunk_id: str) -> int | None:
    return conn.execute(
        text("SELECT (metadata ->> 'tokens')::int FROM chunks WHERE chunk_id = :id"),
        {"id": chunk_id},
    ).scalar_one()


def test_real_token_counts_are_stored_and_survive_an_unchanged_reload(conn: Connection) -> None:
    rows = [row(f"{SPEC}@19.0.0/6.9/{i}", " ".join(["word"] * (10 + i))) for i in range(3)]
    ids = load_all(conn, rows)
    assert all(stored_tokens(conn, i) is None for i in ids)
    embed_pending(conn, HashEmbedder(), ids, progress=lambda s: None)
    assert [stored_tokens(conn, i) for i in ids] == [10, 11, 12]  # HashEmbedder: word counts
    load_all(conn, rows, prune=False)  # same text: embedding and token count are kept
    assert [stored_tokens(conn, i) for i in ids] == [10, 11, 12]
    assert embed_pending(conn, HashEmbedder(), ids, progress=lambda s: None).embedded == 0
    changed = [dict(r) for r in rows]
    changed[1]["text"] += " extra"
    load_all(conn, changed, prune=False)  # changed text: its token count is dropped
    assert stored_tokens(conn, ids[1]) is None and stored_tokens(conn, ids[0]) == 10


def test_chunks_embedded_without_a_token_count_are_embedded_again(conn: Connection) -> None:
    ids = load_all(conn, [row(f"{SPEC}@19.0.0/6.9/0", "some words here")])
    embed_pending(conn, HashEmbedder(), ids, progress=lambda s: None)
    conn.execute(
        text("UPDATE chunks SET metadata = metadata - 'tokens' WHERE chunk_id = :i"),
        {"i": ids[0]},
    )
    stats = embed_pending(conn, HashEmbedder(), ids, progress=lambda s: None)
    assert stats.embedded == 1 and stored_tokens(conn, ids[0]) == 3


def test_embed_stats_report_max_tokens_and_chunks_over_the_limit(conn: Connection) -> None:
    rows = [
        row(f"{SPEC}@19.0.0/6.9/0", "short chunk"),
        row(f"{SPEC}@19.0.0/6.9/1", " ".join(["word"] * 1900)),  # HashEmbedder counts words
        row(f"{SPEC}@19.0.0/6.9/2", " ".join(["word"] * 1841)),
        row(f"{SPEC}@19.0.0/6.9/3", " ".join(["word"] * 1840)),  # exactly at the limit is fine
    ]
    ids = load_all(conn, rows)
    stats = embed_pending(conn, HashEmbedder(), ids, progress=lambda s: None)
    assert stats.max_tokens == 1900
    assert stats.over_limit == [(ids[1], 1900), (ids[2], 1841)]
    assert "max tokens per chunk: 1900" in stats.token_summary()
    assert "chunks above 1840 tokens: 2" in stats.token_summary()
    again = embed_pending(conn, HashEmbedder(), ids, progress=lambda s: None)  # nothing embedded
    assert again.embedded == 0 and again.max_tokens == 1900 and len(again.over_limit) == 2


def test_progress_is_reported_every_100_chunks(conn: Connection) -> None:
    rows = [row(f"{SPEC}@19.0.0/6.9/{i}", f"widget text number {i} power") for i in range(230)]
    ids = load_all(conn, rows)
    lines: list[str] = []
    embed_pending(conn, HashEmbedder(), ids, batch_size=10, progress=lines.append)
    assert len(lines) == 2 and "embedded 100/230" in lines[0] and "ETA" in lines[0]
    assert "tokens/s" in lines[1]


# --- generated tsvector and search --------------------------------------------------------------


def test_fts_normalisation_splits_units_and_keeps_identifiers(conn: Connection) -> None:
    sql = text("SELECT rflens_fts_norm('23dBm and dBm/15kHz at n78 6.2.1 NS_04')")
    assert conn.execute(sql).scalar_one() == "23 dBm and dBm 15 kHz at n78 6.2.1 NS_04"
    body = "limit is -30dBm/1MHz and 23dBm for NS_04 in n78"
    ids = load_all(conn, [row(f"{SPEC}@19.0.0/6.9/0", body)])
    assert ids
    for term in ("dBm", "MHz", "NS_04", "n78", "30 dBm"):
        hits = keyword_search(conn, term, k=5, spec=SPEC)
        assert [h[0] for h in hits] == [ids[0]], term


def test_keyword_query_ors_the_terms_and_ranks_by_coverage(conn: Connection) -> None:
    rows = [
        row(f"{SPEC}@19.0.0/6.9/0", "maximum output power for class three devices"),
        row(f"{SPEC}@19.0.0/6.9/1", "maximum output only"),
        row(f"{SPEC}@19.0.0/6.9/2", "unrelated text about antennas"),
    ]
    load_all(conn, rows)
    # 'gizmo' matches nothing; OR semantics still return the chunks with the other terms
    hits = keyword_search(conn, "maximum output power gizmo", k=10, spec=SPEC)
    assert [h[0] for h in hits] == [rows[0]["chunk_id"], rows[1]["chunk_id"]]
    assert hits[0][1] > hits[1][1]


def test_stemming_matches_word_variants(conn: Connection) -> None:
    load_all(conn, [row(f"{SPEC}@19.0.0/6.9/0", "the transmitter requirements are measured")])
    assert keyword_search(conn, "requirement measure", k=5, spec=SPEC)


def test_hybrid_search_finds_the_table_in_all_modes(conn: Connection) -> None:
    ids = load_all(conn)
    emb = HashEmbedder()
    embed_pending(conn, emb, ids, progress=lambda s: None)
    query = "Class A power dBm tolerance band x2"
    for mode in ("vector", "keyword", "hybrid"):
        hits = search(conn, emb, query, mode=mode, top_k=5, spec=SPEC)
        assert any(h.table_number == "6.2.1-1" for h in hits), mode
    hybrid = search(conn, emb, query, mode="hybrid", top_k=5, spec=SPEC)
    top = hybrid[0]
    assert top.table_number == "6.2.1-1"
    assert top.vector_rank is not None and top.keyword_rank is not None
    assert top.score == pytest.approx(1 / (60 + top.vector_rank) + 1 / (60 + top.keyword_rank))
    # RRF k is configurable
    assert search(conn, emb, query, mode="hybrid", top_k=1, rrf_k=1, spec=SPEC)[0].score > top.score


def test_default_search_mode_is_vector_and_comes_from_config(
    conn: Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    ids = load_all(conn)
    emb = HashEmbedder()
    embed_pending(conn, emb, ids, progress=lambda s: None)
    monkeypatch.delenv("RETRIEVAL_MODE", raising=False)
    default = search(conn, emb, "widget x2", top_k=3, spec=SPEC)  # no mode given
    vector = search(conn, emb, "widget x2", mode="vector", top_k=3, spec=SPEC)
    assert [h.chunk_id for h in default] == [h.chunk_id for h in vector]
    assert default[0].vector_rank == 1 and default[0].keyword_rank is None  # only the vector leg
    monkeypatch.setenv("RETRIEVAL_MODE", "keyword")
    kw = search(conn, emb, "widget x2", top_k=3, spec=SPEC)
    assert kw[0].vector_rank is None and kw[0].keyword_rank == 1
    monkeypatch.setenv("RETRIEVAL_MODE", "hybrid")
    assert search(conn, emb, "widget x2", top_k=3, spec=SPEC)[0].keyword_rank is not None
    monkeypatch.setenv("RETRIEVAL_MODE", "nonsense")
    with pytest.raises(ValueError):
        search(conn, emb, "widget x2", spec=SPEC)


def test_vector_search_ignores_chunks_from_other_models(conn: Connection) -> None:
    ids = load_all(conn)
    embed_pending(conn, HashEmbedder(model_id="old@1"), ids, progress=lambda s: None)
    assert vector_search(conn, HashEmbedder(model_id="new@2"), "power class", k=5, spec=SPEC) == []
    assert vector_search(conn, HashEmbedder(model_id="old@1"), "power class", k=5, spec=SPEC)


def test_split_table_hit_carries_the_notes_chunk_in_related_chunks(conn: Connection) -> None:
    rows_html = "\n".join(
        f"<tr>\n<td>row-{i:03d}</td>\n<td>{i * 3}</td>\n</tr>" for i in range(120)
    )
    notes = "\n\n".join(f"NOTE {n}: " + "explanatory words " * 14 for n in range(1, 4))
    md = (
        "# 8 Big table clause\n\nTable 8-1: Big table\n\n<table>\n<thead>\n<tr>\n<th>Name</th>\n"
        f"<th>Value</th>\n</tr>\n</thead>\n<tbody>\n{rows_html}\n</tbody>\n</table>\n\n{notes}\n"
    )
    cover = (FIXTURE_DIR / "00_cover.md").name, (FIXTURE_DIR / "00_cover.md").read_text("utf-8")
    chunks, _ = chunk_parts([cover, ("01_big.md", md)], 400, 800)
    rows = [json.loads(json.dumps(asdict(c))) for c in chunks]
    notes_chunk = next(r for r in rows if r["table_part"] == "notes")
    ids = load_all(conn, rows)
    emb = HashEmbedder()
    embed_pending(conn, emb, ids, progress=lambda s: None)
    hits = search(conn, emb, "row-050 Big table Name Value", mode="keyword", top_k=3, spec=SPEC)
    assert hits and hits[0].related_chunks == [notes_chunk["chunk_id"]]
    assert hits[0].table_part != "notes"
    found = search(conn, emb, "explanatory words NOTE", mode="keyword", spec=SPEC)
    notes_hit = [h for h in found if h.table_part == "notes"]
    assert notes_hit and notes_hit[0].related_chunks == []


# --- smoke eval ---------------------------------------------------------------------------------


def test_smoke_eval_file_is_reviewed_and_well_formed() -> None:
    assert load_status(DEFAULT_FILE) == "reviewed-labels"
    questions = load_questions(DEFAULT_FILE)
    assert len(questions) == 10
    assert len({q.id for q in questions}) == 10
    assert {q.spec for q in questions} <= {"36.101", "38.101-1"}
    assert all(q.expected_clauses or q.expected_tables for q in questions)
    q07 = next(q for q in questions if q.id == "q07")
    assert "n78" in q07.question and "20 MHz" in q07.question
    assert q07.expected_clauses == ("7.3.2",)


def test_smoke_eval_hit_rule_is_exact_clause_not_prefix() -> None:
    q = Question("a", SPEC, "x", ("6.2.3",), ())
    hit = Hit(
        chunk_id="c", score=1.0, spec=SPEC, version="19.0.0", clause="6.2.3.5",
        clause_title=None, kind="text",
        table_number=None, caption=None, table_part=None, text="", related_chunks=[],
    )
    assert not is_hit(q, hit)  # a sub-clause does not count for its parent clause
    assert is_hit(Question("a", SPEC, "x", ("6.2.3.5",), ()), hit)
    assert not is_hit(Question("a", "other", "x", ("6.2.3.5",), ()), hit)  # the spec must match


def test_smoke_eval_spec_filter_keeps_another_spec_from_displacing_the_answer(
    conn: Connection,
) -> None:
    ids = load_all(conn)
    other = SpecSource("99.102", "99_series", "other.zip", "19.0.0", 19, "1" * 64)
    flood = [
        row(f"99.102@19.0.0/6.9/{i}", f"widget x2 widget x2 widget x2 flood{i}") for i in range(6)
    ]
    flood_ids = load_chunks(conn, other, flood, prune=True)
    emb = HashEmbedder()
    embed_pending(conn, emb, ids + flood_ids, progress=lambda s: None)
    question = Question("a", SPEC, "widget x2", ("6.2.9",), ("6.2.1-1",))
    unfiltered = recall_at_k(conn, emb, [question], k=5)
    filtered = recall_at_k(conn, emb, [question], k=5, filter_spec=True)
    for mode in ("vector", "keyword", "hybrid"):
        assert unfiltered[mode][0][1] is None, mode  # the other spec fills the top 5
        assert filtered[mode][0][1] is not None, mode  # only the question's own spec is searched


def test_smoke_eval_hit_rule_and_recall(conn: Connection) -> None:
    ids = load_all(conn)
    emb = HashEmbedder()
    embed_pending(conn, emb, ids, progress=lambda s: None)
    # distinctive synthetic words, so real chunks that may be in the database cannot interfere
    q_ok = Question("a", SPEC, "widget x2", ("6.2.9",), ("6.2.1-1",))
    q_wrong_spec = Question("b", "00.000", "widget x2", (), ("6.2.1-1",))
    results = recall_at_k(conn, emb, [q_ok, q_wrong_spec], k=5)
    assert set(results) == {"vector", "keyword", "hybrid"}
    for mode in results:
        assert results[mode][0][1] is not None  # found by table number
        assert results[mode][1][1] is None  # right chunk, wrong spec is not a hit
    hits = search(conn, emb, q_ok.question, top_k=5, spec=SPEC)
    assert any(is_hit(q_ok, h) for h in hits)
    assert not any(is_hit(q_wrong_spec, h) for h in hits)
