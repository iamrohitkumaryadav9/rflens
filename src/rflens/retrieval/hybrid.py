"""Hybrid search over chunks: vector top-k and full-text top-k merged by reciprocal rank fusion.

- vector: cosine similarity on the pgvector HNSW index, only chunks embedded by the embedder's
  current model (tag + digest).
- keyword: the query's significant terms OR-ed together (plainto_tsquery drops stopwords and
  stems; its AND is turned into OR), ranked with ts_rank_cd. The query goes through the same
  rflens_fts_norm() as the stored text, so "23dBm" and "dBm/15kHz" match "dBm".
- fusion: score(chunk) = sum over legs of 1 / (k + rank), rank starting at 1, k=60 by default.

No reranker yet. 3GPP text is copyrighted; results are for local use.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from sqlalchemy import Connection, text

from rflens.config import get_retrieval_settings
from rflens.retrieval.embedder import Embedder, vector_literal
from rflens.retrieval.variants import extract_refs, is_downranked

MODES = ("vector", "keyword", "hybrid")
LOOKUP_CHUNKS = 3  # chunks fetched per named clause or table
POOL_FACTOR = 10  # the variant rule re-orders the top k x 10 candidates


def rrf(rankings: Mapping[str, Sequence[str]], k: int = 60) -> dict[str, float]:
    """Reciprocal rank fusion over named best-first id lists."""
    scores: dict[str, float] = {}
    for ranked in rankings.values():
        for rank, chunk_id in enumerate(ranked, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return scores


@dataclass
class Hit:
    chunk_id: str
    score: float  # cosine similarity, ts_rank_cd, or the RRF score, depending on the mode
    spec: str
    version: str
    clause: str
    clause_title: str | None
    kind: str
    table_number: str | None
    caption: str | None
    table_part: str | None
    text: str
    related_chunks: list[str]  # e.g. the notes chunk a split table points to
    vector_rank: int | None = None
    keyword_rank: int | None = None
    via: str = "search"  # 'lookup' for a chunk fetched because the question names its clause


def _spec_filter(spec: str | None) -> tuple[str, str]:
    if spec is None:
        return "", ""
    return "JOIN documents d ON d.id = c.document_id", "AND d.spec = :spec"


def vector_search(
    conn: Connection, embedder: Embedder, query: str, *, k: int, spec: str | None = None
) -> list[tuple[str, float]]:
    join, where = _spec_filter(spec)
    # HNSW returns at most hnsw.ef_search rows (default 40); keep it above the requested k
    conn.execute(text(f"SET LOCAL hnsw.ef_search = {min(1000, max(100, 2 * k))}"))
    rows = conn.execute(
        text(
            f"""
            SELECT c.chunk_id, 1 - (c.embedding <=> CAST(:q AS vector)) AS score
            FROM chunks c {join}
            WHERE c.embedding IS NOT NULL AND c.embedding_model = :model {where}
            ORDER BY c.embedding <=> CAST(:q AS vector), c.chunk_id
            LIMIT :k
            """
        ),
        {
            "q": vector_literal(embedder.embed_query(query)),
            "model": embedder.model_id,
            "k": k,
            "spec": spec,
        },
    ).all()
    return [(r.chunk_id, float(r.score)) for r in rows]


def keyword_search(
    conn: Connection, query: str, *, k: int, spec: str | None = None
) -> list[tuple[str, float]]:
    join, where = _spec_filter(spec)
    rows = conn.execute(
        text(
            f"""
            WITH q AS (
                SELECT replace(
                    plainto_tsquery('english', rflens_fts_norm(:text))::text, ' & ', ' | '
                )::tsquery AS tsq
            )
            SELECT c.chunk_id, ts_rank_cd(c.tsv, q.tsq) AS score
            FROM chunks c {join}, q
            WHERE c.tsv @@ q.tsq {where}
            ORDER BY score DESC, c.chunk_id
            LIMIT :k
            """
        ),
        {"text": query, "k": k, "spec": spec},
    ).all()
    return [(r.chunk_id, float(r.score)) for r in rows]


_COLUMNS = """c.chunk_id, d.spec, d.version, c.clause, c.clause_title, c.kind,
                       c.table_number, c.caption, c.table_part, c.text, c.metadata"""


def _hit(row, score: float, via: str = "search", vec_rank=None, kw_rank=None) -> Hit:
    return Hit(
        chunk_id=row.chunk_id,
        score=score,
        spec=row.spec,
        version=row.version,
        clause=row.clause,
        clause_title=row.clause_title,
        kind=row.kind,
        table_number=row.table_number,
        caption=row.caption,
        table_part=row.table_part,
        text=row.text,
        related_chunks=list(row.metadata.get("related_chunks", [])),
        vector_rank=vec_rank,
        keyword_rank=kw_rank,
        via=via,
    )


def lookup_refs(
    conn: Connection, clauses: list[str], tables: list[str], spec: str | None
) -> list[Hit]:
    """The chunks of the named clauses and tables, first LOOKUP_CHUNKS of each in document order;
    one spec, or both when `spec` is None."""
    hits: list[Hit] = []
    seen: set[str] = set()
    spec_where = "AND d.spec = :spec" if spec else ""
    for column, values in (("c.clause", clauses), ("c.table_number", tables)):
        for value in values:
            rows = conn.execute(
                text(
                    f"""
                    SELECT {_COLUMNS} FROM chunks c JOIN documents d ON d.id = c.document_id
                    WHERE {column} = :v {spec_where}
                    ORDER BY d.spec, (c.metadata->>'position')::int, c.chunk_id LIMIT :n
                    """
                ),
                {"v": value, "spec": spec, "n": LOOKUP_CHUNKS},
            ).all()
            for row in rows:
                if row.chunk_id not in seen:
                    seen.add(row.chunk_id)
                    hits.append(_hit(row, 1.0, via="lookup"))
    return hits


def search(
    conn: Connection,
    embedder: Embedder,
    query: str,
    *,
    mode: str | None = None,
    top_k: int = 5,
    candidates: int | None = None,
    rrf_k: int | None = None,
    spec: str | None = None,
    context: str | None = None,
    clause_lookup: bool | None = None,
    variant_downrank: bool | None = None,
) -> list[Hit]:
    """Search the chunks. `mode` defaults to the configured RETRIEVAL_MODE (interim: vector).

    `clause_lookup` and `variant_downrank` default to the CLAUSE_LOOKUP / VARIANT_DOWNRANK
    settings. `context` is the text in which the variant rule looks for topic words (the question
    itself when the query is a rewrite of it); it defaults to the query.
    """
    settings = get_retrieval_settings()
    mode = mode or settings.retrieval_mode
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    candidates = candidates or settings.search_candidates
    rrf_k = rrf_k or settings.rrf_k
    lookup_on = settings.clause_lookup if clause_lookup is None else clause_lookup
    rerank_on = settings.variant_downrank if variant_downrank is None else variant_downrank
    vec = vector_search(conn, embedder, query, k=candidates, spec=spec) if mode != "keyword" else []
    kw = keyword_search(conn, query, k=candidates, spec=spec) if mode != "vector" else []
    vec_rank = {cid: i for i, (cid, _) in enumerate(vec, start=1)}
    kw_rank = {cid: i for i, (cid, _) in enumerate(kw, start=1)}
    if mode == "vector":
        ordered = dict(vec)
    elif mode == "keyword":
        ordered = dict(kw)
    else:
        ordered = rrf({"vector": [c for c, _ in vec], "keyword": [c for c, _ in kw]}, rrf_k)
    pool = top_k * POOL_FACTOR if rerank_on else top_k
    top = sorted(ordered.items(), key=lambda kv: (-kv[1], kv[0]))[:pool]
    hits: list[Hit] = []
    if top:
        ids = [cid for cid, _ in top]
        rows = {
            r.chunk_id: r
            for r in conn.execute(
                text(
                    f"""
                    SELECT {_COLUMNS} FROM chunks c JOIN documents d ON d.id = c.document_id
                    WHERE c.chunk_id = ANY(CAST(:ids AS text[]))
                    """
                ),
                {"ids": ids},
            )
        }
        hits = [
            _hit(rows[cid], score, vec_rank=vec_rank.get(cid), kw_rank=kw_rank.get(cid))
            for cid, score in top
        ]
    if rerank_on:  # stable: variant clauses whose topic is not mentioned move behind the rest
        topic_text = context or query
        keep = [h for h in hits if not is_downranked(h.spec, h.clause, topic_text)]
        later = [h for h in hits if is_downranked(h.spec, h.clause, topic_text)]
        hits = keep + later
    if lookup_on:
        clauses, tables = extract_refs(query if context is None else f"{query}\n{context}")
        if clauses or tables:
            looked = lookup_refs(conn, clauses, tables, spec)
            names = {h.chunk_id for h in looked}
            hits = looked + [h for h in hits if h.chunk_id not in names]
    return hits[:top_k]
