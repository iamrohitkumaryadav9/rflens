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

MODES = ("vector", "keyword", "hybrid")


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
            "q": vector_literal(embedder.embed_query(query)), "model": embedder.model_id,
            "k": k, "spec": spec,
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


def search(
    conn: Connection,
    embedder: Embedder,
    query: str,
    *,
    mode: str = "hybrid",
    top_k: int = 5,
    candidates: int | None = None,
    rrf_k: int | None = None,
    spec: str | None = None,
) -> list[Hit]:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    settings = get_retrieval_settings()
    candidates = candidates or settings.search_candidates
    rrf_k = rrf_k or settings.rrf_k
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
    top = sorted(ordered.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
    if not top:
        return []
    ids = [cid for cid, _ in top]
    rows = {
        r.chunk_id: r
        for r in conn.execute(
            text(
                """
                SELECT c.chunk_id, d.spec, c.clause, c.clause_title, c.kind, c.table_number,
                       c.caption, c.table_part, c.text, c.metadata
                FROM chunks c JOIN documents d ON d.id = c.document_id
                WHERE c.chunk_id = ANY(CAST(:ids AS text[]))
                """
            ),
            {"ids": ids},
        )
    }
    return [
        Hit(
            chunk_id=cid, score=score, spec=rows[cid].spec, clause=rows[cid].clause,
            clause_title=rows[cid].clause_title, kind=rows[cid].kind,
            table_number=rows[cid].table_number, caption=rows[cid].caption,
            table_part=rows[cid].table_part, text=rows[cid].text,
            related_chunks=list(rows[cid].metadata.get("related_chunks", [])),
            vector_rank=vec_rank.get(cid), keyword_rank=kw_rank.get(cid),
        )
        for cid, score in top
    ]
