"""Load chunk JSONL into documents/chunks and embed what is not embedded yet.

Idempotent and resumable: a chunk is (re)embedded only if it has no embedding, its text hash
changed, or its stored embedding_model differs from the embedder's model id (tag + digest). Each
batch commits on its own (via `commit`), so an interrupted run resumes where it stopped.

3GPP text is copyrighted: this data lives only in the local database.
"""

import hashlib
import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import Connection, text

from rflens.ingest.spec_fetch import SpecSource
from rflens.retrieval.embedder import (
    MAX_SAFE_TOKENS,
    ContextLengthError,
    Embedder,
    EmbedError,
    vector_literal,
)

NOTES_POINTER_RE = re.compile(r"Notes for this table are in chunk (\S+?)\.(?:\s|$)")
# Rough characters per token measured on nomic-embed-text (prose, tables); only used for the ETA.
CHARS_PER_TOKEN = {"text": 4.4, "table": 2.4}


def related_chunks(chunk_text: str) -> list[str]:
    """Chunk ids a table piece points to for its notes ("Notes for this table are in chunk X.")."""
    return NOTES_POINTER_RE.findall(chunk_text)


def text_sha256(chunk_text: str) -> str:
    return hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()


def _in_clause(clause: str, prefix: str) -> bool:
    return clause == prefix or clause.startswith(prefix + ".")


def select_rows(
    chunks_path: Path, *, clause: str | None = None, limit: int | None = None
) -> list[dict]:
    """Chunks in document order, optionally only one clause (and its sub-clauses), first N."""
    rows = [json.loads(line) for line in chunks_path.read_text("utf-8").splitlines()]
    if clause is not None:
        rows = [r for r in rows if _in_clause(r["clause"], clause)]
    return rows[:limit] if limit is not None else rows


def load_chunks(
    conn: Connection, source: SpecSource, rows: list[dict], *, prune: bool
) -> list[str]:
    """Upsert the document and `rows`. With prune=True (a full load) chunks of this document that
    are no longer in the file are deleted. Returns the chunk ids in order."""
    doc_id = conn.execute(
        text(
            """
            INSERT INTO documents (spec, version, release, source_zip, sha256)
            VALUES (:spec, :version, :release, :zip, :sha)
            ON CONFLICT (spec, version) DO UPDATE SET
                release = EXCLUDED.release, source_zip = EXCLUDED.source_zip,
                sha256 = EXCLUDED.sha256
            RETURNING id
            """
        ),
        {
            "spec": source.spec, "version": source.version, "release": source.release,
            "zip": source.zip_name, "sha": source.sha256,
        },
    ).scalar_one()
    params = []
    for r in rows:
        metadata = {
            "position": r["position"], "parent_titles": r["parent_titles"],
            "source_file": r["source_file"], "media": r["media"],
            "image_placeholders": r["image_placeholders"], "forced_split": r["forced_split"],
            "char_count": r["char_count"], "related_chunks": related_chunks(r["text"]),
        }
        params.append(
            {
                "chunk_id": r["chunk_id"], "doc": doc_id, "kind": r["kind"], "clause": r["clause"],
                "title": r["clause_title"], "path": r["parent_path"],
                "table_number": r["table_number"], "caption": r["table_caption"],
                "part": r["table_part"], "text": r["text"], "raw": r["raw_text"],
                "metadata": json.dumps(metadata), "sha": text_sha256(r["text"]),
            }
        )
    if params:
        conn.execute(
            text(
                """
                INSERT INTO chunks (chunk_id, document_id, kind, clause, clause_title,
                    parent_path, table_number, caption, table_part, text, raw_text, metadata,
                    text_sha256)
                VALUES (:chunk_id, :doc, :kind, :clause, :title, :path, :table_number, :caption,
                    :part, :text, :raw, CAST(:metadata AS jsonb), :sha)
                ON CONFLICT (chunk_id) DO UPDATE SET
                    document_id = EXCLUDED.document_id, kind = EXCLUDED.kind,
                    clause = EXCLUDED.clause, clause_title = EXCLUDED.clause_title,
                    parent_path = EXCLUDED.parent_path, table_number = EXCLUDED.table_number,
                    caption = EXCLUDED.caption, table_part = EXCLUDED.table_part,
                    text = EXCLUDED.text, raw_text = EXCLUDED.raw_text,
                    metadata = CASE
                        WHEN chunks.text_sha256 = EXCLUDED.text_sha256
                             AND chunks.metadata ? 'tokens'
                        THEN EXCLUDED.metadata
                             || jsonb_build_object('tokens', chunks.metadata -> 'tokens')
                        ELSE EXCLUDED.metadata END,
                    embedding = CASE WHEN chunks.text_sha256 = EXCLUDED.text_sha256
                                     THEN chunks.embedding END,
                    embedding_model = CASE WHEN chunks.text_sha256 = EXCLUDED.text_sha256
                                           THEN chunks.embedding_model END,
                    text_sha256 = EXCLUDED.text_sha256
                """
            ),
            params,
        )
    ids = [p["chunk_id"] for p in params]
    if prune:
        conn.execute(
            text(
                "DELETE FROM chunks WHERE document_id = :doc "
                "AND chunk_id <> ALL(CAST(:ids AS text[]))"
            ),
            {"doc": doc_id, "ids": ids},
        )
    return ids


@dataclass
class EmbedStats:
    selected: int = 0
    skipped: int = 0
    embedded: int = 0
    tokens: int = 0
    seconds: float = 0.0
    # over all selected chunks, from the real token counts stored in metadata["tokens"]
    max_tokens: int = 0
    over_limit: list[tuple[str, int]] = field(default_factory=list)  # (chunk_id, tokens)

    def token_summary(self) -> str:
        worst = ", ".join(f"{cid} ({n})" for cid, n in self.over_limit[:5])
        return (
            f"max tokens per chunk: {self.max_tokens}; chunks above {MAX_SAFE_TOKENS} tokens: "
            f"{len(self.over_limit)}" + (f" [{worst}]" if worst else "")
        )


def token_summary(
    conn: Connection, chunk_ids: list[str], limit: int = MAX_SAFE_TOKENS
) -> tuple[int, list[tuple[str, int]]]:
    """(max real token count, [(chunk_id, tokens) above limit, worst first]) for the chunks."""
    rows = conn.execute(
        text(
            "SELECT chunk_id, (metadata ->> 'tokens')::int AS tokens FROM chunks "
            "WHERE chunk_id = ANY(CAST(:ids AS text[])) AND metadata ? 'tokens'"
        ),
        {"ids": chunk_ids},
    ).all()
    over = sorted(((r.chunk_id, r.tokens) for r in rows if r.tokens > limit), key=lambda x: -x[1])
    return max((r.tokens for r in rows), default=0), over


def embed_pending(
    conn: Connection,
    embedder: Embedder,
    chunk_ids: list[str],
    *,
    batch_size: int = 8,
    commit: Callable[[], None] = lambda: None,
    progress: Callable[[str], None] = print,
    clock: Callable[[], float] = time.perf_counter,
) -> EmbedStats:
    """Embed the chunks among `chunk_ids` that lack an embedding from the current model."""
    model = embedder.model_id
    pending = conn.execute(
        text(
            """
            SELECT chunk_id, text, kind FROM chunks
            WHERE chunk_id = ANY(CAST(:ids AS text[]))
              AND (embedding IS NULL OR embedding_model <> :model
                   OR NOT metadata ? 'tokens')  -- embedded before token counts were recorded
            ORDER BY array_position(CAST(:ids AS text[]), chunk_id)
            """
        ),
        {"ids": chunk_ids, "model": model},
    ).all()
    stats = EmbedStats(selected=len(chunk_ids), skipped=len(chunk_ids) - len(pending))
    remaining_tokens = sum(len(r.text) / CHARS_PER_TOKEN[r.kind] for r in pending)
    start = clock()
    reported = 0
    for i in range(0, len(pending), batch_size):
        batch = pending[i : i + batch_size]
        try:
            result = embedder.embed_documents([r.text for r in batch])
        except ContextLengthError as e:
            culprit = batch[e.index]
            raise EmbedError(
                f"chunk {culprit.chunk_id} ({len(culprit.text)} chars) exceeds the embedding "
                "model's context length; re-chunk with a smaller CHUNK_TABLE_HARD_CHARS"
            ) from e
        conn.execute(
            text(
                "UPDATE chunks SET embedding = CAST(:v AS vector), embedding_model = :model, "
                "metadata = metadata || CAST(:meta AS jsonb) WHERE chunk_id = :id"
            ),
            [
                {
                    "v": vector_literal(vec), "model": model, "id": r.chunk_id,
                    "meta": json.dumps({"tokens": n}),  # the model's real token count
                }
                for r, vec, n in zip(batch, result.vectors, result.token_counts, strict=True)
            ],
        )
        commit()
        stats.embedded += len(batch)
        stats.tokens += result.tokens
        remaining_tokens -= sum(len(r.text) / CHARS_PER_TOKEN[r.kind] for r in batch)
        stats.seconds = clock() - start
        if stats.embedded // 100 > reported:
            reported = stats.embedded // 100
            rate = stats.tokens / stats.seconds if stats.seconds else 0.0
            eta = max(remaining_tokens, 0) / rate / 60 if rate else float("nan")
            progress(
                f"  embedded {stats.embedded}/{len(pending)}  {rate:.0f} tokens/s  "
                f"ETA {eta:.1f} min"
            )
    stats.seconds = clock() - start
    stats.max_tokens, stats.over_limit = token_summary(conn, chunk_ids)
    return stats
