"""CLI: load 3GPP chunks into Postgres and embed them (make embed-3gpp).

    python -m rflens.retrieval.embed_3gpp [--spec 38.101-1] [--clause 6] [--limit 150]

--clause keeps one clause and its sub-clauses; --limit keeps the first N chunks in document
order (across the selected specs). With either filter only the selected chunks are loaded and
nothing is deleted; with neither, a full load also removes chunks no longer in chunks.jsonl.
"""

import argparse

from rflens.config import get_retrieval_settings
from rflens.db import get_engine
from rflens.ingest.spec_fetch import SPECS
from rflens.retrieval.embedder import EmbedError, OllamaEmbedder
from rflens.retrieval.index import embed_pending, load_chunks, select_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Load and embed 3GPP chunks")
    parser.add_argument("--spec", help="only this spec, e.g. 38.101-1")
    parser.add_argument("--clause", help="only this clause and its sub-clauses, e.g. 6")
    parser.add_argument("--limit", type=int, help="first N chunks in document order")
    args = parser.parse_args()

    settings = get_retrieval_settings()
    embedder = OllamaEmbedder(settings.ollama_base_url, settings.embed_model)
    print(f"embedding model: {embedder.model_id}")  # fails early if Ollama or the model is missing
    sources = [s for s in SPECS if args.spec in (None, s.spec)]
    if not sources:
        raise SystemExit(f"unknown spec {args.spec!r}; known: {[s.spec for s in SPECS]}")
    full = args.clause is None and args.limit is None  # all chunks of each selected spec
    remaining = args.limit
    with get_engine().connect() as conn:
        for source in sources:
            if not source.chunks_path.exists():
                raise SystemExit(f"{source.chunks_path} not found; run make chunk-3gpp first")
            rows = select_rows(source.chunks_path, clause=args.clause, limit=remaining)
            if remaining is not None:
                remaining -= len(rows)
            ids = load_chunks(conn, source, rows, prune=full)
            conn.commit()
            print(f"{source.spec}: {len(ids)} chunks selected")
            try:
                stats = embed_pending(
                    conn, embedder, ids, batch_size=settings.embed_batch_size, commit=conn.commit
                )
            except EmbedError as e:
                raise SystemExit(f"STOPPED: {e}") from e
            print(
                f"{source.spec}: embedded={stats.embedded} skipped={stats.skipped} "
                f"tokens={stats.tokens} elapsed={stats.seconds:.1f}s"
            )
            print(f"{source.spec}: {stats.token_summary()}")


if __name__ == "__main__":
    main()
