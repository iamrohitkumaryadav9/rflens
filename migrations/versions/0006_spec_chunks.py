"""documents and chunks for spec retrieval (pgvector + full text)

documents: one row per source document (a spec release: its zip and the zip's SHA-256).
chunks: one row per chunk from the chunker (data/interim/3gpp/<spec>/chunks.jsonl).

embedding is vector(768), the output size of nomic-embed-text; it stays NULL until the chunk is
embedded, and embedding_model records the model tag and digest that produced it (a chunk whose
embedding_model differs from the configured model counts as not embedded).

tsv is a generated column: to_tsvector('english', rflens_fts_norm(text)). rflens_fts_norm splits
a number from a glued unit ("23dBm" -> "23 dBm") and turns "/" and en dashes into spaces
("dBm/15kHz"), which the stock parser keeps as one token and so never matches "dBm". Queries must
pass their text through the same function. Identifiers such as n78, 6.2.1, NS_04 and A-MPR are
unaffected. 'english' (not 'simple') because stemming matters for prose ("requirement" vs
"requirements") and both configs index those identifiers identically.

Revision ID: 0006
Revises: 0005
Create Date: 2026-10-24
"""

from alembic import op

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        r"""
        CREATE FUNCTION rflens_fts_norm(t text) RETURNS text
        LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$
            SELECT regexp_replace(
                regexp_replace(t, '(\d)(dBm|dBc|dBi|dB|MHz|kHz|GHz|Hz|ms|us|%)', '\1 \2', 'g'),
                '[/–]', ' ', 'g')
        $$
        """
    )
    op.execute(
        """
        CREATE TABLE documents (
            id bigserial PRIMARY KEY,
            spec text NOT NULL,
            version text NOT NULL,
            release integer NOT NULL,
            source_zip text NOT NULL,
            sha256 text NOT NULL,
            created_at timestamptz NOT NULL DEFAULT now(),
            CONSTRAINT uq_documents_spec_version UNIQUE (spec, version)
        )
        """
    )
    op.execute(
        """
        CREATE TABLE chunks (
            chunk_id text PRIMARY KEY,
            document_id bigint NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
            kind text NOT NULL CHECK (kind IN ('text', 'table')),
            clause text NOT NULL,
            clause_title text,
            parent_path text[] NOT NULL DEFAULT '{}',
            table_number text,
            caption text,
            table_part text,
            text text NOT NULL,
            raw_text text NOT NULL,
            metadata jsonb NOT NULL DEFAULT '{}',
            text_sha256 text NOT NULL,
            embedding vector(768),
            embedding_model text,
            tsv tsvector GENERATED ALWAYS AS
                (to_tsvector('english'::regconfig, rflens_fts_norm(text))) STORED,
            CONSTRAINT ck_chunks_embedding_model
                CHECK ((embedding IS NULL) = (embedding_model IS NULL))
        )
        """
    )
    op.execute("CREATE INDEX ix_chunks_document_id ON chunks (document_id)")
    op.execute("CREATE INDEX ix_chunks_clause ON chunks (clause)")
    op.execute("CREATE INDEX ix_chunks_tsv ON chunks USING gin (tsv)")
    op.execute(
        "CREATE INDEX ix_chunks_embedding ON chunks USING hnsw (embedding vector_cosine_ops)"
    )


def downgrade() -> None:
    op.execute("DROP TABLE chunks")
    op.execute("DROP TABLE documents")
    op.execute("DROP FUNCTION rflens_fts_norm(text)")
