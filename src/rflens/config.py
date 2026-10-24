from functools import lru_cache
from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import URL, make_url


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str


class ChunkSettings(BaseSettings):
    """Spec chunk sizes in characters; override with CHUNK_SOFT_CHARS / CHUNK_HARD_CHARS."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    chunk_soft_chars: int = 3000
    chunk_hard_chars: int = 6000
    # Tables are number-dense (about 2.4 chars/token vs 4.4 for prose), so they get a lower cap
    # to stay inside nomic-embed-text's 2,048-token context.
    chunk_table_hard_chars: int = 3600
    # No chunk may have more than this many estimated embedding tokens (see estimate_tokens in
    # ingest/spec_chunker.py). Characters alone cannot do it: dense carrier-aggregation tables
    # run at 1.3-1.8 chars/token, so 3,600 chars can be 2,700 tokens against a 2,048 context.
    # 1,840 is 10% under the context limit (equals MAX_SAFE_TOKENS in retrieval/embedder.py).
    chunk_max_tokens: int = 1840

    @model_validator(mode="after")
    def _check_sizes(self) -> "ChunkSettings":
        if self.chunk_max_tokens < 50:
            raise ValueError("need CHUNK_MAX_TOKENS >= 50")
        if not 0 < self.chunk_soft_chars <= self.chunk_hard_chars:
            raise ValueError("need 0 < CHUNK_SOFT_CHARS <= CHUNK_HARD_CHARS")
        if not 0 < self.chunk_table_hard_chars <= self.chunk_hard_chars:
            raise ValueError("need 0 < CHUNK_TABLE_HARD_CHARS <= CHUNK_HARD_CHARS")
        return self


class RetrievalSettings(BaseSettings):
    """Embedding and search settings (env: OLLAMA_BASE_URL, EMBED_MODEL, EMBED_BATCH_SIZE,
    RRF_K, SEARCH_CANDIDATES, RETRIEVAL_MODE)."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    ollama_base_url: str = "http://localhost:11434"
    embed_model: str = "nomic-embed-text:latest"
    embed_batch_size: int = 8
    rrf_k: int = 60
    search_candidates: int = 50
    # Default search mode when the caller does not pick one. INTERIM: vector, from the smoke-eval
    # diagnosis (the keyword leg is length-biased and adds noise to hybrid); revisit with the
    # week 3 golden set.
    retrieval_mode: Literal["vector", "keyword", "hybrid"] = "vector"


class DocsReaderSettings(BaseSettings):
    """docs-mcp: the read-only database role it connects as, and where it listens.

    Env: DOCS_READER_PASSWORD (required; also read by migration 0007 when it creates the role),
    DOCS_READER_USER, DOCS_MCP_HOST, DOCS_MCP_PORT.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    docs_reader_password: str
    docs_reader_user: str = "rflens_docs_reader"
    docs_mcp_host: str = "127.0.0.1"
    docs_mcp_port: int = 8101

    def database_url(self) -> URL:
        """DATABASE_URL with the user and password swapped for the read-only role."""
        base = make_url(get_settings().database_url)
        return base.set(username=self.docs_reader_user, password=self.docs_reader_password)


class SqlMcpSettings(BaseSettings):
    """sql-mcp: its two database roles, where it listens, and the query limits.

    Env: SQL_READER_PASSWORD and SQL_LOGGER_PASSWORD (required; also read by migration 0008 when
    it creates the roles), SQL_READER_USER, SQL_LOGGER_USER, SQL_MCP_HOST, SQL_MCP_PORT,
    SQL_MAX_COST (planner cost above which a query is rejected), SQL_STATEMENT_TIMEOUT_MS,
    SQL_DEFAULT_LIMIT, SQL_MAX_LIMIT.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    sql_reader_password: str
    sql_logger_password: str
    sql_reader_user: str = "rflens_sql_reader"
    sql_logger_user: str = "rflens_sql_logger"
    sql_mcp_host: str = "127.0.0.1"
    sql_mcp_port: int = 8102
    sql_max_cost: float = 500_000
    sql_statement_timeout_ms: int = 15_000
    sql_default_limit: int = 100
    sql_max_limit: int = 1000

    @model_validator(mode="after")
    def _check_limits(self) -> "SqlMcpSettings":
        if not 0 < self.sql_default_limit <= self.sql_max_limit:
            raise ValueError("need 0 < SQL_DEFAULT_LIMIT <= SQL_MAX_LIMIT")
        if self.sql_max_cost <= 0 or self.sql_statement_timeout_ms <= 0:
            raise ValueError("need SQL_MAX_COST > 0 and SQL_STATEMENT_TIMEOUT_MS > 0")
        return self

    def _url(self, user: str, password: str) -> URL:
        return make_url(get_settings().database_url).set(username=user, password=password)

    def reader_url(self) -> URL:
        return self._url(self.sql_reader_user, self.sql_reader_password)

    def logger_url(self) -> URL:
        return self._url(self.sql_logger_user, self.sql_logger_password)


@lru_cache
def get_settings() -> Settings:
    return Settings()


def get_chunk_settings() -> ChunkSettings:
    return ChunkSettings()


def get_retrieval_settings() -> RetrievalSettings:
    return RetrievalSettings()


def get_docs_reader_settings() -> DocsReaderSettings:
    return DocsReaderSettings()


def get_sql_mcp_settings() -> SqlMcpSettings:
    return SqlMcpSettings()
