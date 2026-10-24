from functools import lru_cache

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    RRF_K, SEARCH_CANDIDATES)."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    ollama_base_url: str = "http://localhost:11434"
    embed_model: str = "nomic-embed-text:latest"
    embed_batch_size: int = 8
    rrf_k: int = 60
    search_candidates: int = 50


@lru_cache
def get_settings() -> Settings:
    return Settings()


def get_chunk_settings() -> ChunkSettings:
    return ChunkSettings()


def get_retrieval_settings() -> RetrievalSettings:
    return RetrievalSettings()
