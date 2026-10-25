from collections.abc import Iterable, Iterator
from functools import lru_cache

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from rflens.config import get_llm_settings, get_settings


@lru_cache
def get_engine() -> Engine:
    return create_engine(get_settings().database_url, pool_pre_ping=True)


@lru_cache
def get_ledger_engine() -> Engine:
    """The LLM ledger's connection: role rflens_llm_ledger, rights on llm_calls only."""
    return create_engine(get_llm_settings().ledger_url(), pool_pre_ping=True)


def analyze(tables: Iterable[str]) -> None:
    """Refresh planner statistics. Loaders call it after their load transaction has committed:
    stale statistics skew the cost estimates that sql-mcp's cost gate relies on. The names are
    the loaders' own table constants, never user input."""
    with get_engine().begin() as conn:
        conn.execute(text("ANALYZE " + ", ".join(tables)))


@lru_cache
def get_sessionmaker() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), expire_on_commit=False)


def get_session() -> Iterator[Session]:
    with get_sessionmaker()() as session:
        yield session
