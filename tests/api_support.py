"""Shared fixtures of the API and UI tests: the real FastAPI app over the real orchestrator, with
the fake provider for the model, in-process sql-mcp and docs-mcp servers (over MCP, on the test
database) and the real checkpointer. No paid call is possible here.
"""

import json
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Connection, create_engine, text

from rflens.agents.sql_tools import SCHEMA_QUESTION, SqlMcpTools
from rflens.api.main import create_app
from rflens.api.service import OrchestratorService, Parts
from rflens.config import (
    ApiSettings,
    GraphSettings,
    LlmSettings,
    get_checkpointer_settings,
    get_sql_mcp_settings,
)
from rflens.db import get_engine
from rflens.graph.docs_tools import DocsMcpTools
from rflens.llm.base import Pricing, Rate
from rflens.llm.fake import FakeProvider
from rflens.llm.gateway import Gateway
from rflens.llm.guards import RateLimiter
from rflens.llm.ledger import Ledger
from rflens.mcp.docs_server import create_docs_server
from rflens.mcp.sql_server import create_sql_server
from rflens.retrieval.embedder import HashEmbedder
from rflens.retrieval.index import embed_pending, load_chunks
from test_graph import BAND_TEXT, LTE, NR, NR_TEXT, chunk_row

API_KEY = "test-key-0123456789"
PRICING = Pricing({"fake": {"fake-1": {"standard": Rate(1.0, 1.0)}}}, usd_to_inr=85.7)


def plan_json(route="sql", spec="none", doc_query="", sql_question="", bands=None, reason=""):
    return json.dumps(
        {
            "route": route,
            "spec": spec,
            "doc_query": doc_query,
            "sql_question": sql_question,
            "bands": bands or [],
            "refuse_reason": reason,
        }
    )


def sql_json(sql="SELECT 41 + 1 AS answer"):
    return json.dumps(
        {
            "refuse": False,
            "refuse_reason": "",
            "sql": sql,
            "tables_used": [],
            "assumptions": [],
        }
    )


def answer_json(text_="The answer is 42 [S1]."):
    return json.dumps({"answer": text_})


@pytest.fixture(scope="module")
def sql_server():
    cfg = get_sql_mcp_settings()
    reader, logger = create_engine(cfg.reader_url()), create_engine(cfg.logger_url())
    yield create_sql_server(reader.connect, logger.connect, cfg)
    reader.dispose()
    logger.dispose()


@pytest.fixture
def docs_conn() -> Iterator[Connection]:
    with get_engine().connect() as connection:
        transaction = connection.begin()
        try:
            for source, clause, body in ((LTE, "5.5", BAND_TEXT), (NR, "6.2.1", NR_TEXT)):
                rows = [chunk_row(source, clause, body)]
                ids = load_chunks(connection, source, rows, prune=False)
                embed_pending(connection, HashEmbedder(), ids, progress=lambda s: None)
            yield connection
        finally:
            transaction.rollback()


class ApiEnv:
    def __init__(self, sql_server, docs_conn, script: list, **api_kw):
        self.provider = FakeProvider(script=script)
        llm = LlmSettings(_env_file=None, llm_provider="fake", llm_fallback="")
        gateway = Gateway(
            llm,
            {"fake": self.provider},
            Ledger(get_engine(), llm.llm_budget_usd, 85.7),
            PRICING,
            rate_limiter=RateLimiter(10_000),
            sleep=lambda s: None,
            emit=lambda line: None,
        )

        @contextmanager
        def connect():
            yield docs_conn

        docs = DocsMcpTools(create_docs_server(connect, HashEmbedder(), "vector"))
        parts = Parts(gateway, SqlMcpTools(sql_server), docs, GraphSettings(_env_file=None))
        self.service = OrchestratorService(lambda: parts)
        api_kw.setdefault("api_rate_limit_per_minute", 10_000)
        self.settings = ApiSettings(_env_file=None, api_key=API_KEY, **api_kw)
        self.app = create_app(self.service, self.settings)
        self.client = TestClient(self.app, headers={"X-API-Key": API_KEY})

    def ask(self, question: str, **body) -> dict:
        response = self.client.post("/ask", json={"question": question, **body})
        assert response.status_code == 200, response.text
        return response.json()


@pytest.fixture
def make_api(sql_server, docs_conn) -> Iterator[Callable[..., ApiEnv]]:
    tags: list[str] = []

    def make(script: list, **api_kw) -> ApiEnv:
        return ApiEnv(sql_server, docs_conn, script, **api_kw)

    yield make
    engine = create_engine(get_checkpointer_settings().sa_url())
    with engine.begin() as c:
        for table in ("checkpoint_writes", "checkpoint_blobs", "checkpoints"):
            c.execute(text(f"DELETE FROM {table} WHERE thread_id LIKE 'api-%'"))
    engine.dispose()
    with get_engine().begin() as c:
        c.execute(text("DELETE FROM sql_query_log WHERE question = :q"), {"q": SCHEMA_QUESTION})
        c.execute(text("DELETE FROM sql_query_log WHERE question LIKE 'api test %'"))
        c.execute(text("DELETE FROM llm_calls"))
    del tags


def new_question(text_: str = "What is forty one plus one?") -> str:
    return f"api test {text_} t{uuid.uuid4().hex[:10]}"
