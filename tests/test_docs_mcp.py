"""docs-mcp tests: in-process MCP client with the fake embedder (no Ollama), real Postgres.

Test data are synthetic chunks loaded inside a rolled-back transaction; the server under test is
handed that transaction's connection. A separate test connects as the real read-only role.
"""

import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import anyio
import psycopg.errors as pg_errors
import pytest
from mcp import Client
from sqlalchemy import Connection, create_engine, text
from sqlalchemy.exc import DBAPIError, ProgrammingError
from starlette.testclient import TestClient

from rflens.config import get_docs_reader_settings, get_settings
from rflens.db import get_engine
from rflens.ingest.spec_chunker import chunk_parts
from rflens.ingest.spec_fetch import SpecSource
from rflens.mcp.docs_server import (
    CHUNK_ID_RE,
    MAX_K,
    create_docs_server,
    http_options,
    make_snippet,
)
from rflens.retrieval.embedder import ContextLengthError, EmbedError, HashEmbedder
from rflens.retrieval.index import embed_pending, load_chunks

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "spec" / "synthetic"
NR = SpecSource("38.101-1", "38_series", "t-nr.zip", "19.0.0", 19, "0" * 64)
LTE = SpecSource("36.101", "36_series", "t-lte.zip", "19.0.0", 19, "1" * 64)
PROTOCOL = "2026-07-28"


@pytest.fixture
def conn() -> Iterator[Connection]:
    """Real Postgres connection; rolled back so real data is never touched."""
    with get_engine().connect() as connection:
        transaction = connection.begin()
        try:
            yield connection
        finally:
            transaction.rollback()


def retag(rows: list[dict], spec: str) -> list[dict]:
    """The synthetic fixtures are TS 99.101-1; relabel them as a real spec name."""
    return [json.loads(json.dumps(r).replace("99.101-1", spec)) for r in rows]


def fixture_rows(soft: int = 3000, hard: int = 6000) -> list[dict]:
    parts = [(p.name, p.read_text("utf-8")) for p in sorted(FIXTURE_DIR.glob("*.md"))]
    chunks, _ = chunk_parts(parts, soft, hard)
    return [json.loads(json.dumps(asdict(c))) for c in chunks]


def simple_row(spec: str, i: int, body: str, clause: str = "6.9") -> dict:
    return {
        "chunk_id": f"{spec}@19.0.0/{clause}/{i}",
        "spec": spec,
        "version": "19.0.0",
        "release": 19,
        "clause": clause,
        "clause_title": "Title",
        "parent_path": [],
        "parent_titles": [],
        "kind": "text",
        "position": i,
        "table_caption": None,
        "table_number": None,
        "table_part": None,
        "source_file": "x.md",
        "text": body,
        "raw_text": body,
        "media": [],
        "image_placeholders": 0,
        "forced_split": False,
        "char_count": len(body),
    }


def split_table_with_notes_rows() -> list[dict]:
    """A table split into pieces whose long notes live in their own chunk (-> related_chunks)."""
    rows_html = "\n".join(
        f"<tr>\n<td>row-{i:03d}</td>\n<td>{i * 3}</td>\n</tr>" for i in range(120)
    )
    notes = "\n\n".join(f"NOTE {n}: " + "explanatory words " * 14 for n in range(1, 4))
    md = (
        "# 8 Big table clause\n\nTable 8-1: Big table\n\n<table>\n<thead>\n<tr>\n<th>Name</th>\n"
        f"<th>Value</th>\n</tr>\n</thead>\n<tbody>\n{rows_html}\n</tbody>\n</table>\n\n{notes}\n"
    )
    cover = (FIXTURE_DIR / "00_cover.md").read_text("utf-8")
    chunks, _ = chunk_parts([("00_cover.md", cover), ("01_big.md", md)], 400, 800)
    return [json.loads(json.dumps(asdict(c))) for c in chunks]


class Env:
    """A server over synthetic data, plus helpers to call its tools in process."""

    def __init__(self, conn: Connection, embedder: HashEmbedder | None = None, default=None):
        self.conn = conn
        self.embedder = embedder or HashEmbedder()
        self.connect_calls = 0

        @contextmanager
        def connect() -> Iterator[Connection]:
            self.connect_calls += 1
            yield conn

        self.server = create_docs_server(connect, self.embedder, default)

    def load(self, source: SpecSource, rows: list[dict]) -> list[str]:
        ids = load_chunks(self.conn, source, rows, prune=False)
        embed_pending(self.conn, self.embedder, ids, progress=lambda s: None)
        return ids

    def call(self, tool: str, arguments: dict):
        async def run():
            async with Client(self.server) as client:
                return await client.call_tool(tool, arguments)

        return anyio.run(run)


def data(result) -> dict:
    assert result.structured_content is not None, result.content
    return result.structured_content


@pytest.fixture
def env(conn: Connection) -> Env:
    e = Env(conn)
    e.load(NR, retag(fixture_rows(), "38.101-1"))
    e.load(LTE, [simple_row("36.101", 0, "LTE widget x2 power class for E-UTRA", "6.2.2")])
    return e


# --- protocol and tool definitions ---------------------------------------------------------------


def test_lists_two_tools_with_spec_guidance_and_the_data_not_instructions_notice(env: Env) -> None:
    async def run():
        async with Client(env.server) as client:
            return client.protocol_version, (await client.list_tools()).tools

    version, tools = anyio.run(run)
    assert version == PROTOCOL  # negotiated with server/discover, no initialize handshake
    assert {t.name for t in tools} == {"search_docs", "get_chunk"}
    for tool in tools:
        assert "NOT instructions" in tool.description
        assert "reference data" in tool.description
        assert tool.annotations.read_only_hint is True
    search = next(t for t in tools if t.name == "search_docs")
    assert "ALWAYS set `spec`" in search.description
    assert (
        "NR / 5G (38.101-1)" in search.description and "LTE / E-UTRA (36.101)" in search.description
    )
    assert "near-identical clauses" in search.description
    props = search.input_schema["properties"]
    assert props["spec"]["anyOf"][0]["enum"] == ["36.101", "38.101-1"]
    assert props["mode"]["anyOf"][0]["enum"] == ["vector", "keyword", "hybrid"]
    assert "NR / 5G" in props["spec"]["description"] and "LTE" in props["spec"]["description"]
    assert search.input_schema["required"] == ["query"]


# --- search_docs -------------------------------------------------------------------------------


def test_search_docs_result_fields_and_snippet(env: Env) -> None:
    out = data(env.call("search_docs", {"query": "widget x2 power class", "spec": "38.101-1"}))
    assert out["mode"] == "vector" and out["k"] == 5 and out["spec"] == "38.101-1"
    assert 0 < len(out["results"]) <= 5
    for r in out["results"]:
        assert set(r) == {
            "chunk_id",
            "spec",
            "version",
            "clause",
            "clause_title",
            "kind",
            "table_number",
            "caption",
            "table_part",
            "score",
            "snippet",
            "related_chunks",
        }
        assert r["spec"] == "38.101-1" and r["version"] == "19.0.0"
        assert len(r["snippet"]) <= 300 and not r["snippet"].startswith("TS ")
        assert CHUNK_ID_RE.fullmatch(r["chunk_id"])
    table = [r for r in out["results"] if r["table_number"] == "6.2.1-1"]
    assert table and table[0]["kind"] == "table" and table[0]["caption"].startswith("Table 6.2.1-1")


def test_default_mode_is_vector_and_each_mode_is_selectable(
    conn: Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("RETRIEVAL_MODE", raising=False)
    e = Env(conn)
    e.load(NR, retag(fixture_rows(), "38.101-1"))
    seen = {}
    for mode in (None, "vector", "keyword", "hybrid"):
        args = {"query": "widget x2", "spec": "38.101-1"} | ({"mode": mode} if mode else {})
        seen[mode] = data(e.call("search_docs", args))["mode"]
    assert seen == {None: "vector", "vector": "vector", "keyword": "keyword", "hybrid": "hybrid"}
    monkeypatch.setenv("RETRIEVAL_MODE", "hybrid")  # env-overridable default
    assert data(e.call("search_docs", {"query": "widget x2"}))["mode"] == "hybrid"
    assert data(Env(conn, default="keyword").call("search_docs", {"query": "widget x2"}))[
        "mode"
    ] == ("keyword")


def test_spec_filter_keeps_the_other_spec_out(env: Env) -> None:
    for mode in ("vector", "keyword", "hybrid"):
        lte = data(
            env.call("search_docs", {"query": "widget x2 power", "spec": "36.101", "mode": mode})
        )
        assert lte["results"] and {r["spec"] for r in lte["results"]} == {"36.101"}
        nr = data(
            env.call("search_docs", {"query": "widget x2 power", "spec": "38.101-1", "mode": mode})
        )
        assert nr["results"] and {r["spec"] for r in nr["results"]} == {"38.101-1"}
    both = data(env.call("search_docs", {"query": "widget x2 power", "mode": "keyword", "k": 20}))
    assert {r["spec"] for r in both["results"]} == {"36.101", "38.101-1"}
    bad = env.call("search_docs", {"query": "widget", "spec": "99.101-1"})
    assert bad.is_error  # not one of the two specs: rejected by the input schema


def test_k_is_capped_at_20_and_must_be_positive(conn: Connection) -> None:
    e = Env(conn)
    e.load(NR, [simple_row("38.101-1", i, f"zeta widget common text {i}") for i in range(30)])
    out = data(e.call("search_docs", {"query": "zeta widget", "spec": "38.101-1", "k": 500}))
    assert MAX_K == 20 and out["k"] == 20 and len(out["results"]) == 20
    assert "capped at 20" in out["notice"]
    exact = data(e.call("search_docs", {"query": "zeta widget", "spec": "38.101-1", "k": 7}))
    assert len(exact["results"]) == 7 and exact["notice"] is None
    zero = e.call("search_docs", {"query": "zeta widget", "k": 0})
    assert zero.is_error and "k must be at least 1" in zero.content[0].text


def test_query_guardrails(env: Env) -> None:
    too_long = env.call("search_docs", {"query": "x" * 501})
    assert too_long.is_error and "maximum is 500" in too_long.content[0].text
    assert not env.call("search_docs", {"query": "widget " + "x" * 400}).is_error
    exactly = env.call("search_docs", {"query": "w" * 500})
    assert not exactly.is_error
    assert env.call("search_docs", {"query": "   "}).is_error


def test_embedding_outage_is_a_clear_error_and_keyword_still_works(conn: Connection) -> None:
    class Down(HashEmbedder):
        def embed_query(self, text: str) -> list[float]:
            raise EmbedError("connection refused")

    e = Env(conn, embedder=Down())
    e.load(NR, retag(fixture_rows(), "38.101-1"))
    down = e.call("search_docs", {"query": "widget x2", "mode": "vector"})
    assert down.is_error and "embedding service is unavailable" in down.content[0].text
    assert "mode='keyword'" in down.content[0].text
    assert not e.call("search_docs", {"query": "widget x2", "mode": "keyword"}).is_error


def test_sql_metacharacters_in_the_query_are_just_text(env: Env) -> None:
    evil = "widget'; DROP TABLE chunks; --"
    out = data(env.call("search_docs", {"query": evil, "mode": "keyword"}))
    assert out["query"] == evil
    assert env.conn.execute(text("SELECT count(*) FROM chunks")).scalar_one() > 0


# --- get_chunk and related chunks -----------------------------------------------------------------


def test_get_chunk_returns_text_metadata_and_related_notes(conn: Connection) -> None:
    e = Env(conn)
    rows = retag(split_table_with_notes_rows(), "38.101-1")
    notes = next(r for r in rows if r["table_part"] == "notes")
    e.load(NR, rows)
    found = data(
        e.call("search_docs", {"query": "row-050 Big table", "mode": "keyword", "spec": "38.101-1"})
    )
    piece = next(r for r in found["results"] if r["related_chunks"])
    assert piece["related_chunks"] == [notes["chunk_id"]]  # search exposes the pointer

    full = data(
        e.call("get_chunk", {"chunk_id": piece["chunk_id"]})
    )  # include_related defaults true
    assert full["chunk_id"] == piece["chunk_id"] and full["kind"] == "table"
    assert full["spec"] == "38.101-1" and full["version"] == "19.0.0" and full["clause"] == "8"
    assert full["table_number"] == "8-1" and "row-" in full["text"]
    assert full["related_chunk_ids"] == [notes["chunk_id"]]
    assert [r["chunk_id"] for r in full["related"]] == [notes["chunk_id"]]
    assert "NOTE 1:" in full["related"][0]["text"] and full["related"][0]["table_part"] == "notes"

    bare = data(e.call("get_chunk", {"chunk_id": piece["chunk_id"], "include_related": False}))
    assert bare["related"] == [] and bare["related_chunk_ids"] == [notes["chunk_id"]]
    plain = data(e.call("get_chunk", {"chunk_id": notes["chunk_id"]}))
    assert plain["related"] == [] and plain["related_chunk_ids"] == []  # notes point nowhere


def test_unknown_chunk_id_is_a_clear_error_not_a_crash(env: Env) -> None:
    result = env.call("get_chunk", {"chunk_id": "38.101-1@19.0.0/6.2.1/99999"})
    assert result.is_error
    assert (
        "unknown chunk_id" in result.content[0].text and "no such chunk" in result.content[0].text
    )


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "abc",
        "38.101-1",
        "38.101-1@19.0.0/6.2.1",
        "38.101-1@19.0.0/6.2.1/1; DROP TABLE chunks",
        "38.101-1@19.0.0/6.2.1/1' OR '1'='1",
        "38.101-1@19.0.0/../../etc/passwd/1",
        "38.101-1@19.0.0/6.2.1/1\n",
        "x" * 1000,
        "38.101-1@19.0.0/%s/1",
    ],
)
def test_malformed_chunk_ids_are_rejected_before_any_sql(env: Env, bad: str) -> None:
    before = env.connect_calls
    result = env.call("get_chunk", {"chunk_id": bad})
    assert result.is_error and "invalid chunk_id" in result.content[0].text
    assert env.connect_calls == before  # no database connection was even opened


def test_chunk_id_format_accepts_the_real_shapes() -> None:
    for ok in (
        "38.101-1@19.7.0/6.2.1/1",
        "36.101@19.6.0/A.3.9.2/4",
        "38.101-1@19.7.0/6.2.3.2~2/0",
        "38.101-1@19.7.0/Annex A/0",
        "38.101-1@19.7.0/6.5.A.2.2.4/12",
    ):
        assert CHUNK_ID_RE.fullmatch(ok), ok
    for bad in ("38.101-1@19.7/6.2.1/1", "38.101-1@19.7.0/6.2.1/", "38.101-1@19.7.0//1"):
        assert not CHUNK_ID_RE.fullmatch(bad), bad


def test_make_snippet() -> None:
    body = "word " * 200
    snippet = make_snippet(f"TS 38.101-1 V19.7.0 clause 6.2.1: Title\n\n{body}")
    assert len(snippet) <= 300 and snippet.endswith("…") and snippet.startswith("word word")
    short = make_snippet("TS 38.101-1 V19.7.0 clause 6.2.1: Title\n\nshort   text\nhere")
    assert short == "short text here"


def test_a_crash_inside_a_tool_does_not_leak_its_details(conn: Connection) -> None:
    class Boom(HashEmbedder):
        def embed_query(self, text: str) -> list[float]:
            raise RuntimeError("secret internal detail")

    e = Env(conn, embedder=Boom())
    result = e.call("search_docs", {"query": "widget x2", "mode": "vector"})
    assert result.is_error and "secret internal detail" not in result.content[0].text


def test_context_length_error_type_is_not_swallowed_as_a_search_result(conn: Connection) -> None:
    class TooLong(HashEmbedder):
        def embed_query(self, text: str) -> list[float]:
            raise ContextLengthError(0)

    result = Env(conn, embedder=TooLong()).call(
        "search_docs", {"query": "widget", "mode": "vector"}
    )
    assert result.is_error


# --- the HTTP layer (same options as `make docs-mcp`), without opening a port --------------------


def http_post(client: TestClient, body: dict, **headers: str):
    base = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "MCP-Protocol-Version": PROTOCOL,
    }
    return client.post("/mcp", json=body, headers=base | headers)


META = {
    "io.modelcontextprotocol/protocolVersion": PROTOCOL,
    "io.modelcontextprotocol/clientCapabilities": {},
}


def test_http_stateless_discover_call_and_header_rules(env: Env) -> None:
    app = env.server.streamable_http_app(host="127.0.0.1", **http_options("127.0.0.1"))
    with TestClient(app, base_url="http://127.0.0.1:8101") as client:
        discover = http_post(
            client,
            {"jsonrpc": "2.0", "id": 1, "method": "server/discover", "params": {"_meta": META}},
            **{"Mcp-Method": "server/discover"},
        )
        assert discover.status_code == 200
        assert discover.json()["result"]["supportedVersions"] == [PROTOCOL]
        assert "mcp-session-id" not in discover.headers  # no protocol-level sessions

        call = http_post(
            client,
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "search_docs",
                    "_meta": META,
                    "arguments": {"query": "widget x2", "spec": "38.101-1", "mode": "keyword"},
                },
            },
            **{"Mcp-Method": "tools/call", "Mcp-Name": "search_docs"},
        )
        assert call.status_code == 200
        result = call.json()["result"]
        assert result["resultType"] == "complete" and not result.get("isError")
        assert result["structuredContent"]["results"]
        assert "mcp-session-id" not in call.headers

        no_method = http_post(
            client,
            {"jsonrpc": "2.0", "id": 3, "method": "server/discover", "params": {"_meta": META}},
        )
        assert no_method.status_code == 400 and no_method.json()["error"]["code"] == -32020

        evil_origin = http_post(
            client,
            {"jsonrpc": "2.0", "id": 4, "method": "server/discover", "params": {"_meta": META}},
            **{"Mcp-Method": "server/discover", "Origin": "http://evil.example"},
        )
        assert evil_origin.status_code == 403

        ok_origin = http_post(
            client,
            {"jsonrpc": "2.0", "id": 5, "method": "server/discover", "params": {"_meta": META}},
            **{"Mcp-Method": "server/discover", "Origin": "http://localhost:3000"},
        )
        assert ok_origin.status_code == 200

        wrong_host = http_post(  # DNS rebinding: a Host header that is not localhost
            client,
            {"jsonrpc": "2.0", "id": 6, "method": "server/discover", "params": {"_meta": META}},
            **{"Mcp-Method": "server/discover", "Host": "evil.example"},
        )
        assert wrong_host.status_code == 421


# --- the read-only database role ------------------------------------------------------------------


@pytest.fixture(scope="module")
def reader_engine():
    settings = get_docs_reader_settings()
    engine = create_engine(settings.database_url())
    yield engine
    engine.dispose()


def test_reader_role_can_select_but_not_write_or_read_other_tables(reader_engine) -> None:
    with reader_engine.connect() as c:
        assert c.execute(text("SELECT current_user")).scalar_one() == "rflens_docs_reader"
        assert c.execute(text("SHOW default_transaction_read_only")).scalar_one() == "on"
        assert c.execute(text("SELECT count(*) FROM chunks")).scalar_one() >= 0
        assert c.execute(text("SELECT count(*) FROM documents")).scalar_one() >= 0
    forbidden = [
        "INSERT INTO documents (spec, version, release, source_zip, sha256) "
        "VALUES ('x', 'y', 1, 'z', 'w')",
        "INSERT INTO chunks (chunk_id, document_id, kind, clause, text, raw_text, text_sha256) "
        "VALUES ('a', 1, 'text', '1', 't', 't', 'h')",
        "UPDATE chunks SET text = 'changed'",
        "UPDATE documents SET spec = 'changed'",
        "DELETE FROM chunks",
        "TRUNCATE chunks",
        "SELECT * FROM data_sources",
        "SELECT * FROM secom_runs",
        "SELECT * FROM fcc_grantees",
        "CREATE TABLE rflens_should_not_exist (a int)",
        "DROP TABLE chunks",
    ]
    for statement in forbidden:
        # layer 1: the role's sessions default to read-only transactions (or the privilege check)
        with reader_engine.connect() as c, pytest.raises(DBAPIError) as err:
            c.execute(text(statement))
        assert isinstance(
            err.value.orig, (pg_errors.InsufficientPrivilege, pg_errors.ReadOnlySqlTransaction)
        ), statement
        # layer 2: even in a read-write transaction the privileges alone deny it
        with reader_engine.connect() as c, pytest.raises(ProgrammingError) as err:
            c.execute(text("SET TRANSACTION READ WRITE"))
            c.execute(text(statement))
        assert isinstance(err.value.orig, pg_errors.InsufficientPrivilege), statement


def test_the_server_runs_as_the_reader_role_in_every_mode() -> None:
    settings = get_docs_reader_settings()
    engine = create_engine(settings.database_url())
    try:
        server = create_docs_server(engine.connect, HashEmbedder())

        async def run():
            async with Client(server) as client:
                outs = {}
                for mode in ("vector", "keyword", "hybrid"):
                    res = await client.call_tool(
                        "search_docs", {"query": "maximum output power", "mode": mode, "k": 3}
                    )
                    outs[mode] = res.is_error
                unknown = await client.call_tool(
                    "get_chunk", {"chunk_id": "38.101-1@19.7.0/6.2.1/99999"}
                )
                return outs, unknown

        outs, unknown = anyio.run(run)
    finally:
        engine.dispose()
    assert outs == {"vector": False, "keyword": False, "hybrid": False}
    assert unknown.is_error and "unknown chunk_id" in unknown.content[0].text


def test_reader_settings_swap_only_the_credentials() -> None:
    reader = get_docs_reader_settings().database_url()
    base = get_settings().database_url
    from sqlalchemy.engine import make_url

    admin = make_url(base)
    assert reader.username == "rflens_docs_reader" and reader.username != admin.username
    assert reader.password != admin.password
    assert (reader.host, reader.port, reader.database) == (admin.host, admin.port, admin.database)
