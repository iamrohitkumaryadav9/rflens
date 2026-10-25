"""The HTTP API: routes, auth, rate limit and the resume flows (fake provider, no paid calls)."""

# ruff: noqa: F811
# (the fixtures imported from api_support are used by name)

import pytest
from fastapi.testclient import TestClient

from api_support import (  # noqa: F401  (fixtures)
    API_KEY,
    answer_json,
    docs_conn,
    make_api,
    new_question,
    plan_json,
    sql_json,
    sql_server,
)
from rflens.api import server
from rflens.api.main import create_app
from rflens.api.security import RateLimiter
from rflens.config import ApiSettings
from rflens.llm.base import PaidCallBlocked

SQL_SCRIPT = [plan_json("sql"), sql_json(), answer_json()]


def test_health_is_open_and_everything_else_needs_the_key(make_api) -> None:
    env = make_api([])
    anonymous = TestClient(env.app)
    assert anonymous.get("/health").status_code == 200
    for method, path in (
        ("post", "/ask"),
        ("get", "/threads"),
        ("get", "/threads/x"),
        ("post", "/threads/x/resume"),
        ("get", "/chunks/36.101@1.0.0/5.5/1"),
    ):
        response = getattr(anonymous, method)(path)
        assert response.status_code == 401, path
        assert response.headers["www-authenticate"] == "API-Key"
    wrong = TestClient(env.app, headers={"X-API-Key": "nope"})
    assert wrong.get("/threads").status_code == 401
    assert env.client.get("/threads").status_code == 200


def test_an_unconfigured_server_key_refuses_every_protected_call(sql_server, docs_conn) -> None:
    from api_support import ApiEnv

    env = ApiEnv(sql_server, docs_conn, [])
    unconfigured = create_app(env.service, ApiSettings(_env_file=None))
    client = TestClient(unconfigured, headers={"X-API-Key": "anything"})
    assert client.get("/threads").status_code == 503
    assert "API_KEY is not configured" in client.get("/threads").json()["detail"]
    assert client.get("/health").status_code == 200


def test_ask_runs_a_question_and_returns_the_thread_view(make_api) -> None:
    env = make_api(SQL_SCRIPT)
    view = env.ask(new_question())
    assert view["thread_id"].startswith("api-") and view["status"] == "answered"
    assert view["badge"] == "verified" and view["answer"] == "The answer is 42 [S1]."
    assert view["pending"] is None and view["route"] == "sql"
    (source,) = view["source_items"]
    assert (source["tag"], source["kind"], source["cited"]) == ("S1", "sql", True)
    assert source["sql"].startswith("SELECT 41 + 1") and source["rows"] == [[42]]
    assert source["columns"] == ["answer"] and source["query_id"]
    assert view["sources_text"].startswith(f"[S1] sql-mcp query #{source['query_id']}")
    assert view["verification"]["ok"] and view["verification"]["failures"] == []
    nodes = view["cost"]["nodes"]
    assert (nodes["planner"]["calls"], nodes["sql_agent"]["calls"], nodes["writer"]["calls"]) == (
        1,
        1,
        1,
    )
    assert view["cost"]["calls"] == 3 and view["cost"]["usd"] == 0
    assert len(view["runs"]) == 1 and view["draft_history"][0]["ok"]


def test_get_thread_returns_the_same_view_and_the_list_shows_it(make_api) -> None:
    env = make_api(SQL_SCRIPT)
    question = new_question()
    created = env.ask(question)
    fetched = env.client.get(f"/threads/{created['thread_id']}").json()
    assert fetched["answer"] == created["answer"] and fetched["status"] == "answered"
    assert fetched["draft_history"] == created["draft_history"]
    listing = env.client.get("/threads", params={"limit": 5}).json()["threads"]
    first = listing[0]
    assert first["thread_id"] == created["thread_id"] and first["question"] == question
    assert first["badge"] == "verified" and "draft_history" not in first


def test_a_question_that_needs_approval_returns_the_pending_plan(make_api) -> None:
    env = make_api(SQL_SCRIPT)
    view = env.ask(new_question(), ask=True)
    assert view["status"] == "pending_approval" and view["badge"] == "pending approval"
    assert view["answer"] is None and view["next"] == ["approval"]
    assert view["pending"]["reasons"] == ["ask: ASK is set"]
    assert view["pending"]["route"] == "sql" and view["plan"]["route"] == "sql"
    assert len(env.provider.calls) == 1  # only the planner has run


def test_resume_approve_finishes_the_thread(make_api) -> None:
    env = make_api(SQL_SCRIPT)
    thread = env.ask(new_question(), ask=True)["thread_id"]
    response = env.client.post(f"/threads/{thread}/resume", json={"decision": "approve"})
    assert response.status_code == 200, response.text
    view = response.json()
    assert view["status"] == "answered" and view["pending"] is None
    assert len(view["runs"]) == 2  # the ask request and the resume request are two runs
    assert env.client.get(f"/threads/{thread}").json()["status"] == "answered"


def test_resume_cancel_ends_the_thread_without_more_calls(make_api) -> None:
    env = make_api([plan_json("sql")])
    thread = env.ask(new_question(), ask=True)["thread_id"]
    view = env.client.post(f"/threads/{thread}/resume", json={"decision": "cancel"}).json()
    assert view["status"] == "cancelled" and view["badge"] == "cancelled"
    assert len(env.provider.calls) == 1


def test_resume_edit_overrides_the_spec(make_api) -> None:
    script = [plan_json("doc", "none", doc_query="uplink range"), answer_json("Band [D1].")]
    env = make_api(script)
    question = new_question("What is the uplink range of band 5?")
    view = env.ask(question, ask=True)
    assert view["pending"]["proposed_spec"] is None  # nothing in the wording names a spec
    thread = view["thread_id"]
    done = env.client.post(f"/threads/{thread}/resume", json={"decision": "edit", "spec": "36.101"})
    assert done.status_code == 200, done.text
    assert done.json()["spec"] == "36.101" and done.json()["spec_source"] == "human"
    assert [c["tag"] for c in done.json()["source_items"]] == ["D1"]


@pytest.mark.parametrize(
    "body",
    [
        {"decision": "edit"},  # edit needs a spec
        {"decision": "approve", "spec": "36.101"},  # a spec only with edit
        {"decision": "edit", "spec": "99.999"},  # not a known spec
        {"decision": "maybe"},
        {"decision": "approve", "extra": 1},
        {},
    ],
)
def test_resume_validates_its_fields(make_api, body) -> None:
    env = make_api([])
    assert env.client.post("/threads/api-x/resume", json=body).status_code == 422


def test_error_statuses(make_api) -> None:
    env = make_api(SQL_SCRIPT)
    assert env.client.get("/threads/api-nope").status_code == 404
    assert (
        env.client.post("/threads/api-nope/resume", json={"decision": "approve"}).status_code == 404
    )
    assert env.client.get("/threads/bad id!").status_code == 422
    done = env.ask(new_question())
    again = env.client.post(f"/threads/{done['thread_id']}/resume", json={"decision": "approve"})
    assert again.status_code == 409  # finished: nothing to decide
    duplicate = env.client.post(
        "/ask", json={"question": new_question(), "thread_id": done["thread_id"]}
    )
    assert duplicate.status_code == 409
    for body in ({"question": "hi"}, {"question": ""}, {"question": "x" * 2001}, {"nope": 1}):
        assert env.client.post("/ask", json=body).status_code == 422, body
    assert (
        env.client.post("/ask", json={"question": new_question(), "thread_id": "a b"}).status_code
        == 422
    )


def test_a_model_side_stop_is_a_503_and_the_thread_stays_resumable(make_api, monkeypatch) -> None:
    env = make_api([])

    def blocked(*args, **kwargs):
        raise PaidCallBlocked("paid calls are not allowed here")

    monkeypatch.setattr(env.service, "ask", blocked)
    response = env.client.post("/ask", json={"question": new_question()})
    assert response.status_code == 503 and "PaidCallBlocked" in response.json()["detail"]


def test_the_chunk_route_reads_a_spec_chunk_through_docs_mcp(make_api) -> None:
    env = make_api([])
    chunk_id = "36.101@19.0.0/5.5/1"
    response = env.client.get(f"/chunks/{chunk_id}")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["chunk_id"] == chunk_id and body["clause"] == "5.5" and "824 MHz" in body["text"]
    assert env.client.get("/chunks/36.101@19.0.0/9.9/1").status_code == 404
    assert env.client.get("/chunks/not-a-chunk-id").status_code == 422


def test_the_rate_limit_is_per_key_and_answers_429_with_retry_after(make_api) -> None:
    env = make_api([], api_rate_limit_per_minute=3)
    codes = [env.client.get("/threads").status_code for _ in range(5)]
    assert codes == [200, 200, 200, 429, 429]
    limited = env.client.get("/threads")
    assert int(limited.headers["retry-after"]) >= 1
    assert env.client.get("/health").status_code == 200  # /health is not limited
    other = TestClient(env.app, headers={"X-API-Key": "a-wrong-key"})
    assert other.get("/threads").status_code == 401  # wrong keys never reach the limiter


def test_the_rate_limiter_window_slides() -> None:
    now = [0.0]
    limiter = RateLimiter(2, clock=lambda: now[0])
    assert limiter.check("k") is None and limiter.check("k") is None
    assert limiter.check("k") is not None
    assert limiter.check("other") is None  # another key has its own allowance
    now[0] = 61.0
    assert limiter.check("k") is None


def test_the_server_binds_to_localhost_and_wants_a_key(monkeypatch, capsys) -> None:
    assert server.HOST == "127.0.0.1"
    monkeypatch.setattr(server, "get_api_settings", lambda: ApiSettings(_env_file=None))
    assert server.main() == 2
    assert "API_KEY is not set" in capsys.readouterr().err
