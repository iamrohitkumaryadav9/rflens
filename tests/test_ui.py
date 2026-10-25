"""The Streamlit UI on top of the API: the page is driven with Streamlit's AppTest, the API is the
real FastAPI app (fake provider, in-process MCP servers, real checkpointer). No paid calls.
"""

# ruff: noqa: F811
# (the fixtures imported from api_support are used by name)

from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from api_support import (  # noqa: F401
    answer_json,
    docs_conn,
    make_api,
    new_question,
    plan_json,
    sql_json,
    sql_server,
)
from rflens.ui.client import ApiClient, ApiError, badge_style, cost_line, source_title

APP = Path(__file__).resolve().parents[1] / "src" / "rflens" / "ui" / "app.py"
SQL_SCRIPT = [plan_json("sql"), sql_json(), answer_json()]


def page(env) -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.session_state["client"] = ApiClient(env.client)
    return at.run()


def texts(elements) -> str:
    return "\n".join(str(getattr(e, "value", "")) for e in elements)


def ask(at: AppTest, question: str, approval: bool = False) -> AppTest:
    at.text_area(key="question").set_value(question)
    if approval:
        at.checkbox(key="ask_first").check()
    return at.button(key="ask").click().run()


def test_the_page_loads_without_a_thread(make_api) -> None:
    at = page(make_api([]))
    assert not at.exception
    assert at.title[0].value == "RFLens"
    assert [b.key for b in at.button] == ["ask"]
    assert "none yet" in texts(at.sidebar.caption)


def test_an_answer_comes_with_a_badge_clickable_sources_and_cost(make_api) -> None:
    env = make_api(SQL_SCRIPT)
    at = ask(page(env), new_question())
    assert not at.exception
    assert "The answer is 42 [S1]." in texts(at.markdown)
    assert ":green-badge[verified]" in texts(at.markdown)
    labels = [e.label for e in at.expander]
    assert any(label.startswith("[S1] SQL result: query #") for label in labels)
    assert "Verification: passed" in labels and "Calls and cost per node" in labels
    sql = texts(at.code)
    assert "SELECT 41 + 1" in sql
    assert "3 model call(s)" in texts(at.caption)
    assert len(at.dataframe) >= 2  # the SQL rows and the calls per node


def test_pending_approval_shows_the_plan_panel_and_approve_finishes_the_thread(make_api) -> None:
    env = make_api(SQL_SCRIPT)
    at = ask(page(env), new_question(), approval=True)
    assert not at.exception
    assert any(s.value == "Approval needed" for s in at.subheader)
    assert "ask: ASK is set" in texts(at.warning)
    assert {b.key for b in at.button} >= {"approve", "edit", "cancel"}
    assert "The answer is 42" not in texts(at.markdown)
    at = at.button(key="approve").click().run()
    assert not at.exception
    assert "The answer is 42 [S1]." in texts(at.markdown)
    assert not any(s.value == "Approval needed" for s in at.subheader)


def test_cancel_ends_the_thread(make_api) -> None:
    env = make_api([plan_json("sql")])
    at = ask(page(env), new_question(), approval=True)
    at = at.button(key="cancel").click().run()
    assert not at.exception
    assert "Not run: the approval step was declined." in texts(at.markdown)
    assert ":gray-badge[cancelled]" in texts(at.markdown)
    assert len(env.provider.calls) == 1


def test_edit_sends_the_chosen_spec_and_a_spec_source_is_shown_in_the_sources(make_api) -> None:
    script = [plan_json("doc", "none", doc_query="uplink"), answer_json("Band 5 uplink [D1].")]
    env = make_api(script)
    at = ask(page(env), new_question("What is the uplink range of band 5?"), approval=True)
    select = at.selectbox[0]
    assert select.options == ["36.101", "38.101-1", "none"]
    select.select("36.101")
    at = at.button(key="edit").click().run()
    assert not at.exception
    doc = [e for e in at.expander if e.label.startswith("[D1] TS 36.101")]
    assert doc, [e.label for e in at.expander]
    assert "824 MHz" in texts(at.get("text"))  # the chunk text, read through docs-mcp


def test_recent_threads_are_listed_and_can_be_reopened(make_api) -> None:
    env = make_api(SQL_SCRIPT)
    question = new_question()
    created = env.ask(question, ask=True)  # left pending, as if the browser had been closed
    at = page(env)
    sidebar_keys = [b.key for b in at.sidebar.button]
    assert f"thread-{created['thread_id']}" in sidebar_keys
    at = at.sidebar.button(key=f"thread-{created['thread_id']}").click().run()
    assert not at.exception
    assert any(s.value == "Approval needed" for s in at.subheader)
    assert question in texts(at.markdown)


def test_an_api_error_is_shown_not_raised(make_api) -> None:
    env = make_api([])
    at = ask(page(env), "hi")  # shorter than the API accepts
    assert not at.exception
    assert "422" in texts(at.error)


# --- the client and the helpers (no Streamlit) --------------------------------------------------


def test_the_client_turns_http_errors_into_api_errors(make_api) -> None:
    env = make_api([])
    client = ApiClient(TestClient(env.app, headers={"X-API-Key": "wrong"}))
    with pytest.raises(ApiError) as e:
        client.threads()
    assert e.value.status == 401 and "X-API-Key" in e.value.detail
    good = ApiClient(env.client)
    with pytest.raises(ApiError) as missing:
        good.thread("api-nope")
    assert missing.value.status == 404
    assert good.threads() == []


def test_client_from_env_uses_the_api_key_and_url(monkeypatch) -> None:
    monkeypatch.setenv("API_KEY", "abc")
    monkeypatch.setenv("API_URL", "http://127.0.0.1:9999")
    client = ApiClient.from_env()
    assert isinstance(client.http, httpx.Client)
    assert client.http.headers["x-api-key"] == "abc" and str(client.http.base_url).startswith(
        "http://127.0.0.1:9999"
    )


def test_helpers() -> None:
    assert badge_style("verified") == ("green", "verified")
    assert badge_style("something") == ("gray", "something")
    sql = {"kind": "sql", "tag": "S1", "query_id": 7, "row_count": 3}
    assert source_title(sql) == "[S1] SQL result: query #7, 3 row(s)"
    doc = {
        "kind": "doc",
        "tag": "D2",
        "spec": "36.101",
        "version": "19.6.0",
        "clause": "5.5",
        "clause_title": "Operating bands",
        "table_number": "5.5-1",
    }
    assert source_title(doc) == "[D2] TS 36.101 V19.6.0, clause 5.5, table 5.5-1 - Operating bands"
    assert cost_line({"calls": 3, "usd": 0.0036}) == "3 model call(s), $0.0036"
