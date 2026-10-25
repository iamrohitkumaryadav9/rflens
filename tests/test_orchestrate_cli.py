"""The kill-and-resume demo, automated: real processes, the fake provider, a real sql-mcp over HTTP
and the real test database.

`orchestrate` plans a question, saves the checkpoint and waits at the approval prompt; the test
kills that process outright, then a brand-new process (`orchestrate-resume`) finishes the thread.
"""

import json
import os
import queue
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text

from rflens.db import get_engine
from rflens.graph.app import open_checkpointer

ROOT = Path(__file__).resolve().parents[1]
CLI = [sys.executable, "-u", "-m", "rflens.graph.cli"]
WAITING = "Resume it from any process"
PLAN = {
    "route": "sql",
    "spec": "none",
    "doc_query": "",
    "sql_question": "",
    "bands": [],
    "refuse_reason": "",
}
SQL = {
    "refuse": False,
    "refuse_reason": "",
    "sql": "SELECT 41 + 1 AS answer",
    "tables_used": [],
    "assumptions": [],
}
ANSWER = {"answer": "The answer is 42 [S1]."}


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def sql_mcp_url() -> Iterator[str]:
    port = free_port()
    env = {**os.environ, "SQL_MCP_PORT": str(port), "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(
        [sys.executable, "-m", "rflens.mcp.sql_server"],
        cwd=ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.time() + 90
        while time.time() < deadline:
            try:
                socket.create_connection(("127.0.0.1", port), timeout=1).close()
                break
            except OSError:
                if proc.poll() is not None:
                    pytest.fail("sql-mcp did not start")
                time.sleep(0.5)
        else:
            pytest.fail("sql-mcp did not start in time")
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        proc.kill()
        proc.wait()


@pytest.fixture
def thread() -> Iterator[str]:
    value = f"demo-test-{uuid.uuid4().hex[:8]}"
    yield value
    with open_checkpointer() as saver:
        saver.delete_thread(value)
    with get_engine().begin() as c:
        c.execute(text("DELETE FROM llm_calls WHERE question_id = :q"), {"q": value})
        c.execute(text("DELETE FROM sql_query_log WHERE question LIKE :q"), {"q": f"%{value}%"})
        c.execute(text("DELETE FROM sql_query_log WHERE question = 'sql-agent schema context'"))


def env_for(script: list, sql_mcp_url: str, tmp_path: Path, name: str) -> dict[str, str]:
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps([json.dumps(item) for item in script]), encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k not in ("ASK", "LLM_ALLOW_PAID")}
    return {
        **env,
        "LLM_PROVIDER": "fake",
        "LLM_FALLBACK": "",
        "RFLENS_FAKE_SCRIPT": str(path),
        "SQL_MCP_URL": sql_mcp_url,
        "PYTHONUNBUFFERED": "1",
    }


def hard_kill(proc: subprocess.Popen, shown: str) -> None:
    """Terminate the interpreter that printed its pid (the venv launcher in front of it is a
    separate process on Windows), then the launcher. Nothing runs any cleanup."""
    match = re.search(r"pid (\d+)", shown)
    if match:
        try:
            os.kill(int(match.group(1)), signal.SIGTERM)  # TerminateProcess on Windows
        except OSError:
            pass
    proc.kill()
    proc.wait()


def run_cli(args: list[str], env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [*CLI, *args], cwd=ROOT, env=env, capture_output=True, text=True, timeout=180
    )


def start_and_wait(question: str, thread: str, env: dict[str, str]):
    """Start `orchestrate` and return it once it says it is waiting at the approval step."""
    proc = subprocess.Popen(
        [*CLI, "run", "--question", question, "--thread", thread, "--ask"],
        cwd=ROOT,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    lines: queue.Queue = queue.Queue()
    threading.Thread(target=lambda: [lines.put(line) for line in proc.stdout], daemon=True).start()
    seen: list[str] = []
    deadline = time.time() + 120
    while time.time() < deadline:
        try:
            line = lines.get(timeout=1)
        except queue.Empty:
            if proc.poll() is not None:
                break
            continue
        seen.append(line)
        if WAITING in line:
            return proc, "".join(seen)
    proc.kill()
    pytest.fail("orchestrate never reached the approval prompt:\n" + "".join(seen))


def ledger(thread: str) -> list[tuple[str, str]]:
    with get_engine().connect() as c:
        return [
            (r.run_id, r.purpose)
            for r in c.execute(
                text(
                    "SELECT run_id, purpose FROM llm_calls WHERE question_id = :q "
                    "AND status IN ('pending', 'ok', 'error') ORDER BY id"
                ),
                {"q": thread},
            )
        ]


def checkpoint_count(thread: str) -> int:
    from rflens.config import get_checkpointer_settings

    engine = create_engine(get_checkpointer_settings().sa_url())
    try:
        with engine.connect() as c:
            return c.execute(
                text("SELECT count(*) FROM checkpoints WHERE thread_id = :t"), {"t": thread}
            ).scalar_one()
    finally:
        engine.dispose()


def test_kill_at_the_approval_prompt_then_resume_in_a_new_process(
    sql_mcp_url, thread, tmp_path
) -> None:
    question = f"What is forty one plus one? {thread}"
    first = env_for([PLAN], sql_mcp_url, tmp_path, "first")
    proc, shown = start_and_wait(question, thread, first)
    try:
        assert "route:         sql" in shown and "estimated worst-case cost" in shown
        assert "approval needed: ask: ASK is set" in shown
        pid = int(re.search(rf"thread {thread}\s+pid (\d+)", shown).group(1))
    finally:
        hard_kill(proc, shown)
    assert proc.returncode != 0 and pid

    # the checkpoint survived, and the ledger holds the one planner call of the first process
    assert checkpoint_count(thread) >= 1
    before = ledger(thread)
    assert [purpose for _, purpose in before] == ["planner"]
    status = run_cli(["status", "--thread", thread], first)
    assert status.returncode == 0, status.stderr
    assert "next node(s):      ['approval']" in status.stdout
    assert "waiting for approval" in status.stdout and "1 run id(s)" in status.stdout

    # a brand-new process, its own run id, a fresh script for the calls that remain
    second = env_for([SQL, ANSWER], sql_mcp_url, tmp_path, "second")
    resumed = run_cli(["resume", "--thread", thread, "--decision", "approve"], second)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    out = resumed.stdout
    assert f"thread {thread}" in out and "status: answered" in out
    assert "The answer is 42 [S1]." in out and "Sources:" in out
    assert "[S1] sql-mcp query #" in out

    after = ledger(thread)
    assert [purpose for _, purpose in after].count("planner") == 1  # not planned a second time
    assert len({run_id for run_id, _ in after}) == 2  # two processes, two run ids
    assert {p for _, p in after} == {"planner", "sql-agent", "writer"}
    final = run_cli(["status", "--thread", thread], second)
    assert "none (finished)" in final.stdout and "2 run id(s)" in final.stdout
    assert "status: answered" in final.stdout


def test_cancelling_after_the_kill_ends_the_thread_without_more_model_calls(
    sql_mcp_url, thread, tmp_path
) -> None:
    first = env_for([PLAN], sql_mcp_url, tmp_path, "first")
    proc, shown = start_and_wait(f"What is forty one plus one? {thread}", thread, first)
    hard_kill(proc, shown)
    second = env_for([], sql_mcp_url, tmp_path, "second")  # nothing scripted: no call may happen
    resumed = run_cli(["resume", "--thread", thread, "--decision", "cancel"], second)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert "status: cancelled" in resumed.stdout
    assert [purpose for _, purpose in ledger(thread)] == ["planner"]


def test_a_thread_id_that_already_has_a_checkpoint_is_not_started_again(
    sql_mcp_url, thread, tmp_path
) -> None:
    first = env_for([PLAN], sql_mcp_url, tmp_path, "first")
    proc, shown = start_and_wait(f"What is forty one plus one? {thread}", thread, first)
    hard_kill(proc, shown)
    again = run_cli(["run", "--question", "anything", "--thread", thread], first)
    assert again.returncode == 2 and "already exists" in again.stderr
    assert [p for _, p in ledger(thread)] == ["planner"]  # no second planner call


def test_resuming_a_thread_that_does_not_exist_says_so(sql_mcp_url, tmp_path) -> None:
    env = env_for([], sql_mcp_url, tmp_path, "none")
    done = run_cli(["resume", "--thread", "no-such-thread-xyz", "--decision", "approve"], env)
    assert done.returncode == 2 and "no checkpoint for thread" in done.stderr


def test_without_a_terminal_the_thread_stays_saved_at_the_approval_step(
    sql_mcp_url, thread, tmp_path
) -> None:
    env = env_for([PLAN], sql_mcp_url, tmp_path, "first")
    done = subprocess.run(
        [
            *CLI,
            "run",
            "--question",
            f"What is forty one plus one? {thread}",
            "--thread",
            thread,
            "--ask",
        ],
        cwd=ROOT,
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert done.returncode == 3 and "stays saved at the approval step" in done.stderr
    assert checkpoint_count(thread) >= 1


# --- rflens ask: the one-shot command -----------------------------------------------------------

BAD = {"answer": "It was measured 777 times [S1]."}


def ask_cli(args: list[str], env: dict[str, str]) -> subprocess.CompletedProcess:
    return run_cli(["ask", *args], env)


def test_ask_prints_the_answer_sources_verification_and_cost_per_node(
    sql_mcp_url, thread, tmp_path
) -> None:
    env = env_for([PLAN, SQL, ANSWER], sql_mcp_url, tmp_path, "ask")
    done = ask_cli([f"What is forty one plus one? {thread}", "--thread", thread], env)
    assert done.returncode == 0, done.stdout + done.stderr
    out = done.stdout
    assert f"thread {thread}" in out and "status: answered" in out
    assert "The answer is 42 [S1]." in out and "Sources:" in out and "[S1] sql-mcp query #" in out
    assert "verification: ok=True" in out
    for node in ("planner", "sql_agent", "writer"):
        assert f"  {node}" in out and "calls=1" in out
    assert "writer draft" not in out  # one clean draft: nothing to explain
    assert [p for _, p in ledger(thread)] == ["planner", "sql-agent", "writer"]


def test_ask_show_drafts_prints_every_draft_with_its_failures(
    sql_mcp_url, thread, tmp_path
) -> None:
    env = env_for([PLAN, SQL, BAD, ANSWER], sql_mcp_url, tmp_path, "drafts")
    question = f"What is forty one plus one? {thread}"
    done = ask_cli([question, "--thread", thread, "--show-drafts"], env)
    assert done.returncode == 0, done.stdout + done.stderr
    out = done.stdout
    assert "--- writer draft 1 (verified ok: False) ---" in out
    assert "It was measured 777 times [S1]." in out and "failure: unsupported_number" in out
    assert "--- writer draft 2 (verified ok: True) ---" in out and "status: answered" in out


def test_drafts_are_shown_by_default_only_when_the_writer_needed_more_than_one(
    sql_mcp_url, thread, tmp_path
) -> None:
    env = env_for([PLAN, SQL, BAD, ANSWER], sql_mcp_url, tmp_path, "more")
    done = ask_cli([f"What is forty one plus one? {thread}", "--thread", thread], env)
    assert "--- writer draft 1" in done.stdout and "--- writer draft 2" in done.stdout


def test_ask_json_prints_one_document_and_keeps_the_rest_on_stderr(
    sql_mcp_url, thread, tmp_path
) -> None:
    env = env_for([PLAN, SQL, BAD, ANSWER], sql_mcp_url, tmp_path, "json")
    question = f"What is forty one plus one? {thread}"
    plain = ask_cli([question, "--thread", thread, "--json"], env)
    assert plain.returncode == 0, plain.stdout + plain.stderr
    doc = json.loads(plain.stdout)  # nothing but the document on stdout
    assert doc["thread_id"] == thread and doc["status"] == "answered" and doc["badge"] == "verified"
    assert doc["answer"] == "The answer is 42 [S1]." and doc["sources_text"].startswith("[S1]")
    assert doc["cost"]["calls"] == 4 and doc["verification"]["ok"]
    assert doc["source_items"][0]["tag"] == "S1" and "draft_history" not in doc
    assert "LLM run" in plain.stderr  # the gateway's run summary goes to stderr

    env2 = env_for([PLAN, SQL, BAD, ANSWER], sql_mcp_url, tmp_path, "json2")
    with_drafts = ask_cli([question, "--thread", thread + "-b", "--json", "--show-drafts"], env2)
    assert [d["ok"] for d in json.loads(with_drafts.stdout)["draft_history"]] == [False, True]
    with open_checkpointer() as saver:
        saver.delete_thread(thread + "-b")
    with get_engine().begin() as c:
        c.execute(text("DELETE FROM llm_calls WHERE question_id = :q"), {"q": thread + "-b"})


def test_ask_that_needs_approval_without_a_terminal_leaves_the_thread_saved(
    sql_mcp_url, thread, tmp_path
) -> None:
    env = env_for([PLAN], sql_mcp_url, tmp_path, "pending")
    done = subprocess.run(
        [*CLI, "ask", f"What is forty one plus one? {thread}", "--thread", thread, "--ask"],
        cwd=ROOT,
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert done.returncode == 3 and "waiting for approval" in done.stderr
    assert "ask: ASK is set" in done.stdout
    assert checkpoint_count(thread) >= 1
    resumed = run_cli(
        ["resume", "--thread", thread, "--decision", "approve"],
        env_for([SQL, ANSWER], sql_mcp_url, tmp_path, "pending2"),
    )
    assert "status: answered" in resumed.stdout


def test_a_thread_id_that_exists_is_not_asked_again(sql_mcp_url, thread, tmp_path) -> None:
    env = env_for([PLAN, SQL, ANSWER], sql_mcp_url, tmp_path, "twice")
    q = f"What is forty one plus one? {thread}"
    assert ask_cli([q, "--thread", thread], env).returncode == 0
    again = ask_cli([q, "--thread", thread], env)
    assert again.returncode == 2 and "already exists" in again.stderr


def test_the_rflens_command_is_installed_from_the_project_scripts() -> None:
    import tomllib

    scripts = tomllib.loads((ROOT / "pyproject.toml").read_text("utf-8"))["project"]["scripts"]
    assert scripts == {"rflens": "rflens.graph.cli:main"}
