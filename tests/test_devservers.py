"""devservers: stop exactly the recorded PID, never anything found by port, name or command line."""

import socket
import subprocess
import sys
import time

import pytest

from rflens import devservers
from rflens.devservers import SERVERS, Server

SLEEPER = (sys.executable, "-c", "import time; time.sleep(120)")


def sleeper() -> subprocess.Popen:
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
    return subprocess.Popen(SLEEPER, creationflags=flags)


@pytest.fixture
def run_dir(tmp_path):
    return tmp_path / "run"


def test_stop_ends_only_the_recorded_pid(run_dir) -> None:
    ours, bystander = sleeper(), sleeper()  # same command line, same name: only ours is recorded
    try:
        run_dir.mkdir()
        devservers.pid_file("api", run_dir).write_text(str(ours.pid))
        assert devservers.status("api", run_dir) == f"api: pid {ours.pid} running"
        assert devservers.stop("api", run_dir) == f"api: stopped pid {ours.pid}"
        ours.wait(timeout=10)
        assert bystander.poll() is None, "a process that was not recorded must keep running"
        assert not devservers.pid_file("api", run_dir).exists()
    finally:
        for p in (ours, bystander):
            if p.poll() is None:
                p.kill()
            p.wait()


def test_stop_without_a_pid_file_or_with_a_stale_one_touches_nothing(run_dir) -> None:
    other = sleeper()
    try:
        assert devservers.stop("api", run_dir) == "api: no pid file, nothing to stop"
        run_dir.mkdir()
        gone = sleeper()
        gone.kill()
        gone.wait()
        devservers.pid_file("api", run_dir).write_text(str(gone.pid))
        assert "stale" in devservers.status("api", run_dir)
        assert devservers.stop("api", run_dir) == f"api: pid {gone.pid} was already gone"
        assert other.poll() is None
    finally:
        other.kill()
        other.wait()


def test_a_taken_port_is_reported_and_nothing_is_started_or_stopped(run_dir, monkeypatch) -> None:
    blocker = socket.socket()
    blocker.bind(("127.0.0.1", 0))
    blocker.listen()
    port = blocker.getsockname()[1]
    started = []
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: started.append(a))
    try:
        line = devservers.start("api", port, run_dir)
    finally:
        blocker.close()
    assert f"port {port} is taken by a process this tool did not start" in line
    assert f"--port api={port + 1}" in line
    assert started == [] and not devservers.pid_file("api", run_dir).exists()


def test_up_records_the_pid_and_down_stops_it(run_dir, monkeypatch) -> None:
    monkeypatch.setitem(
        SERVERS, "demo", Server("demo", 59999, "DEMO_PORT", ("-c", "import time; time.sleep(120)"))
    )
    line = devservers.start("demo", 0 or None, run_dir)
    pid = devservers.read_pid("demo", run_dir)
    try:
        assert pid and f"started pid {pid}" in line
        assert devservers.is_alive(pid)
        again = devservers.start("demo", None, run_dir)
        assert "already running" in again and devservers.read_pid("demo", run_dir) == pid
    finally:
        message = devservers.stop("demo", run_dir)
    assert message == f"demo: stopped pid {pid}"
    for _ in range(40):
        if not devservers.is_alive(pid):
            break
        time.sleep(0.25)
    assert not devservers.is_alive(pid)


def test_the_run_directory_is_gitignored_and_no_script_stops_processes_by_port() -> None:
    root = devservers.ROOT
    assert "run/" in (root / ".gitignore").read_text().splitlines()
    forbidden = ("Get-NetTCPConnection", "OwningProcess", "lsof", "fuser", "pkill", "killall")
    for path in [root / "Makefile", *(root / "scripts").glob("*.py"), root / "README.md"]:
        text = path.read_text(encoding="utf-8")
        assert not [f for f in forbidden if f in text], path.name


def test_the_suggested_port_is_free_and_not_assigned_to_another_server(monkeypatch) -> None:
    assert SERVERS["docs-mcp"].port + 1 == SERVERS["sql-mcp"].port  # the naive +1 would collide
    busy = {SERVERS["docs-mcp"].port}
    monkeypatch.setattr(devservers, "port_in_use", lambda port: port in busy)
    suggestion = devservers.suggest_port(SERVERS["docs-mcp"].port)
    assigned = {s.port for s in SERVERS.values()}
    assert suggestion not in assigned and suggestion not in busy
    assert suggestion == SERVERS["analytics-mcp"].port + 1  # 8102 and 8103 are assigned
    busy.add(suggestion)  # and when that one is busy too, it moves on
    again = devservers.suggest_port(SERVERS["docs-mcp"].port)
    assert again not in assigned | busy and again > suggestion
    # a port requested for another server in this invocation is reserved as well
    assert devservers.suggest_port(8100, reserved={8101}) not in {8101, 8102, 8103}


def test_the_message_for_a_taken_port_carries_that_suggestion(run_dir, monkeypatch) -> None:
    busy = {SERVERS["docs-mcp"].port}
    monkeypatch.setattr(devservers, "port_in_use", lambda port: port in busy)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("must not start"))
    line = devservers.start("docs-mcp", None, run_dir)
    assert "--port docs-mcp=8104" in line and "docs-mcp=8102" not in line


STARTER = """
import sys
from pathlib import Path
from rflens import devservers
from rflens.devservers import SERVERS, Server
SERVERS["demo"] = Server("demo", 59998, "DEMO_PORT", ("-c", "import time; time.sleep(120)"))
print(devservers.start("demo", None, Path(sys.argv[1])))
"""


def test_a_server_started_by_another_process_is_stopped_by_its_recorded_pid(run_dir) -> None:
    """The real use case: `servers-up` exits, and `servers-down` runs later in a new process."""
    started = subprocess.run(
        [sys.executable, "-c", STARTER, str(run_dir)], capture_output=True, text=True, timeout=60
    )
    assert started.returncode == 0, started.stderr
    pid = devservers.read_pid("demo", run_dir)
    try:
        assert pid and f"started pid {pid}" in started.stdout
        assert devservers.is_alive(pid), "the server must outlive the process that started it"
        assert devservers.stop("demo", run_dir) == f"demo: stopped pid {pid}"
        assert not devservers.is_alive(pid)
        assert not devservers.pid_file("demo", run_dir).exists()
    finally:
        if pid and devservers.is_alive(pid):
            devservers.kill_tree(pid, force=True)


def test_a_zombie_is_not_alive_and_a_live_child_is() -> None:
    child = sleeper()
    try:
        assert devservers.is_alive(child.pid)
        child.kill()  # exited, but not waited for yet: a zombie on POSIX
        for _ in range(50):
            if not devservers.is_alive(child.pid):
                break
            time.sleep(0.1)
        assert not devservers.is_alive(child.pid)
    finally:
        child.wait()
