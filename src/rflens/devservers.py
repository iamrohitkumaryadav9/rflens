"""Start and stop the local dev servers by recorded PID, never by port, name or command line.

    make servers-up [SERVERS="sql-mcp docs-mcp api ui"]
    make servers-status
    make servers-down [SERVERS=...]

`up` starts each server as a detached child, writes its PID to run/<name>.pid (gitignored) and its
output to run/<name>.log. `down` stops exactly the PIDs in those files (the process tree of that
PID, which is what we started) and removes the files. A server whose port is already taken is not
started and nothing is stopped to make room: the command reports it, and `--port NAME=PORT` picks
another port (clients then need the matching URL, for example SQL_MCP_URL).
"""

import argparse
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = ROOT / "run"


@dataclass(frozen=True)
class Server:
    name: str
    port: int
    port_env: str | None  # environment variable that sets the port, or None for a CLI flag
    argv: tuple[str, ...]

    def command(self, port: int) -> list[str]:
        cmd = [sys.executable, *self.argv]
        if self.port_env is None:
            cmd += ["--server.port", str(port)]
        return cmd


SERVERS = {
    s.name: s
    for s in (
        Server("sql-mcp", 8102, "SQL_MCP_PORT", ("-m", "rflens.mcp.sql_server")),
        Server("docs-mcp", 8101, "DOCS_MCP_PORT", ("-m", "rflens.mcp.docs_server")),
        Server("api", 8000, "API_PORT", ("-m", "rflens.api.server")),
        Server(
            "ui",
            8502,
            None,
            ("-m", "streamlit", "run", "src/rflens/ui/app.py"),
        ),
    )
}
ORDER = ("sql-mcp", "docs-mcp", "api", "ui")


def pid_file(name: str, run_dir: Path = RUN_DIR) -> Path:
    return run_dir / f"{name}.pid"


def read_pid(name: str, run_dir: Path = RUN_DIR) -> int | None:
    try:
        return int(pid_file(name, run_dir).read_text().strip())
    except (FileNotFoundError, ValueError):
        return None


def is_alive(pid: int) -> bool:
    if sys.platform == "win32":
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH", "/FO", "CSV"],
            capture_output=True,
            text=True,
        ).stdout
        return f'"{pid}"' in out
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return not _is_zombie(pid)  # a killed child stays a zombie until its parent waits for it


def _is_zombie(pid: int) -> bool:
    """True for a process that has exited but is not reaped yet (state Z). kill(pid, 0) succeeds
    for it, so without this a stopped child of this very process would look alive."""
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as f:
            return f.read().rsplit(")", 1)[1].split()[0] == "Z"
    except (FileNotFoundError, IndexError):
        pass
    except OSError:
        return False
    out = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True)
    return out.stdout.strip().startswith("Z")


def port_in_use(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def suggest_port(taken: int, reserved: set[int] | None = None) -> int:
    """The next port above `taken` that no server of the set is assigned and that is free now."""
    reserved = (reserved or set()) | {s.port for s in SERVERS.values()}
    port = taken + 1
    while port in reserved or port_in_use(port):
        port += 1
    return port


def kill_tree(pid: int, force: bool = False) -> None:
    """Stop the process we started and the children it spawned (the venv launcher on Windows has
    one). Only this PID's tree: nothing is looked up by port, name or command line.

    On POSIX `start()` makes the server the leader of its own process group, so the group is the
    tree. A recorded PID that does not lead a group (it shares its parent's group) must not be
    signalled through killpg: that would raise ESRCH for it, or hit the parent's group. It gets
    the signal itself."""
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
        return
    sig = signal.SIGKILL if force else signal.SIGTERM
    try:
        if os.getpgid(pid) == pid:
            os.killpg(pid, sig)
        else:
            os.kill(pid, sig)
    except ProcessLookupError:
        pass  # already gone


def start(
    name: str,
    port: int | None = None,
    run_dir: Path = RUN_DIR,
    reserved: set[int] | None = None,
) -> str:
    server = SERVERS[name]
    port = port or server.port
    existing = read_pid(name, run_dir)
    if existing is not None and is_alive(existing):
        return f"{name}: already running (pid {existing}, recorded in {pid_file(name, run_dir)})"
    if port_in_use(port):
        return (
            f"{name}: port {port} is taken by a process this tool did not start; not starting, "
            f"nothing stopped. Pick another port: --port {name}={suggest_port(port, reserved)}"
        )
    run_dir.mkdir(exist_ok=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    if server.port_env:
        env[server.port_env] = str(port)
    log = (run_dir / f"{name}.log").open("ab")
    flags = (
        subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
        if sys.platform == "win32"
        else 0
    )
    proc = subprocess.Popen(
        server.command(port),
        cwd=ROOT,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=log,
        creationflags=flags,
        start_new_session=sys.platform != "win32",
    )
    pid_file(name, run_dir).write_text(str(proc.pid))
    return f"{name}: started pid {proc.pid} on port {port} (log run/{name}.log)"


def _wait_gone(pid: int, seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if not is_alive(pid):
            return True
        time.sleep(0.1)
    return not is_alive(pid)


def stop(name: str, run_dir: Path = RUN_DIR) -> str:
    pid = read_pid(name, run_dir)
    if pid is None:
        return f"{name}: no pid file, nothing to stop"
    if is_alive(pid):
        kill_tree(pid)
        if not _wait_gone(pid, 5.0):
            kill_tree(pid, force=True)  # it ignored the polite signal
            _wait_gone(pid, 5.0)
        if is_alive(pid):
            return f"{name}: pid {pid} is still running after SIGKILL; pid file kept"
        message = f"{name}: stopped pid {pid}"
    else:
        message = f"{name}: pid {pid} was already gone"
    pid_file(name, run_dir).unlink(missing_ok=True)
    return message


def status(name: str, run_dir: Path = RUN_DIR) -> str:
    pid = read_pid(name, run_dir)
    if pid is None:
        return f"{name}: not started by this tool"
    return f"{name}: pid {pid} {'running' if is_alive(pid) else 'gone (stale pid file)'}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["up", "down", "status"])
    parser.add_argument("names", nargs="*", help=f"default: {' '.join(ORDER)}")
    parser.add_argument("--port", action="append", default=[], metavar="NAME=PORT")
    args = parser.parse_args(argv)
    names = args.names or list(ORDER)
    unknown = [n for n in names if n not in SERVERS]
    if unknown:
        parser.error(f"unknown server(s): {', '.join(unknown)}")
    ports = {k: int(v) for k, v in (p.split("=", 1) for p in args.port)}
    code = 0
    for name in names if args.action != "down" else reversed(names):
        if args.action == "up":
            line = start(name, ports.get(name), reserved=set(ports.values()))
            code = code or (2 if "port" in line and "taken" in line else 0)
        elif args.action == "down":
            line = stop(name)
        else:
            line = status(name)
        print(line)
    return code


if __name__ == "__main__":
    sys.exit(main())
