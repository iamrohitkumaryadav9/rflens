"""Scripted kill-and-resume run (paid calls when the provider is a real one).

    make sql-mcp         # another terminal
    make docs-mcp        # another terminal (needs Ollama)
    LLM_ALLOW_PAID=true LLM_BUDGET_USD=<spent + cap> \\
        uv run python scripts/demo_kill_resume.py mixed-01

Starts `orchestrate` on a golden question with ASK=1, shows `orchestrate-status`, hard-kills the
process at the approval prompt, shows the status again, then resumes the thread in a brand-new
process (`orchestrate-resume`), shows its final answer (compared with the golden reference) and the
status once more. The checkpoints of the thread are deleted at the end; its llm_calls rows stay
(the ledger is an audit log).
"""

import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from rflens.eval.golden import load_golden
from rflens.graph.app import open_checkpointer

ROOT = Path(__file__).resolve().parents[1]
CLI = [sys.executable, "-u", "-m", "rflens.graph.cli"]
WAITING = "Resume it from any process"


def banner(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}", flush=True)


def cli(*args: str) -> str:
    done = subprocess.run([*CLI, *args], cwd=ROOT, capture_output=True, text=True, timeout=600)
    print(done.stdout, end="")
    if done.returncode != 0:
        print(done.stderr, end="")
    return done.stdout


def main(argv: list[str]) -> int:
    golden = {q.id: q for q in load_golden()}[argv[0] if argv else "mixed-01"]
    thread = f"demo-{golden.id}-{uuid.uuid4().hex[:6]}"
    banner(f"1. orchestrate (ASK=1) on {golden.id}: {golden.question}")
    proc = subprocess.Popen(
        [*CLI, "run", "--question", golden.question, "--thread", thread, "--ask"],
        cwd=ROOT,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    lines: queue.Queue = queue.Queue()
    threading.Thread(target=lambda: [lines.put(x) for x in proc.stdout], daemon=True).start()
    shown: list[str] = []
    deadline = time.time() + 300
    while time.time() < deadline:
        try:
            line = lines.get(timeout=1)
        except queue.Empty:
            if proc.poll() is not None:
                print("the process ended before the approval prompt")
                return 1
            continue
        print(line, end="")
        shown.append(line)
        if WAITING in line:
            break
    banner("2. orchestrate-status BEFORE the kill")
    cli("status", "--thread", thread)
    match = re.search(r"pid (\d+)", "".join(shown))
    banner(f"3. hard kill of pid {match.group(1) if match else proc.pid} (no cleanup code runs)")
    if match:
        os.kill(int(match.group(1)), signal.SIGTERM)  # the interpreter itself (TerminateProcess)
    proc.kill()
    proc.wait()
    print(f"killed, exit code {proc.returncode}")
    banner("4. orchestrate-status AFTER the kill")
    cli("status", "--thread", thread)
    banner("5. orchestrate-resume in a brand-new process (DECISION=approve)")
    cli("resume", "--thread", thread, "--decision", "approve")
    banner("6. orchestrate-status AFTER the resume")
    cli("status", "--thread", thread)
    with open_checkpointer() as saver:
        saver.delete_thread(thread)
    print(f"\ncheckpoints of {thread} deleted; its llm_calls rows stay in the ledger")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
