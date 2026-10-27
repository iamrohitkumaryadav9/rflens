"""The real sandbox: a Docker daemon and the image `rflens-analytics:v1` (make analytics-image).

Two kinds of test:
- isolation: the exact flags of `docker_command` are applied to a probe command instead of the
  runner, and the probe tries what a hostile analysis could try (network, writing outside /out,
  root, capabilities, memory, processes, running forever);
- the whole stack: analytics-mcp with the real DockerSandbox over rows in the test database.

A missing daemon or image is a failure, not a skip: these are the checks that the isolation holds.
"""

import json
import re
import subprocess
import uuid
from collections.abc import Iterator
from pathlib import Path

import anyio
import pytest
from mcp import Client
from sqlalchemy import create_engine, text

from rflens.analytics import sandbox as sandbox_module
from rflens.analytics.sandbox import DockerSandbox, docker_command
from rflens.config import AnalyticsSettings, get_analytics_settings
from rflens.db import get_engine
from rflens.mcp.analytics_server import create_analytics_server

IMAGE = "rflens-analytics:v1"
FIRST_RUN = 910_001


def settings(tmp_path: Path, **kw) -> AnalyticsSettings:
    return AnalyticsSettings(
        analytics_reader_password="unused",
        analytics_logger_password="unused",
        analytics_artifacts_dir=str(tmp_path),
        analytics_image=IMAGE,
        **kw,
    )


@pytest.fixture(scope="session", autouse=True)
def docker_and_image():
    info = subprocess.run(["docker", "info"], capture_output=True, text=True, check=False)
    assert info.returncode == 0, "Docker daemon not available: " + info.stderr[-300:]
    image = subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True, check=False)
    assert image.returncode == 0, f"image {IMAGE} is missing: run `make analytics-image`"


def probe(tmp_path: Path, code: str, **kw) -> subprocess.CompletedProcess:
    """Run python `code` in a container with exactly the flags of an analysis job."""
    cfg = settings(tmp_path, **kw)
    in_dir, out_dir = tmp_path / "in", tmp_path / "out"
    in_dir.mkdir(exist_ok=True)
    out_dir.mkdir(exist_ok=True)
    out_dir.chmod(0o777)
    cmd = docker_command(cfg, f"rflens-probe-{uuid.uuid4().hex[:10]}", in_dir, out_dir)
    cmd = [*cmd[:-4], IMAGE, "python", "-I", "-c", code]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=False)


def test_no_network(tmp_path) -> None:
    code = (
        "import socket\n"
        "for host in ('1.1.1.1', '8.8.8.8'):\n"
        "    try:\n"
        "        socket.create_connection((host, 53), 3)\n"
        "        print('CONNECTED', host)\n"
        "    except OSError as e:\n"
        "        print('blocked', type(e).__name__)\n"
        "try:\n"
        "    socket.getaddrinfo('example.com', 80)\n"
        "    print('RESOLVED')\n"
        "except OSError:\n"
        "    print('no dns')\n"
    )
    out = probe(tmp_path, code).stdout
    assert "CONNECTED" not in out and "RESOLVED" not in out and "no dns" in out


def test_read_only_root_and_read_only_input_but_writable_output(tmp_path) -> None:
    code = (
        "import os\n"
        "def can_write(path):\n"
        "    try:\n"
        "        open(path, 'w').write('x')\n"
        "        return True\n"
        "    except OSError:\n"
        "        return False\n"
        "print('app', can_write('/app/evil.py'))\n"
        "print('usr', can_write('/usr/local/lib/python3.12/evil.py'))\n"
        "print('etc', can_write('/etc/evil'))\n"
        "print('in', can_write('/in/evil'))\n"
        "print('out', can_write('/out/ok.txt'))\n"
        "print('tmp', can_write('/tmp/ok.txt'))\n"
    )
    (tmp_path / "in").mkdir()
    (tmp_path / "in" / "x").write_text("x")
    out = dict(line.split() for line in probe(tmp_path, code).stdout.splitlines())
    assert out == {
        "app": "False",
        "usr": "False",
        "etc": "False",
        "in": "False",
        "out": "True",
        "tmp": "True",
    }
    assert (tmp_path / "out" / "ok.txt").read_text() == "x"  # only /out reaches the host


def test_tmp_is_not_executable_and_there_is_no_database_or_secret(tmp_path) -> None:
    code = (
        "import os, subprocess\n"
        "open('/tmp/run.sh', 'w').write('#!/bin/sh\\necho ran\\n')\n"
        "os.chmod('/tmp/run.sh', 0o755)\n"
        "try:\n"
        "    print('exec', subprocess.run(['/tmp/run.sh'], capture_output=True).returncode)\n"
        "except OSError as e:\n"
        "    print('exec blocked', type(e).__name__)\n"
        # GPG_KEY is set by the python base image (the public key id of the Python release signer)
        "print('env', sorted(k for k in os.environ if k != 'GPG_KEY' and any(s in k.upper() for s "
        "in ('DATABASE', 'PASSWORD', 'KEY', 'SECRET', 'TOKEN', 'PG'))))\n"
    )
    out = probe(tmp_path, code).stdout
    assert "exec blocked" in out or re.search(r"exec (126|1|13)\b", out), out
    assert "env []" in out, out


def test_not_root_and_no_capabilities_and_no_new_privileges(tmp_path) -> None:
    code = (
        "import os\n"
        "print('uid', os.getuid())\n"
        "status = dict(l.split(':\\t') for l in open('/proc/self/status').read().splitlines() "
        "if ':\\t' in l)\n"
        "print('capeff', status['CapEff'].strip())\n"
        "print('capbnd', status['CapBnd'].strip())\n"
        "print('nnp', status['NoNewPrivs'].strip())\n"
    )
    out = dict(line.split() for line in probe(tmp_path, code).stdout.splitlines())
    assert out["uid"] != "0"
    assert set(out["capeff"]) == {"0"} and set(out["capbnd"]) == {"0"}
    assert out["nnp"] == "1"


def test_memory_limit_kills_a_greedy_job(tmp_path) -> None:
    res = probe(
        tmp_path,
        "x = bytearray(900 * 1024 * 1024)\nx[::4096] = b'a' * len(x[::4096])\nprint('ALLOCATED')",
    )
    assert "ALLOCATED" not in res.stdout
    assert res.returncode in (137, 1)  # OOM kill, or MemoryError


def test_process_limit_stops_a_fork_bomb(tmp_path) -> None:
    code = (
        "import os, time\n"
        "pids = []\n"
        "try:\n"
        "    for _ in range(400):\n"
        "        pid = os.fork()\n"
        "        if pid == 0:\n"
        "            time.sleep(8); os._exit(0)\n"
        "        pids.append(pid)\n"
        "    print('FORKED', len(pids))\n"
        "except OSError as e:\n"
        "    print('limited at', len(pids), type(e).__name__)\n"
        "finally:\n"
        "    for p in pids:\n"
        "        os.kill(p, 9)\n"
    )
    out = probe(tmp_path, code).stdout
    assert "FORKED" not in out and "limited at" in out
    assert int(re.search(r"limited at (\d+)", out).group(1)) < 80


def test_a_job_that_runs_too_long_is_killed_by_its_container_name(tmp_path, monkeypatch) -> None:
    seen: dict = {}
    real = sandbox_module.docker_command

    def sleeper(cfg, name, in_dir, out_dir):
        seen["name"] = name
        return [*real(cfg, name, in_dir, out_dir)[:-4], IMAGE, "python", "-I", "-c",
                "import time; time.sleep(120)"]  # fmt: skip

    monkeypatch.setattr(sandbox_module, "docker_command", sleeper)
    result = DockerSandbox(settings(tmp_path, analytics_timeout_s=4)).run(
        {"analysis": "describe", "params": {}}, b"run_id,label,run_ts\n"
    )
    assert result.status == "timeout" and "4 s" in result.message
    assert result.duration_ms < 60_000
    left = subprocess.run(
        ["docker", "ps", "-a", "--filter", f"name={seen['name']}", "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert left == ""  # the container is gone (killed, and --rm removed it)
    assert list((tmp_path / "_jobs").iterdir()) == []  # the job directory was removed


# --- the real runner ------------------------------------------------------------------------


def csv_for(n: int = 60) -> bytes:
    rows = ["run_id,label,run_ts,f1,f2"]
    for i in range(n):
        label = "fail" if i >= 40 else "pass"
        rows.append(f"{i + 1},{label},2008-08-01 10:00:00,{i + 1.0},{(i * 7) % 11}.0")
    return ("\n".join(rows) + "\n").encode()


def test_the_real_runner_in_the_real_sandbox(tmp_path) -> None:
    box = DockerSandbox(settings(tmp_path))
    params = {"feature": 1, "filters": {"labels": None, "conditions": []}}
    res = box.run({"analysis": "describe", "params": params}, csv_for())
    assert res.status == "ok", res.message
    result = res.payload["result"]
    assert res.payload["ok"] and result["n"] == 60 and result["mean"] == 30.5
    assert result["median"] == 30.5 and result["min"] == 1.0 and result["max"] == 60.0
    plot = box.run(
        {
            "analysis": "plot",
            "params": {
                "kind": "histogram",
                "feature": 1,
                "bins": 10,
                "by_label": True,
                "filters": {"labels": None, "conditions": []},
            },
        },
        csv_for(),
    )
    assert plot.status == "ok" and plot.png and plot.png[:8] == b"\x89PNG\r\n\x1a\n"
    assert list((tmp_path / "_jobs").iterdir()) == []


def test_an_extract_that_is_not_a_csv_is_an_error_not_a_crash(tmp_path) -> None:
    res = DockerSandbox(settings(tmp_path)).run(
        {"analysis": "describe", "params": {"feature": 1}}, b"nonsense\n"
    )
    assert res.status == "ok" and res.payload["ok"] is False
    assert res.payload["error"]["code"] == "bad_extract"


# --- the whole stack: analytics-mcp, real sandbox, rows in the test database ----------------


@pytest.fixture
def secom_rows() -> Iterator[None]:
    with get_engine().begin() as c:
        c.execute(text("DELETE FROM secom_runs WHERE run_id >= :a"), {"a": FIRST_RUN})
        for i in range(60):
            run_id = FIRST_RUN + i
            c.execute(
                text("INSERT INTO secom_runs (run_id, label, run_ts) VALUES (:i, :l, :t)"),
                {"i": run_id, "l": "fail" if i >= 40 else "pass", "t": "2008-08-02 10:00:00"},
            )
            for idx in (1, 2):
                c.execute(
                    text("INSERT INTO secom_measurements VALUES (:r, :f, :v)"),
                    {"r": run_id, "f": idx, "v": float(i + 1) if idx == 1 else float((i * 7) % 11)},
                )
    yield
    with get_engine().begin() as c:
        c.execute(text("DELETE FROM secom_runs WHERE run_id >= :a"), {"a": FIRST_RUN})


def test_mcp_to_container_to_artifact(secom_rows, tmp_path) -> None:
    cfg = get_analytics_settings()
    cfg = cfg.model_copy(
        update={"analytics_artifacts_dir": str(tmp_path), "analytics_image": IMAGE}
    )
    reader = create_engine(cfg.reader_url())
    logger = create_engine(cfg.logger_url())
    question = f"pytest-analytics-docker-{uuid.uuid4().hex}"
    try:
        server = create_analytics_server(reader.connect, logger.connect, DockerSandbox(cfg), cfg)

        async def go(tool, args):
            async with Client(server) as client:
                return await client.call_tool(tool, args)

        out = anyio.run(go, "group_compare", {"params": {"feature": 1}, "question": question})
        assert not out.is_error, out.content
        body = out.structured_content
        assert body["result"]["higher_in"] == "fail" and body["result"]["groups"]["pass"]["n"] == 40
        assert body["n_runs_used"] == 60 and body["versions"]["scipy"]
        plot = anyio.run(
            go,
            "plot",
            {"params": {"kind": "box_by_label", "feature": 1}, "question": question},
        )
        art = plot.structured_content["artifacts"][0]
        assert (tmp_path / f"{art['id']}.png").read_bytes()[:4] == b"\x89PNG"
        refused = anyio.run(
            go, "group_compare", {"params": {"feature": 1, "min_n": 50}, "question": question}
        )
        assert refused.is_error and "rejected (too_few_values)" in refused.content[0].text
        with get_engine().connect() as c:
            statuses = [
                r[0]
                for r in c.execute(
                    text("SELECT status FROM analytics_call_log WHERE question = :q ORDER BY id"),
                    {"q": question},
                )
            ]
        assert statuses == ["ok", "ok", "rejected"]
        assert json.loads((tmp_path / f"{body['analysis_id']}.json").read_text())["call_id"]
    finally:
        with get_engine().begin() as c:
            c.execute(text("DELETE FROM analytics_call_log WHERE question = :q"), {"q": question})
        reader.dispose()
        logger.dispose()
