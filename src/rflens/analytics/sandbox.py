"""Run one analysis in a Docker container with no network, limited memory/CPU/processes/time.

    result = DockerSandbox(settings).run({"analysis": ..., "params": {...}}, csv_bytes)

The container is `docker run --rm --network none --read-only --cap-drop ALL
--security-opt no-new-privileges`, with /in (job.json, data.csv) mounted read-only, /out the only
writable mount (one fresh directory per job), a small /tmp tmpfs, and a non-root user. It holds
fixed code only (analytics_runner/), never code from a request. A job that runs past the time
limit is killed by the container name this call chose; nothing is ever stopped by pattern or by
port.
"""

import json
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from rflens.config import AnalyticsSettings, ArtifactSettings

ROOT = Path(__file__).resolve().parents[3]
MAX_RESULT_FILE = 1_000_000
MAX_PNG = 2_000_000
STDERR_CHARS = 600
KILL_WAIT_S = 20


@dataclass
class SandboxResult:
    status: Literal["ok", "timeout", "error"]
    payload: dict[str, Any] | None = None  # parsed /out/result.json
    png: bytes | None = None
    message: str = ""
    duration_ms: int = 0
    container: str = ""
    command: list[str] = field(default_factory=list)


def artifacts_dir(settings: AnalyticsSettings | ArtifactSettings) -> Path:
    """artifacts/ at the repository root (gitignored), or ANALYTICS_ARTIFACTS_DIR."""
    chosen = settings.analytics_artifacts_dir
    return Path(chosen) if chosen else ROOT / "artifacts"


def docker_command(
    settings: AnalyticsSettings, name: str, in_dir: Path, out_dir: Path
) -> list[str]:
    """The exact `docker run` command line of one job (a function of its inputs, so tests can
    assert the isolation flags without Docker)."""
    cmd = [
        "docker", "run", "--rm", "--name", name,
        "--network", "none",
        "--memory", settings.analytics_memory,
        "--memory-swap", settings.analytics_memory,
        "--cpus", settings.analytics_cpus,
        "--pids-limit", str(settings.analytics_pids),
        "--read-only",
        "--tmpfs", "/tmp:rw,noexec,nosuid,size=128m,mode=1777",
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "--mount", f"type=bind,source={in_dir},target=/in,readonly",
        "--mount", f"type=bind,source={out_dir},target=/out",
    ]  # fmt: skip
    if hasattr(os, "getuid"):  # POSIX: write /out as the caller; Windows uses the image's user
        cmd += ["--user", f"{os.getuid()}:{os.getgid()}"]
    cmd += [settings.analytics_image, "python", "-I", "/app/runner.py"]
    return cmd


class DockerSandbox:
    def __init__(self, settings: AnalyticsSettings):
        self.settings = settings

    def run(self, job: dict[str, Any], csv_bytes: bytes) -> SandboxResult:
        jobs = artifacts_dir(self.settings) / "_jobs"
        jobs.mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix="job-", dir=jobs))
        in_dir, out_dir = work / "in", work / "out"
        in_dir.mkdir()
        out_dir.mkdir()
        name = f"rflens-analytics-{uuid.uuid4().hex[:12]}"
        started = time.monotonic()

        def finish(**kw: Any) -> SandboxResult:
            return SandboxResult(
                duration_ms=int((time.monotonic() - started) * 1000), container=name, **kw
            )

        try:
            (in_dir / "data.csv").write_bytes(csv_bytes)
            (in_dir / "job.json").write_text(json.dumps(job), encoding="utf-8")
            if hasattr(os, "getuid"):
                out_dir.chmod(0o777)
            cmd = docker_command(self.settings, name, in_dir, out_dir)
            try:
                proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                )
            except OSError as e:
                return finish(status="error", message=f"cannot run docker: {e}")
            try:
                _, stderr = proc.communicate(timeout=self.settings.analytics_timeout_s)
            except subprocess.TimeoutExpired:
                # kill exactly the container this call started, then the docker client process
                subprocess.run(["docker", "kill", name], capture_output=True, check=False)
                try:  # the client returns once the killed container has exited (and --rm is done)
                    proc.communicate(timeout=KILL_WAIT_S)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.communicate()
                return finish(
                    status="timeout",
                    message=f"the analysis ran longer than {self.settings.analytics_timeout_s:g} s",
                )
            result_file = out_dir / "result.json"
            if not result_file.exists():
                return finish(
                    status="error",
                    message=f"no result (docker exit {proc.returncode}): "
                    f"{(stderr or '').strip()[-STDERR_CHARS:]}",
                )
            if result_file.stat().st_size > MAX_RESULT_FILE:
                return finish(status="error", message="the result file is too large")
            payload = json.loads(result_file.read_text(encoding="utf-8"))
            png_file = out_dir / "plot.png"
            png = None
            if png_file.exists():
                if png_file.stat().st_size > MAX_PNG:
                    return finish(status="error", message="the plot is too large")
                png = png_file.read_bytes()
            if proc.returncode not in (0, 3):
                tail = (stderr or "").strip()[-STDERR_CHARS:]
                return finish(status="error", message=f"docker exit {proc.returncode}: {tail}")
            return finish(status="ok", payload=payload, png=png)
        finally:
            shutil.rmtree(work, ignore_errors=True)
