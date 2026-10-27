"""make ci-local: run the CI job (.github/workflows/ci.yml) locally, in Linux containers.

    make ci-local            # needs Docker
    uv run python -m rflens.ci_local --plan     # print what it would run; needs no Docker

What it reproduces, so that a test that passes only on a developer machine fails here first:
- a fresh Postgres from the workflow's `services:` block (same image and env), new for every run;
- a clean copy of the working tree: tracked and untracked files that git does not ignore, never
  `.env`, `data/` or `.venv`, so nothing depends on the developer's secrets or downloaded data;
- the workflow's job `env:` (plus CI=true and GITHUB_ACTIONS=true, as GitHub sets them), and
  nothing else: no .env file, no Ollama, no sql-mcp or docs-mcp, no API keys;
- the workflow's `run:` steps, in order, stopping at the first failure like Actions does.

Steps that need a Docker daemon (the analytics sandbox container tests) cannot run inside the
ci-local container, which has none. They are NOT silently skipped: the plan, the end of the run
and the final verdict list them as NOT RUN, and `make test-analytics` runs them.

It does not run `uses:` steps (checkout, setup-uv): the copy replaces checkout and uv is installed
with pip. The runner is python:3.12-bookworm (Debian, with git), not GitHub's ubuntu image.
Run it before pushing changes to tests, the workflow, migrations or settings.
"""

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
NETWORK = "rflens-ci-local"
RUNNER_IMAGE = "python:3.12-bookworm"
NEVER_COPY = {".env", "data", ".venv"}  # even if git does not ignore them
JOB = "test"


class CiLocalError(Exception):
    pass


def load_job(path: Path = WORKFLOW, job: str = JOB) -> dict:
    """The parts of the workflow job that matter: service image and env, job env, `run:` steps."""
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))["jobs"][job]
    postgres = spec["services"]["postgres"]
    return {
        "image": postgres["image"],
        "service_env": {k: str(v) for k, v in (postgres.get("env") or {}).items()},
        "env": {k: str(v) for k, v in (spec.get("env") or {}).items()},
        "steps": [s["run"].strip() for s in spec["steps"] if "run" in s],
    }


def runner_env(job: dict, db_host: str) -> dict[str, str]:
    """The environment of the test steps: the job env, with the service reachable by name, and
    the variables GitHub Actions always sets."""
    env = {k: v.replace("localhost:5432", f"{db_host}:5432") for k, v in job["env"].items()}
    env["CI"] = "true"
    env["GITHUB_ACTIONS"] = "true"
    return env


NOT_RUN_REASON = "each needs a Docker daemon and the ci-local container has none"


def needs_docker(command: str) -> bool:
    """A step that starts containers (docker ...) or runs the tests that do (tests_docker/)."""
    return command.startswith("docker ") or "tests_docker" in command


def split_steps(steps: list[str]) -> tuple[list[str], list[str]]:
    """(steps ci-local runs, steps it cannot run). Never silent: run() prints the second list."""
    return [s for s in steps if not needs_docker(s)], [s for s in steps if needs_docker(s)]


def not_run_notice(skipped: list[str]) -> list[str]:
    if not skipped:
        return []
    return [
        f"NOT RUN: {len(skipped)} workflow step(s), because {NOT_RUN_REASON}:",
        *[f"  - {s}" for s in skipped],
        "  Run them with `make test-analytics`; the GitHub workflow runs them.",
    ]


def steps_script(steps: list[str], not_run: list[str] | None = None) -> str:
    """A bash script that runs the steps in order and stops at the first failure. `not_run` steps
    are not executed; the script prints a notice for them before it ends."""
    lines = [
        "set -u",
        "export HOME=/tmp PATH=/tmp/.local/bin:$PATH UV_PROJECT_ENVIRONMENT=/tmp/venv",
        "export UV_LINK_MODE=copy PIP_DISABLE_PIP_VERSION_CHECK=1",
        "python -m pip install -q --user uv || exit 90",
        "n=0",
    ]
    for command in steps:
        lines.append(f'n=$((n+1)); echo; echo "::step $n:: {command}"')
        lines.append(f"{command}\nrc=$?")
        lines.append('if [ $rc -ne 0 ]; then echo "::failed step $n (exit $rc)::"; exit $rc; fi')
    for notice in not_run_notice(not_run or []):
        lines.append(f"echo {shlex.quote(notice)}")
    lines.append('echo "::all steps passed::"')
    return "\n".join(lines) + "\n"


def tracked_and_untracked(repo: Path) -> list[str]:
    out = subprocess.run(
        ["git", "-C", str(repo), "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        capture_output=True,
        check=True,
    ).stdout.decode("utf-8")
    return sorted({p for p in out.split("\0") if p})


def snapshot(repo: Path, dest: Path) -> int:
    """Copy the working tree as it would be committed: files git tracks or would add, minus
    anything deleted in the working tree and minus NEVER_COPY. Returns the number of files."""
    copied = 0
    for rel in tracked_and_untracked(repo):
        if Path(rel).parts[0] in NEVER_COPY:
            continue
        source = repo / rel
        if not source.is_file():  # tracked but deleted in the working tree
            continue
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied += 1
    return copied


def docker(*args: str, check: bool = True, quiet: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", *args],
        capture_output=quiet,
        text=True,
        check=check,
    )


def docker_available() -> bool:
    try:
        return docker("info", check=False).returncode == 0
    except OSError:
        return False


def start_database(job: dict, name: str) -> None:
    docker("network", "create", NETWORK, check=False)
    env_args = [a for k, v in job["service_env"].items() for a in ("-e", f"{k}={v}")]
    docker("run", "-d", "--name", name, "--network", NETWORK, *env_args, job["image"])
    user = job["service_env"].get("POSTGRES_USER", "postgres")
    database = job["service_env"].get("POSTGRES_DB", user)
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        time.sleep(2)
        probe = docker("exec", name, "psql", "-U", user, "-d", database, "-tAc", "select 1",
                       check=False)  # fmt: skip
        if probe.returncode == 0 and probe.stdout.strip() == "1":
            return
    raise CiLocalError(f"the {job['image']} container did not accept connections in 120 s")


def run(plan_only: bool = False) -> int:
    job = load_job()
    db_name = f"rflens-ci-local-pg-{uuid.uuid4().hex[:6]}"
    env = runner_env(job, db_name)
    to_run, skipped = split_steps(job["steps"])
    script = steps_script(to_run, skipped)
    print(f"workflow: {WORKFLOW.relative_to(REPO_ROOT)} (job '{JOB}')")
    print(f"database: {job['image']}, fresh; runner: {RUNNER_IMAGE}")
    print("env passed (names only): " + ", ".join(sorted(env)))
    print("steps:")
    for i, command in enumerate(to_run, 1):
        print(f"  {i}. {command}")
    print("\n".join(not_run_notice(skipped)))
    if plan_only:
        return 0
    if not docker_available():
        print("ERROR: Docker is not available (is Docker running?)", file=sys.stderr)
        return 2

    work = Path(tempfile.mkdtemp(prefix="rflens-ci-work-"))
    support = Path(tempfile.mkdtemp(prefix="rflens-ci-steps-"))
    status = 1
    try:
        files = snapshot(REPO_ROOT, work)
        (support / "steps.sh").write_text(script, encoding="utf-8", newline="\n")
        print(f"\nclean copy of the working tree: {files} files (no .env, no data/, no .venv)")
        start_database(job, db_name)
        command = [
            "docker", "run", "--rm", "--network", NETWORK,
            "-v", f"{work}:/work", "-v", f"{support}:/ci:ro", "-w", "/work",
            *[a for k, v in env.items() for a in ("-e", f"{k}={v}")],
        ]  # fmt: skip
        if hasattr(os, "getuid"):  # keep files in the copy owned by the caller on Linux and macOS
            command += ["--user", f"{os.getuid()}:{os.getgid()}"]
        started = time.monotonic()
        status = subprocess.run([*command, RUNNER_IMAGE, "bash", "/ci/steps.sh"]).returncode
        elapsed = time.monotonic() - started
        verdict = "PASSED" if status == 0 else "FAILED"
        print(f"\nci-local: {verdict} in {elapsed:.0f} s")
        if skipped:
            print(f"ci-local: {len(skipped)} workflow step(s) NOT RUN, because {NOT_RUN_REASON}:")
            print("\n".join(f"  - {s}" for s in skipped))
            print("  They are not covered by this result: `make test-analytics` runs them.")
    except (CiLocalError, subprocess.CalledProcessError, OSError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        status = 2
    finally:
        docker("rm", "-f", db_name, check=False)
        docker("network", "rm", NETWORK, check=False)
        shutil.rmtree(work, ignore_errors=True)
        shutil.rmtree(support, ignore_errors=True)
    return status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the CI job locally in Docker")
    parser.add_argument("--plan", action="store_true", help="print the plan, run nothing")
    args = parser.parse_args(argv)
    return run(plan_only=args.plan)


if __name__ == "__main__":
    raise SystemExit(main())
