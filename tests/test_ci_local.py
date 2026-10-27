"""make ci-local: the Docker harness that reproduces the CI job. These tests need no Docker; they
check that the harness reads the real workflow, builds the right environment and copy, and stops at
the first failing step. (`make ci-local` itself is the end-to-end check.)
"""

import subprocess
from pathlib import Path

import pytest
import yaml

from rflens.ci_local import (
    NEVER_COPY,
    WORKFLOW,
    load_job,
    main,
    needs_docker,
    not_run_notice,
    runner_env,
    snapshot,
    split_steps,
    steps_script,
)

STEPS = [
    "uv sync --frozen",
    "uv run ruff check .",
    "uv run ruff format --check .",
    "uv run python -m rflens.testdb",
    "uv run pytest",
]
DOCKER_STEPS = [
    "docker build -f analytics_runner/Dockerfile -t rflens-analytics:v1 .",
    "docker run --rm --network none --memory 512m --cpus 1 --read-only --tmpfs /tmp --cap-drop ALL "
    "--security-opt no-new-privileges -e PYTHONPATH=/app rflens-analytics:v1 "
    "python -m pytest -q -p no:cacheprovider /app/tests",
    "uv run pytest tests_docker -q -p no:cacheprovider",
]


def test_the_harness_reads_the_real_workflow() -> None:
    job = load_job()
    assert job["steps"] == [*STEPS, *DOCKER_STEPS]  # run: steps in order, YAML comments gone
    assert job["image"].startswith("pgvector/pgvector:")
    assert job["service_env"]["POSTGRES_DB"] == "rflens"
    assert {"TEST_DATABASE_URL", "DOCS_READER_PASSWORD", "SQL_READER_PASSWORD"} <= set(job["env"])
    assert {"ANALYTICS_READER_PASSWORD", "ANALYTICS_LOGGER_PASSWORD"} <= set(job["env"])
    assert "LLM_ALLOW_PAID" not in job["env"]  # CI never makes paid calls


def test_the_environment_is_the_workflow_env_plus_what_github_sets() -> None:
    env = runner_env(load_job(), "db-host")
    assert "@db-host:5432/rflens_test" in env["TEST_DATABASE_URL"]  # the service by name
    assert env["CI"] == "true" and env["GITHUB_ACTIONS"] == "true"
    # nothing a developer machine would add: no dev database, no keys, never paid calls
    assert not {"DATABASE_URL", "GEMINI_API_KEY", "LLM_ALLOW_PAID", "OLLAMA_BASE_URL"} & set(env)
    assert all(v.startswith("ci-only-") for k, v in env.items() if k.endswith("_PASSWORD"))


def test_the_steps_script_runs_in_order_and_stops_at_the_first_failure() -> None:
    script = steps_script(STEPS)
    positions = [script.index(f"\n{command}\n") for command in STEPS]
    assert positions == sorted(positions)
    assert script.count("; exit $rc; fi") == len(STEPS)  # every step stops the job on failure
    assert "::all steps passed::" in script and "python -m pip install -q --user uv" in script


def test_docker_steps_are_reported_as_not_run_never_dropped_silently() -> None:
    to_run, skipped = split_steps([*STEPS, *DOCKER_STEPS])
    assert to_run == STEPS and skipped == DOCKER_STEPS
    assert not any(needs_docker(s) for s in STEPS) and all(needs_docker(s) for s in DOCKER_STEPS)
    notice = "\n".join(not_run_notice(skipped))
    assert "NOT RUN: 3 workflow step(s)" in notice
    assert "each needs a Docker daemon and the ci-local container has none" in notice
    assert "make test-analytics" in notice and all(s in notice for s in skipped)
    assert not_run_notice([]) == []
    script = steps_script(to_run, skipped)
    assert "NOT RUN: 3 workflow step(s)" in script and "tests_docker" in script
    assert script.index("NOT RUN") < script.index("::all steps passed::")
    assert "\\ndocker build" not in script  # the docker steps are echoed, never executed
    assert script.count("; exit $rc; fi") == len(STEPS)


def git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *args],
        check=True,
        capture_output=True,
    )


def test_the_copy_is_the_working_tree_without_secrets_data_or_deleted_files(
    tmp_path: Path,
) -> None:
    repo, copy = tmp_path / "repo", tmp_path / "copy"
    (repo / "data").mkdir(parents=True)
    (repo / "src").mkdir()
    git(repo, "init", "-q")
    (repo / ".gitignore").write_text("ignored.txt\n.venv/\n", encoding="utf-8")
    (repo / "src" / "kept.py").write_text("tracked\n", encoding="utf-8")
    (repo / "deleted.py").write_text("tracked, then deleted\n", encoding="utf-8")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "first")
    (repo / "deleted.py").unlink()  # deleted in the working tree, still in HEAD
    (repo / "src" / "new_untracked.py").write_text("not committed yet\n", encoding="utf-8")
    (repo / "ignored.txt").write_text("git ignores this\n", encoding="utf-8")
    (repo / ".env").write_text("SECRET=1\n", encoding="utf-8")  # NOT in .gitignore here
    (repo / "data" / "big.zip").write_text("downloaded data\n", encoding="utf-8")
    count = snapshot(repo, copy)
    copied = sorted(p.relative_to(copy).as_posix() for p in copy.rglob("*") if p.is_file())
    assert copied == [".gitignore", "src/kept.py", "src/new_untracked.py"]
    assert count == 3
    assert {".env", "data", ".venv"} == NEVER_COPY


def test_a_tracked_data_or_env_file_is_still_not_copied(tmp_path: Path) -> None:
    repo, copy = tmp_path / "repo", tmp_path / "copy"
    (repo / "data").mkdir(parents=True)
    git(repo, "init", "-q")
    (repo / ".env").write_text("SECRET=1\n", encoding="utf-8")
    (repo / "data" / "x.csv").write_text("a\n", encoding="utf-8")
    (repo / "ok.txt").write_text("ok\n", encoding="utf-8")
    git(repo, "add", "-f", ".")  # someone committed them by mistake
    assert snapshot(repo, copy) == 1 and (copy / "ok.txt").exists()
    assert not (copy / ".env").exists() and not (copy / "data").exists()


def test_plan_mode_prints_the_steps_and_needs_no_docker(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["--plan"]) == 0
    out = capsys.readouterr().out
    assert "uv run pytest" in out and "fresh" in out and "names only" in out
    assert "NOT RUN: 3 workflow step(s)" in out and "tests_docker" in out
    assert "ci-only-" not in out  # env values are never printed, only their names


def test_the_workflow_actions_are_not_the_node_20_majors() -> None:
    """GitHub deprecates Node 20 actions. checkout switched to Node 24 in v5 and setup-uv in v7
    (their release notes); an exact pin like v10.3.2 is fine, a floating major may not exist."""
    steps = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"]["test"]["steps"]
    used = {s["uses"].split("@")[0]: s["uses"].split("@")[1] for s in steps if "uses" in s}
    major = {name: int(ref.lstrip("v").split(".")[0]) for name, ref in used.items()}
    assert major["actions/checkout"] >= 5
    assert major["astral-sh/setup-uv"] >= 7
