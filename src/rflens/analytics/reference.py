"""Reference values for the golden stats questions, computed independently of the analyses.

    python -m rflens.analytics.reference          # prints the JSON

Reads the full SECOM extract through the read-only analytics role, then runs
analytics_runner/reference.py in the sandbox image (same isolation flags as an analysis job). That
script shares no code with analytics_runner/analyses.py: it parses the extract itself, calls
scipy.stats directly with explicit arguments, recomputes every Mann-Whitney p-value from the
textbook formula and stops if the two disagree. `make golden-stats-refresh` stores the values in
eval/golden.yaml.
"""

import json
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine

from rflens.analytics.extract import build_extract
from rflens.analytics.sandbox import artifacts_dir, docker_command
from rflens.config import AnalyticsSettings, get_analytics_settings

TIMEOUT_S = 300  # the textbook recomputation of 590 features takes a few seconds; be generous


def run_reference(settings: AnalyticsSettings | None = None) -> dict[str, Any]:
    cfg = settings or get_analytics_settings()
    engine = create_engine(cfg.reader_url())
    try:
        with engine.connect() as conn:
            extract = build_extract(conn, None)
    finally:
        engine.dispose()
    base = artifacts_dir(cfg) / "_jobs"
    base.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="reference-", dir=base))
    try:
        in_dir, out_dir = work / "in", work / "out"
        in_dir.mkdir()
        out_dir.mkdir()
        (in_dir / "data.csv").write_bytes(extract.csv_bytes)
        name = f"rflens-reference-{uuid.uuid4().hex[:10]}"
        cmd = docker_command(cfg, name, in_dir, out_dir)
        cmd = [*cmd[:-4], cfg.analytics_image, "python", "-I", "/app/reference.py", "/in/data.csv"]
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_S, check=False)
        if done.returncode != 0:
            raise RuntimeError(f"reference.py failed ({done.returncode}): {done.stderr[-500:]}")
        reference = json.loads(done.stdout)
        reference["extract_sha256"] = extract.sha256
        reference["n_runs"] = extract.n_runs
        return reference
    finally:
        import shutil

        shutil.rmtree(work, ignore_errors=True)


def main() -> None:
    print(json.dumps(run_reference(), indent=1))


if __name__ == "__main__":
    main()
