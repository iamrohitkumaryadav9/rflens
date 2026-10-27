"""Shared pieces of the analytics tests: a fake sandbox with canned results, and synthetic SECOM
rows committed to the test database (the reader role only sees committed rows)."""

from collections.abc import Iterator
from typing import Any

import pytest
from sqlalchemy import text

from rflens.analytics.sandbox import SandboxResult
from rflens.db import get_engine

FIRST_RUN = 900_001
N_RUNS = 60

COMPARE_60 = {
    "feature": 60,
    "groups": {
        "pass": {"n": 1463, "n_missing": 3, "median": 0.9, "mean": 2.9},
        "fail": {"n": 97, "n_missing": 7, "median": 1.5, "mean": 4.6},
    },
    "alpha": 0.05,
    "test": "mann_whitney_u (two-sided, asymptotic)",
    "statistic_u_fail": 97_000.0,
    "p_value": 3.4e-10,
    "rank_biserial": 0.384,
    "auc_fail_greater_than_pass": 0.692,
    "higher_in": "fail",
    "effect_size": {"name": "rank_biserial", "value": 0.384},
    "significant_at_alpha": True,
}
RANKING = {
    "method": "mann_whitney_u per feature",
    "positive_group": "fail",
    "alpha": 0.05,
    "n_tested": 4,
    "n_excluded": 1,
    "excluded_by_reason": {"constant": 1},
    "excluded": [{"feature": 9, "reason": "constant"}],
    "n_significant_q_below_alpha": 2,
    "top": [
        {
            "rank": 1,
            "feature": 60,
            "n_pass": 1463,
            "n_fail": 97,
            "missing_share": 0.0038,
            "statistic_u_fail": 97_000.0,
            "p_value": 3.4e-10,
            "q_value": 1.0e-7,
            "rank_biserial": 0.384,
            "higher_in": "fail",
        },
        {
            "rank": 2,
            "feature": 104,
            "n_pass": 1460,
            "n_fail": 104,
            "missing_share": 0.001,
            "statistic_u_fail": 90_000.0,
            "p_value": 0.0004,
            "q_value": 0.012,
            "rank_biserial": 0.3,
            "higher_in": "fail",
        },
    ],
}


def payload(analysis: str, result: dict[str, Any], used: int = N_RUNS) -> dict[str, Any]:
    return {
        "ok": True,
        "analysis": analysis,
        "n_runs_in_extract": N_RUNS,
        "n_runs_used": used,
        "missing_policy": "dropped pairwise per feature, never imputed",
        "versions": {"python": "3.12", "numpy": "x", "scipy": "y"},
        "result": result,
    }


def ok_outcome(
    analysis: str,
    result: dict[str, Any] | None = None,
    png: bytes | None = None,
    used: int = N_RUNS,
) -> SandboxResult:
    return SandboxResult(
        status="ok", payload=payload(analysis, result or {"n": 42, "p_value": 0.01}, used), png=png
    )


class FakeSandbox:
    """Stands in for DockerSandbox: records the job and extract, returns a canned outcome. With
    `by_analysis`, the outcome depends on the analysis asked for."""

    def __init__(self, outcome: SandboxResult | None = None, by_analysis: dict | None = None):
        self.outcome = outcome
        self.by_analysis = by_analysis or {}
        self.jobs: list[dict] = []
        self.extracts: list[bytes] = []

    def run(self, job, csv_bytes):
        self.jobs.append(job)
        self.extracts.append(csv_bytes)
        if job["analysis"] in self.by_analysis:
            return self.by_analysis[job["analysis"]]
        return self.outcome or ok_outcome(job["analysis"])


@pytest.fixture(scope="module")
def secom_rows() -> Iterator[None]:
    """60 runs (40 pass, 20 fail), features 1..4; feature 3 is missing for every 3rd run and
    feature 4 only has values for the first 10. The rows are deleted afterwards."""
    ids = list(range(FIRST_RUN, FIRST_RUN + N_RUNS))
    with get_engine().begin() as c:
        c.execute(text("DELETE FROM secom_runs WHERE run_id >= :a"), {"a": FIRST_RUN})
        for i, run_id in enumerate(ids):
            c.execute(
                text("INSERT INTO secom_runs (run_id, label, run_ts) VALUES (:id, :label, :ts)"),
                {
                    "id": run_id,
                    "label": "fail" if i >= 40 else "pass",
                    "ts": f"2008-08-{1 + i % 28:02d} 10:30:00",
                },
            )
            for idx in (1, 2, 3, 4):
                if (idx == 3 and i % 3 == 0) or (idx == 4 and i >= 10):
                    continue  # missing: no row, or a NULL value
                c.execute(
                    text(
                        "INSERT INTO secom_measurements (run_id, feature_idx, value) "
                        "VALUES (:r, :f, :v)"
                    ),
                    {"r": run_id, "f": idx, "v": i * 10 + idx + 0.5},
                )
        c.execute(
            text(
                "INSERT INTO secom_measurements (run_id, feature_idx, value) VALUES (:r, 3, NULL)"
            ),
            {"r": ids[0]},
        )
    yield
    with get_engine().begin() as c:
        c.execute(text("DELETE FROM secom_runs WHERE run_id >= :a"), {"a": FIRST_RUN})
