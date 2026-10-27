"""The data extract that goes into the sandbox: one wide CSV of runs x features.

Read through a connection of the read-only role rflens_analytics_reader (SELECT on the SECOM tables
only). The container never gets a database address or credential, only this file, mounted
read-only. An empty cell is a missing value (NULL in secom_measurements).
"""

import csv
import hashlib
import io
from dataclasses import dataclass

from sqlalchemy import Connection, text

from rflens.analytics.menu import DATASETS


class ExtractError(Exception):
    pass


@dataclass(frozen=True)
class Extract:
    csv_bytes: bytes
    sha256: str
    n_runs: int
    features: list[int]  # the feature columns, ascending


def build_extract(
    conn: Connection, features: list[int] | None, max_runs: int | None = None
) -> Extract:
    """`features` None: every feature (1..n_features). Rows ordered by run_id."""
    ds = DATASETS["secom"]
    limit = max_runs or ds.max_runs
    cols = list(range(1, ds.n_features + 1)) if features is None else sorted(set(features))
    runs = conn.execute(text("SELECT run_id, label, run_ts FROM secom_runs ORDER BY run_id")).all()
    if len(runs) > limit:
        raise ExtractError(f"the dataset has {len(runs)} runs; the limit is {limit}")
    if not runs:
        raise ExtractError("secom_runs is empty: load it with `make load-secom`")
    values: dict[tuple[int, int], float] = {}
    for run_id, idx, value in conn.execute(
        text(
            "SELECT run_id, feature_idx, value FROM secom_measurements "
            "WHERE feature_idx = ANY(CAST(:idx AS int[])) AND value IS NOT NULL"
        ),
        {"idx": cols},
    ):
        values[(run_id, idx)] = float(value)
    out = io.StringIO(newline="")
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(["run_id", "label", "run_ts", *[f"f{i}" for i in cols]])
    for run_id, label, run_ts in runs:
        writer.writerow(
            [
                run_id,
                label,
                run_ts.isoformat(sep=" "),
                *[repr(values[(run_id, i)]) if (run_id, i) in values else "" for i in cols],
            ]
        )
    data = out.getvalue().encode("utf-8")
    return Extract(data, hashlib.sha256(data).hexdigest(), len(runs), cols)
