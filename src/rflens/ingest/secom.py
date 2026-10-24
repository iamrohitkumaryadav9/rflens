"""Load UCI SECOM (id=179) into Postgres.

Works from `data.original`: the ucimlrepo `features` and `targets` frames are None for this
dataset. Columns are `class`, `timestamp` and `Attribute 1` ... `Attribute 590`.
"""

import json
import re
import time
from collections import defaultdict
from datetime import date
from pathlib import Path

import pandas as pd
from sqlalchemy import Connection, text

from rflens.db import get_engine

UCI_ID = 179
SOURCE_NAME = "uci_secom"
LICENCE = "CC BY 4.0"
NOTES = (
    "Licence from the UCI dataset page, not present in API metadata. "
    "Cite: McCann, M. & Johnston, A. (2008). SECOM [Dataset]. "
    "UCI Machine Learning Repository. https://doi.org/10.24432/C54305. "
    "API metadata says has_missing_values=no; this is wrong (41,951 NaN cells)."
)

RAW_DIR = Path(__file__).resolve().parents[3] / "data" / "raw" / "secom"
CSV_PATH = RAW_DIR / "secom.csv"
METADATA_PATH = RAW_DIR / "metadata.json"

# Source encoding: class -1 = pass, class 1 = fail.
LABELS = {-1: "pass", 1: "fail"}
ATTRIBUTE_RE = re.compile(r"^Attribute (\d+)$")

TS_FULL = "%d/%m/%Y %H:%M:%S"  # 19/07/2008 11:55:00
TS_SHORT = "%d/%m/%Y %H:%M"  # 1/8/2008 2:02 (no zero padding, no seconds)


def parse_timestamps(raw: pd.Series) -> pd.Series:
    """Parse SECOM timestamps, day-first, in either of the two formats the source mixes.

    Raises ValueError if any value matches neither format.
    """
    parsed = pd.to_datetime(raw, format=TS_FULL, errors="coerce")
    missing = parsed.isna()
    if missing.any():
        parsed[missing] = pd.to_datetime(raw[missing], format=TS_SHORT, errors="coerce")
    bad = parsed.isna()
    if bad.any():
        examples = raw[bad].head(5).tolist()
        raise ValueError(f"{int(bad.sum())} timestamps do not parse, e.g. {examples}")
    return parsed


def write_cache(original: pd.DataFrame, path: Path = CSV_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    original.to_csv(path, index=False)


def read_cache(path: Path = CSV_PATH) -> pd.DataFrame:
    """Read the cached CSV with explicit dtypes; NaN round-trips as NaN (empty field)."""
    dtypes = defaultdict(lambda: "float64", {"class": "int64", "timestamp": "str"})
    return pd.read_csv(path, dtype=dtypes)


def fetch_original() -> tuple[pd.DataFrame, dict]:
    """Return (data.original, metadata), downloading from UCI only if not cached."""
    if CSV_PATH.exists() and METADATA_PATH.exists():
        return read_cache(), json.loads(METADATA_PATH.read_text("utf-8"))
    from ucimlrepo import fetch_ucirepo

    dataset = fetch_ucirepo(id=UCI_ID)
    original = dataset.data.original
    metadata = {
        "repository_url": dataset.metadata["repository_url"],
        "dataset_doi": dataset.metadata["dataset_doi"],
    }
    write_cache(original)
    METADATA_PATH.write_text(json.dumps(metadata), encoding="utf-8")
    return original, metadata


def transform(original: pd.DataFrame) -> tuple[pd.DataFrame, list[int], pd.DataFrame]:
    """Validate the raw frame. Returns (runs, feature_idx per sensor column, sensor values)."""
    if len(original) == 0:
        raise ValueError("SECOM frame is empty")
    unknown = set(original["class"].unique()) - set(LABELS)
    if unknown:
        raise ValueError(f"unexpected label values {sorted(unknown)}; expected -1 or 1")
    sensor_cols = [c for c in original.columns if c not in ("class", "timestamp")]
    matches = [ATTRIBUTE_RE.match(c) for c in sensor_cols]
    if not sensor_cols or not all(matches):
        raise ValueError("unexpected non-'Attribute N' sensor columns")
    feature_idx = [int(m.group(1)) for m in matches]

    runs = pd.DataFrame(
        {
            "run_id": range(1, len(original) + 1),
            "label": original["class"].map(LABELS).to_numpy(),
            "run_ts": parse_timestamps(original["timestamp"]).to_numpy(),
        }
    )
    return runs, feature_idx, original[sensor_cols]


def load(conn: Connection, original: pd.DataFrame, metadata: dict) -> int:
    """Replace the SECOM data using `conn`'s transaction (caller commits). Returns run count."""
    runs, feature_idx, sensors = transform(original)

    conn.execute(text("DELETE FROM secom_measurements"))
    conn.execute(text("DELETE FROM secom_runs"))

    cur_conn = conn.connection.driver_connection  # psycopg connection, same transaction
    with cur_conn.cursor() as cur:
        with cur.copy("COPY secom_runs (run_id, label, run_ts) FROM STDIN") as copy:
            for run_id, label, ts in runs.itertuples(index=False):
                copy.write_row((run_id, label, ts.to_pydatetime()))
        with cur.copy(
            "COPY secom_measurements (run_id, feature_idx, value) FROM STDIN"
        ) as copy:
            values = sensors.to_numpy(dtype="float64")
            for run_id, row in zip(runs["run_id"], values, strict=True):
                for idx, value in zip(feature_idx, row, strict=True):
                    copy.write_row((run_id, idx, None if value != value else float(value)))

    conn.execute(
        text(
            """
            INSERT INTO data_sources (name, licence, url, access_date, row_count, notes)
            VALUES (:name, :licence, :url, :access_date, :row_count, :notes)
            ON CONFLICT (name) DO UPDATE SET
                licence = EXCLUDED.licence,
                url = EXCLUDED.url,
                access_date = EXCLUDED.access_date,
                row_count = EXCLUDED.row_count,
                notes = EXCLUDED.notes
            """
        ),
        {
            "name": SOURCE_NAME,
            "licence": LICENCE,
            "url": metadata["repository_url"],
            "access_date": date.today(),
            "row_count": len(runs),
            "notes": NOTES,
        },
    )
    return len(runs)


def main() -> None:
    start = time.perf_counter()
    original, metadata = fetch_original()
    with get_engine().begin() as conn:
        load(conn, original, metadata)
        runs = conn.execute(text("SELECT count(*) FROM secom_runs")).scalar_one()
        fails = conn.execute(
            text("SELECT count(*) FROM secom_runs WHERE label = 'fail'")
        ).scalar_one()
        measurements = conn.execute(text("SELECT count(*) FROM secom_measurements")).scalar_one()
    elapsed = time.perf_counter() - start
    print(f"runs={runs} fails={fails} measurements={measurements} elapsed={elapsed:.1f}s")


if __name__ == "__main__":
    main()
