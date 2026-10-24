from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import Connection, text

from rflens.db import get_engine
from rflens.ingest.secom import SOURCE_NAME, load, parse_timestamps, read_cache, write_cache

METADATA = {"repository_url": "https://archive.ics.uci.edu/dataset/179/secom"}


def make_original() -> pd.DataFrame:
    """Small frame in the layout of ucimlrepo's data.original for SECOM."""
    return pd.DataFrame(
        {
            "class": [-1, -1, 1, -1],
            "timestamp": [
                "19/07/2008 11:55:00",
                "31/07/2008 23:50:00",
                "1/8/2008 2:02",
                "1/8/2008 10:20",
            ],
            "Attribute 1": [3030.93, np.nan, 2932.61, 1.5],
            "Attribute 2": [2564.0, 2465.14, np.nan, np.nan],
            "Attribute 3": [1.0, 2.0, 3.0, 4.0],
        }
    )


def test_csv_cache_round_trip(tmp_path: Path) -> None:
    original = make_original()
    path = tmp_path / "secom.csv"
    write_cache(original, path)
    pd.testing.assert_frame_equal(read_cache(path), original)


def test_parse_timestamps_both_formats() -> None:
    raw = pd.Series(["19/07/2008 11:55:00", "1/8/2008 2:02"])
    parsed = parse_timestamps(raw)
    assert parsed.iloc[0] == pd.Timestamp("2008-07-19 11:55:00")
    assert parsed.iloc[1] == pd.Timestamp("2008-08-01 02:02:00")


def test_parse_timestamps_day_first_ordering() -> None:
    parsed = parse_timestamps(pd.Series(["31/07/2008 23:50:00", "1/8/2008 0:05"]))
    assert parsed.iloc[0] < parsed.iloc[1]
    assert parsed.iloc[1] == pd.Timestamp("2008-08-01 00:05:00")


def test_parse_timestamps_raises_on_bad_value() -> None:
    with pytest.raises(ValueError, match="do not parse"):
        parse_timestamps(pd.Series(["19/07/2008 11:55:00", "not a date"]))


@pytest.fixture
def conn() -> Iterator[Connection]:
    """Real Postgres connection; everything is rolled back so the full load is never touched."""
    with get_engine().connect() as connection:
        transaction = connection.begin()
        try:
            yield connection
        finally:
            transaction.rollback()


def counts(conn: Connection) -> tuple[int, int, int]:
    return (
        conn.execute(text("SELECT count(*) FROM secom_runs")).scalar_one(),
        conn.execute(text("SELECT count(*) FROM secom_measurements")).scalar_one(),
        conn.execute(text("SELECT count(*) FROM data_sources WHERE name = :n"), {"n": SOURCE_NAME})
        .scalar_one(),
    )


def test_row_counts_match_input(conn: Connection) -> None:
    original = make_original()
    assert load(conn, original, METADATA) == 4
    runs, measurements, _ = counts(conn)
    assert runs == 4
    assert measurements == 4 * 3
    labels = dict(conn.execute(text("SELECT label, count(*) FROM secom_runs GROUP BY label")).all())
    assert labels == {"pass": 3, "fail": 1}


def test_nan_stored_as_null(conn: Connection) -> None:
    load(conn, make_original(), METADATA)
    nulls = conn.execute(text("SELECT count(*) FROM secom_measurements WHERE value IS NULL"))
    assert nulls.scalar_one() == 3
    value = conn.execute(
        text("SELECT value FROM secom_measurements WHERE run_id = 2 AND feature_idx = 1")
    ).scalar_one()
    assert value is None
    value = conn.execute(
        text("SELECT value FROM secom_measurements WHERE run_id = 1 AND feature_idx = 1")
    ).scalar_one()
    assert value == pytest.approx(3030.93)


def test_load_twice_gives_same_counts(conn: Connection) -> None:
    original = make_original()
    load(conn, original, METADATA)
    first = counts(conn)
    load(conn, original, METADATA)
    assert counts(conn) == first


def test_data_sources_row(conn: Connection) -> None:
    load(conn, make_original(), METADATA)
    row = conn.execute(
        text("SELECT licence, url, row_count FROM data_sources WHERE name = :n"),
        {"n": SOURCE_NAME},
    ).one()
    assert row == ("CC BY 4.0", METADATA["repository_url"], 4)


def test_unexpected_label_raises(conn: Connection) -> None:
    original = make_original()
    original.loc[0, "class"] = 0
    with pytest.raises(ValueError, match="unexpected label"):
        load(conn, original, METADATA)


def test_bad_timestamp_raises(conn: Connection) -> None:
    original = make_original()
    original.loc[0, "timestamp"] = "garbage"
    with pytest.raises(ValueError, match="do not parse"):
        load(conn, original, METADATA)
