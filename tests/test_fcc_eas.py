import hashlib
import json
from collections.abc import Iterator
from datetime import date
from pathlib import Path

import pytest
from sqlalchemy import Connection, text
from sqlalchemy.exc import IntegrityError

from rflens.db import get_engine
from rflens.ingest import fcc_eas
from rflens.ingest.fcc_eas import GRANTEES, TEST_FIRMS, ensure_file, load, parse_csv

# Real rows from the FCC CSVs with personal contact and address columns blanked.
FIXTURE_DIR = Path(__file__).parent / "fixtures" / "fcc_eas"
GRANTEES_CSV = FIXTURE_DIR / "grantees.csv"
TEST_FIRMS_CSV = FIXTURE_DIR / "accredited_test_firms.csv"
PATHS = {"fcc_grantees": GRANTEES_CSV, "fcc_test_firms": TEST_FIRMS_CSV}


@pytest.fixture
def conn() -> Iterator[Connection]:
    """Real Postgres connection; rolled back so a full load is never touched."""
    with get_engine().connect() as connection:
        transaction = connection.begin()
        try:
            yield connection
        finally:
            transaction.rollback()


def counts(conn: Connection) -> dict[str, int]:
    return {
        t: conn.execute(text(f"SELECT count(*) FROM {t}")).scalar_one()
        for t in ("fcc_grantees", "fcc_test_firms")
    }


def write_grantees(tmp_path: Path, *replacements: tuple[str, str]) -> Path:
    content = GRANTEES_CSV.read_text(encoding="utf-8")
    for old, new in replacements:
        assert old in content
        content = content.replace(old, new, 1)
    path = tmp_path / "grantees.csv"
    path.write_text(content, encoding="utf-8")
    return path


# --- parsing: fail loudly ----------------------------------------------------------------------


def test_parse_fixture_rows() -> None:
    assert len(parse_csv(GRANTEES, GRANTEES_CSV)) == 8
    assert len(parse_csv(TEST_FIRMS, TEST_FIRMS_CSV)) == 7


def test_unparseable_date_raises(tmp_path: Path) -> None:
    path = write_grantees(tmp_path, ("09/18/2013 12:00:00 AM", "2013-09-18"))
    with pytest.raises(ValueError, match="grantees.csv line 3"):
        parse_csv(GRANTEES, path)


def test_date_with_time_of_day_raises(tmp_path: Path) -> None:
    path = write_grantees(tmp_path, ("09/18/2013 12:00:00 AM", "09/18/2013 03:30:00 PM"))
    with pytest.raises(ValueError, match="time of day"):
        parse_csv(GRANTEES, path)


@pytest.mark.parametrize("bad", ["ZY", "ZY66", "zy6", "ZY6 ", "2AA2", "2AA221", "ZY-6"])
def test_bad_grantee_code_raises(tmp_path: Path, bad: str) -> None:
    path = write_grantees(tmp_path, ("ZY6,", f"{bad},"))
    with pytest.raises(ValueError, match="grantee_code"):
        parse_csv(GRANTEES, path)


def test_repeated_grantee_code_raises(tmp_path: Path) -> None:
    path = write_grantees(tmp_path, ("2AA22,", "ZY6,"))
    with pytest.raises(ValueError, match="repeated key"):
        parse_csv(GRANTEES, path)


def test_header_mismatch_raises(tmp_path: Path) -> None:
    path = write_grantees(tmp_path, ("grantee_name", "name"))
    with pytest.raises(ValueError, match="unexpected header"):
        parse_csv(GRANTEES, path)


def test_wrong_field_count_raises(tmp_path: Path) -> None:
    path = write_grantees(tmp_path, ("2AA22,", "2AA22,extra,"))
    with pytest.raises(ValueError, match="fields, expected 10"):
        parse_csv(GRANTEES, path)


def test_invalid_utf8_raises(tmp_path: Path) -> None:
    path = tmp_path / "grantees.csv"
    path.write_bytes(GRANTEES_CSV.read_bytes().replace(b"Hsinchu", b"Hsinch\xe9"))
    with pytest.raises(UnicodeDecodeError):
        parse_csv(GRANTEES, path)


def test_invalid_contract_raises(tmp_path: Path) -> None:
    content = TEST_FIRMS_CSV.read_text(encoding="utf-8").replace(",N\n", ",Maybe\n", 1)
    path = tmp_path / "accredited_test_firms.csv"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="contract"):
        parse_csv(TEST_FIRMS, path)


# --- load into real Postgres -------------------------------------------------------------------


def test_row_counts_and_code_lengths(conn: Connection) -> None:
    assert load(conn, PATHS) == {"fcc_grantees": 8, "fcc_test_firms": 7}
    assert counts(conn) == {"fcc_grantees": 8, "fcc_test_firms": 7}
    lengths = dict(
        conn.execute(
            text("SELECT length(grantee_code), count(*) FROM fcc_grantees GROUP BY 1")
        ).all()
    )
    assert lengths == {3: 3, 5: 5}  # ZY6, OAF, U7F and five 5-character codes


def test_placeholders_become_null_and_nil_is_kept(conn: Connection) -> None:
    load(conn, PATHS)
    rows = {
        r.grantee_code: r
        for r in conn.execute(
            text("SELECT grantee_code, grantee_name, city, state, country FROM fcc_grantees")
        )
    }
    assert rows["2AA22"].state is None  # N/A
    assert rows["2AA22"].city == "Hsinchu"
    assert rows["2AA23"].city == "NIL"  # NIL is not a placeholder
    assert rows["2AJCJ"].city == "nil"  # only the explicit list is converted
    assert rows["U7F"].city is None  # NA
    assert rows["2AI25"].country is None  # N/A
    assert rows["ZY6"].state == "Missouri"


def test_blank_name_row_is_null(conn: Connection) -> None:
    load(conn, PATHS)
    row = conn.execute(
        text(
            "SELECT grantee_name, city, state, country FROM fcc_grantees WHERE grantee_code = 'OAF'"
        )
    ).one()
    assert row == (None, None, None, None)


def test_date_received_parsed(conn: Connection) -> None:
    load(conn, PATHS)
    value = conn.execute(
        text("SELECT date_received FROM fcc_grantees WHERE grantee_code = '2AA22'")
    ).scalar_one()
    assert value == date(2013, 9, 18)


def test_designation_number_collisions_both_loaded(conn: Connection) -> None:
    load(conn, PATHS)
    rows = conn.execute(
        text(
            "SELECT designation_number, firm_name, country FROM fcc_test_firms "
            "WHERE designation_number IN ('US5037', 'US1273') ORDER BY 1, 2"
        )
    ).all()
    assert [(r.designation_number, r.country) for r in rows] == [
        ("US1273", "China"),
        ("US1273", "India"),
        ("US5037", "China"),
        ("US5037", "United States"),
    ]


def test_test_firm_fields(conn: Connection) -> None:
    load(conn, PATHS)
    mra = dict(conn.execute(text("SELECT designation_number, mra FROM fcc_test_firms")).all())
    assert mra["US1168"] is None  # N/A
    assert mra["CA2068"] == "APEC TEL"
    row = conn.execute(
        text(
            "SELECT contract, expiration_date FROM fcc_test_firms "
            "WHERE designation_number = 'CA2068'"
        )
    ).one()
    assert row.contract == "N"
    assert isinstance(row.expiration_date, date)


def test_load_twice_gives_same_counts(conn: Connection) -> None:
    load(conn, PATHS)
    first = counts(conn)
    load(conn, PATHS)
    assert counts(conn) == first


def test_data_sources_rows(conn: Connection) -> None:
    load(conn, PATHS)
    rows = {
        r.name: r
        for r in conn.execute(
            text(
                "SELECT name, licence, url, row_count, notes FROM data_sources "
                "WHERE name LIKE 'fcc_eas_%'"
            )
        )
    }
    assert set(rows) == {"fcc_eas_grantees", "fcc_eas_test_firms"}
    g, t = rows["fcc_eas_grantees"], rows["fcc_eas_test_firms"]
    assert (g.row_count, t.row_count) == (8, 7)
    assert g.licence == "US Government Work (https://www.usa.gov/government-works)"
    assert g.url == "https://opendata.fcc.gov/d/3b3k-34jp"
    for row, path, dataset_id in ((g, GRANTEES_CSV, "3b3k-34jp"), (t, TEST_FIRMS_CSV, "nubx-v54a")):
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        assert f"SHA-256 {sha}" in row.notes
        assert f"https://opendata.fcc.gov/api/views/{dataset_id}/rows.csv" in row.notes
        assert "last updated 2021-03-22" in row.notes


def test_database_check_rejects_bad_grantee_code(conn: Connection) -> None:
    with pytest.raises(IntegrityError, match="ck_fcc_grantees_code"), conn.begin_nested():
        conn.execute(text("INSERT INTO fcc_grantees (grantee_code) VALUES ('ab')"))


def test_failed_parse_leaves_database_untouched(conn: Connection, tmp_path: Path) -> None:
    load(conn, PATHS)
    before = counts(conn)
    bad = write_grantees(tmp_path, ("09/18/2013 12:00:00 AM", "not a date"))
    with pytest.raises(ValueError):
        load(conn, {**PATHS, "fcc_grantees": bad})
    assert counts(conn) == before


# --- cached download ---------------------------------------------------------------------------


def no_network(*args: object, **kwargs: object) -> None:
    raise AssertionError("must not download when the cached SHA-256 matches")


def test_ensure_file_uses_cache_when_sha_matches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cached = tmp_path / GRANTEES.filename
    cached.write_bytes(GRANTEES_CSV.read_bytes())
    sha = hashlib.sha256(cached.read_bytes()).hexdigest()
    (tmp_path / "manifest.json").write_text(json.dumps({GRANTEES.filename: sha}), encoding="utf-8")
    monkeypatch.setattr(fcc_eas, "RAW_DIR", tmp_path)
    monkeypatch.setattr(fcc_eas.urllib.request, "urlopen", no_network)
    assert ensure_file(GRANTEES) == cached


def test_ensure_file_writes_manifest_for_existing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / GRANTEES.filename).write_bytes(GRANTEES_CSV.read_bytes())
    monkeypatch.setattr(fcc_eas, "RAW_DIR", tmp_path)
    monkeypatch.setattr(fcc_eas.urllib.request, "urlopen", no_network)
    ensure_file(GRANTEES)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest == {GRANTEES.filename: hashlib.sha256(GRANTEES_CSV.read_bytes()).hexdigest()}
