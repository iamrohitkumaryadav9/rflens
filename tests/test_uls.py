import hashlib
import json
import zipfile
from collections.abc import Iterator
from decimal import Decimal
from pathlib import Path

import pytest
from sqlalchemy import Connection, text

from rflens.db import get_engine
from rflens.ingest import uls
from rflens.ingest.uls import dms_to_decimal, ensure_zip, load, parse_file, read_counts

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "uls"
FIXTURE_COUNTS = {"HD": 15, "EN": 28, "FR": 966, "LO": 178}


@pytest.fixture(scope="session")
def fixture_zip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Zip built from the committed fixture files, in the layout of an FCC l_<service>.zip."""
    path = tmp_path_factory.mktemp("uls") / "l_fixture.zip"
    with zipfile.ZipFile(path, "w") as z:
        for member in ("counts", "HD.dat", "EN.dat", "FR.dat", "LO.dat"):
            z.write(FIXTURE_DIR / member, member)
    return path


@pytest.fixture
def conn() -> Iterator[Connection]:
    """Real Postgres connection; rolled back so the full cell load is never touched."""
    with get_engine().connect() as connection:
        transaction = connection.begin()
        try:
            yield connection
        finally:
            transaction.rollback()


def count(conn: Connection, table: str, service: str) -> int:
    return conn.execute(
        text(f"SELECT count(*) FROM {table} WHERE service = :s"), {"s": service}
    ).scalar_one()


def table_counts(conn: Connection, service: str) -> dict[str, int]:
    tables = ("uls_hd", "uls_en", "uls_fr", "uls_lo", "uls_file_counts")
    return {t: count(conn, t, service) for t in tables}


# --- DMS conversion: one real row of each hemisphere combination from l_cell.zip -------------


@pytest.mark.parametrize(
    ("dms", "axis", "expected"),
    [
        # usi 13450 (WPOI220) location 1: N / W
        (("29", "46", "54.0", "N"), "lat", 29.7816667),
        (("92", "10", "57.0", "W"), "lon", -92.1825),
        # usi 13384 (KNKR259) location 1: S
        (("14", "16", "14.0", "S"), "lat", -14.2705556),
        (("170", "41", "40.0", "W"), "lon", -170.6944444),
        # usi 11561 (KNKQ317) location 1: E
        (("13", "29", "21.0", "N"), "lat", 13.4891667),
        (("144", "49", "49.0", "E"), "lon", 144.8302778),
        # usi 11913 (KNKA758) location 11: seconds of exactly 60.0 in the source
        (("77", "55", "60.0", "W"), "lon", -77.9333333),
    ],
)
def test_dms_to_decimal_real_rows(dms: tuple[str, ...], axis: str, expected: float) -> None:
    assert dms_to_decimal(*dms, axis=axis) == pytest.approx(expected, abs=1e-6)


def test_dms_missing_component_is_none() -> None:
    assert dms_to_decimal("30", "4", "", "N", axis="lat") is None
    assert dms_to_decimal("", "", "", "", axis="lon") is None


@pytest.mark.parametrize(
    ("dms", "axis"),
    [
        (("30", "4", "5.0", "E"), "lat"),  # E is not a latitude direction
        (("92", "10", "57.0", "N"), "lon"),
        (("30", "60", "5.0", "N"), "lat"),  # minutes out of range
        (("91", "0", "0.0", "N"), "lat"),
    ],
)
def test_dms_invalid_raises(dms: tuple[str, ...], axis: str) -> None:
    with pytest.raises(ValueError):
        dms_to_decimal(*dms, axis=axis)


# --- parsing: fail loudly ----------------------------------------------------------------------


def hd_lines() -> list[bytes]:
    return (FIXTURE_DIR / "HD.dat").read_bytes().split(b"\n")[:-1]


def test_read_counts() -> None:
    counts, snapshot = read_counts((FIXTURE_DIR / "counts").read_text(encoding="utf-8"))
    assert counts == FIXTURE_COUNTS
    assert snapshot.isoformat() == "2026-10-18"


def test_parse_file_ok_and_row_shape() -> None:
    rows = parse_file("HD", (FIXTURE_DIR / "HD.dat").read_bytes(), FIXTURE_COUNTS["HD"])
    assert len(rows) == FIXTURE_COUNTS["HD"]
    assert rows[0][0] == 1  # line_no first


def test_line_count_mismatch_raises() -> None:
    raw = (FIXTURE_DIR / "HD.dat").read_bytes()
    with pytest.raises(ValueError, match="counts file says 16"):
        parse_file("HD", raw, FIXTURE_COUNTS["HD"] + 1)


def test_field_count_mismatch_raises() -> None:
    lines = hd_lines()
    lines[2] = lines[2].replace(b"|", b"", 1)
    with pytest.raises(ValueError, match="line 3: 58 fields, expected 59"):
        parse_file("HD", b"\n".join(lines) + b"\n", FIXTURE_COUNTS["HD"])


def test_non_utf8_line_raises() -> None:
    lines = hd_lines()
    lines[1] = lines[1].replace(b"|KN", b"|\xe9N", 1)  # lone latin-1 byte
    assert b"\xe9" in lines[1]
    with pytest.raises(ValueError, match="line 2: not valid UTF-8"):
        parse_file("HD", b"\n".join(lines) + b"\n", FIXTURE_COUNTS["HD"])


def test_bad_date_raises() -> None:
    lines = hd_lines()
    lines[0] = lines[0].replace(b"02/07/1997", b"1997-02-07", 1)
    with pytest.raises(ValueError, match="line 1"):
        parse_file("HD", b"\n".join(lines) + b"\n", FIXTURE_COUNTS["HD"])


def test_wrong_record_type_raises() -> None:
    lines = (FIXTURE_DIR / "EN.dat").read_bytes().split(b"\n")[:-1]
    lines[0] = lines[0].replace(b"EN|", b"XX|", 1)  # right field count, wrong record type
    with pytest.raises(ValueError, match="line 1: record type 'XX'"):
        parse_file("EN", b"\n".join(lines) + b"\n", FIXTURE_COUNTS["EN"])


# --- load into real Postgres -------------------------------------------------------------------


def test_row_counts_match_fixture(conn: Connection, fixture_zip: Path) -> None:
    loaded = load(conn, fixture_zip, "fx1")
    assert loaded == FIXTURE_COUNTS
    counts = table_counts(conn, "fx1")
    assert counts == {
        "uls_hd": FIXTURE_COUNTS["HD"],
        "uls_en": FIXTURE_COUNTS["EN"],
        "uls_fr": FIXTURE_COUNTS["FR"],
        "uls_lo": FIXTURE_COUNTS["LO"],
        "uls_file_counts": 4,
    }
    file_counts = dict(
        conn.execute(
            text("SELECT record_type, line_count FROM uls_file_counts WHERE service = 'fx1'")
        ).all()
    )
    assert file_counts == FIXTURE_COUNTS
    provenance = conn.execute(
        text(
            "SELECT DISTINCT sha256, snapshot_date FROM uls_file_counts WHERE service = 'fx1'"
        )
    ).all()
    assert [(r.sha256, r.snapshot_date.isoformat()) for r in provenance] == [
        (hashlib.sha256(fixture_zip.read_bytes()).hexdigest(), "2026-10-18")
    ]


def test_load_twice_gives_same_counts(conn: Connection, fixture_zip: Path) -> None:
    load(conn, fixture_zip, "fx1")
    first = table_counts(conn, "fx1")
    load(conn, fixture_zip, "fx1")
    assert table_counts(conn, "fx1") == first


def test_reload_replaces_only_its_own_service(conn: Connection, fixture_zip: Path) -> None:
    load(conn, fixture_zip, "fx1")
    load(conn, fixture_zip, "fx2")
    before = table_counts(conn, "fx2")
    load(conn, fixture_zip, "fx1")
    assert table_counts(conn, "fx2") == before
    assert table_counts(conn, "fx1") == before


def test_non_ascii_entity_name(conn: Connection, fixture_zip: Path) -> None:
    load(conn, fixture_zip, "fx1")
    name = conn.execute(
        text(
            "SELECT entity_name FROM uls_en "
            "WHERE service = 'fx1' AND usi = 193253 AND entity_type = 'L'"
        )
    ).scalar_one()
    assert name.startswith("360° Communications")


def test_duplicate_fr_key_group_both_loaded(conn: Connection, fixture_zip: Path) -> None:
    load(conn, fixture_zip, "fx1")
    erp = conn.execute(
        text(
            "SELECT power_erp FROM uls_fr WHERE service = 'fx1' AND usi = 13401 "
            "AND location_number = 5 AND antenna_number = 1 AND freq_seq_id = 1 "
            "ORDER BY line_no"
        )
    ).scalars().all()
    assert erp == [Decimal("140.820"), Decimal("276.000")]


def test_empty_frequency_assigned_is_null(conn: Connection, fixture_zip: Path) -> None:
    load(conn, fixture_zip, "fx1")
    nulls = conn.execute(
        text(
            "SELECT count(*) FROM uls_fr WHERE service = 'fx1' AND usi = 12120 "
            "AND frequency_assigned IS NULL"
        )
    ).scalar_one()
    assert nulls == 6


def test_non_positive_power_erp_stored_as_is(conn: Connection, fixture_zip: Path) -> None:
    load(conn, fixture_zip, "fx1")
    minimum = conn.execute(
        text("SELECT min(power_erp) FROM uls_fr WHERE service = 'fx1' AND usi = 11689")
    ).scalar_one()
    assert minimum == Decimal("-155.000")


@pytest.mark.parametrize(
    ("usi", "location", "lat", "lon"),
    [
        (13450, 1, 29.7816667, -92.1825),  # N / W
        (13384, 1, -14.2705556, -170.6944444),  # S
        (11561, 1, 13.4891667, 144.8302778),  # E
        (12510, 55, 30.1690833, -3.9443889),  # source longitude of 3 degrees, stored as-is
        (11913, 11, 34.3237778, -77.9333333),  # seconds of exactly 60.0
    ],
)
def test_lat_lon_stored_as_signed_decimal(
    conn: Connection, fixture_zip: Path, usi: int, location: int, lat: float, lon: float
) -> None:
    load(conn, fixture_zip, "fx1")
    row = conn.execute(
        text(
            "SELECT latitude, longitude FROM uls_lo "
            "WHERE service = 'fx1' AND usi = :u AND location_number = :l"
        ),
        {"u": usi, "l": location},
    ).one()
    assert row.latitude == pytest.approx(lat, abs=1e-6)
    assert row.longitude == pytest.approx(lon, abs=1e-6)


def test_incomplete_latitude_is_null(conn: Connection, fixture_zip: Path) -> None:
    load(conn, fixture_zip, "fx1")
    row = conn.execute(
        text(
            "SELECT latitude, longitude FROM uls_lo "
            "WHERE service = 'fx1' AND usi = 13056 AND location_number = 56"
        )
    ).one()
    assert row.latitude is None
    assert row.longitude == pytest.approx(-(87 + 57 / 60 + 19 / 3600), abs=1e-6)


def test_out_of_range_coordinate_is_null_with_warning(
    conn: Connection, fixture_zip: Path
) -> None:
    warnings: list[str] = []
    load(conn, fixture_zip, "fx1", warnings)
    row = conn.execute(
        text(
            "SELECT latitude, longitude FROM uls_lo "
            "WHERE service = 'fx1' AND usi = 12361 AND location_number = 33"
        )
    ).one()  # source latitude is 36 deg 73 min 11.3 sec N
    assert row.latitude is None
    assert row.longitude == pytest.approx(-(84 + 5 / 60 + 35.7 / 3600), abs=1e-6)
    assert len(warnings) == 1
    assert "usi 12361, location 33" in warnings[0]


def test_joins_resolve(conn: Connection, fixture_zip: Path) -> None:
    load(conn, fixture_zip, "fx1")
    row = conn.execute(
        text(
            "SELECT h.call_sign, e.entity_name, count(*) AS n FROM uls_hd h "
            "JOIN uls_en e USING (service, usi) JOIN uls_fr f USING (service, usi) "
            "WHERE h.service = 'fx1' AND h.usi = 13401 AND e.entity_type = 'L' "
            "GROUP BY 1, 2"
        )
    ).one()
    assert (row.call_sign, row.n) == ("KNKN520", 36)
    assert row.entity_name


def test_data_sources_row(conn: Connection, fixture_zip: Path) -> None:
    load(conn, fixture_zip, "fx1")
    row = conn.execute(
        text("SELECT licence, url, row_count, notes FROM data_sources WHERE name = 'uls_fx1'")
    ).one()
    sha = hashlib.sha256(fixture_zip.read_bytes()).hexdigest()
    assert row.licence == "US Government Work (https://www.usa.gov/government-works)"
    assert row.url == "https://data.fcc.gov/download/pub/uls/complete/l_fx1.zip"
    assert row.row_count == FIXTURE_COUNTS["HD"]  # licences, not total rows
    assert f"SHA-256 {sha}" in row.notes
    assert "l_fx1.zip dated 2026-10-18" in row.notes
    assert "frequency assumed MHz, power unit unknown" in row.notes


def test_failed_parse_leaves_database_untouched(
    conn: Connection, fixture_zip: Path, tmp_path: Path
) -> None:
    load(conn, fixture_zip, "fx1")
    before = table_counts(conn, "fx1")
    bad = tmp_path / "l_bad.zip"
    with zipfile.ZipFile(bad, "w") as z:
        for member in ("counts", "EN.dat", "FR.dat", "LO.dat"):
            z.write(FIXTURE_DIR / member, member)
        z.writestr("HD.dat", b"\n".join(hd_lines()[:-1]) + b"\n")  # one line short of counts
    with pytest.raises(ValueError, match=f"counts file says {FIXTURE_COUNTS['HD']}"):
        load(conn, bad, "fx1")
    assert table_counts(conn, "fx1") == before


# --- cached download ---------------------------------------------------------------------------


def no_network(*args: object, **kwargs: object) -> None:
    raise AssertionError("must not download when the cached SHA-256 matches")


def test_ensure_zip_uses_cache_when_sha_matches(
    tmp_path: Path, fixture_zip: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = tmp_path / "svc"
    folder.mkdir()
    cached = folder / "l_svc.zip"
    cached.write_bytes(fixture_zip.read_bytes())
    sha = hashlib.sha256(cached.read_bytes()).hexdigest()
    (folder / "manifest.json").write_text(json.dumps({"sha256": sha}), encoding="utf-8")
    monkeypatch.setattr(uls, "RAW_ROOT", tmp_path)
    monkeypatch.setattr(uls.urllib.request, "urlopen", no_network)
    assert ensure_zip("svc") == cached


def test_ensure_zip_writes_manifest_for_existing_zip(
    tmp_path: Path, fixture_zip: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = tmp_path / "svc"
    folder.mkdir()
    (folder / "l_svc.zip").write_bytes(fixture_zip.read_bytes())
    monkeypatch.setattr(uls, "RAW_ROOT", tmp_path)
    monkeypatch.setattr(uls.urllib.request, "urlopen", no_network)
    ensure_zip("svc")
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["sha256"] == hashlib.sha256(fixture_zip.read_bytes()).hexdigest()
