"""Load an FCC ULS weekly complete licence file (complete/l_<service>.zip) into Postgres.

Record layouts (field positions) come from the FCC "ULS Data File Format" document, public access
database definitions v9.0. Files are pipe-delimited with CRLF line endings, no quoting or escaping,
and empty fields for missing values. Only HD, EN, FR and LO are loaded.

The load fails loudly (ValueError) on: a line with the wrong field count, a line that is not valid
UTF-8, a line count that disagrees with the zip's own `counts` file, an unparseable value.
"""

import argparse
import hashlib
import json
import re
import time
import urllib.request
import zipfile
from collections.abc import Callable
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

from sqlalchemy import Connection, text

from rflens.db import get_engine

BASE_URL = "https://data.fcc.gov/download/pub/uls/complete"
RAW_ROOT = Path(__file__).resolve().parents[3] / "data" / "raw" / "uls"
LICENCE = "US Government Work (https://www.usa.gov/government-works)"
NOTES = (
    "Licence from the data.gov catalogue record "
    "https://catalog.data.gov/dataset/fcc-universal-licensing-system-uls, not from an FCC terms "
    "page (fcc.gov returned 403 to automated requests). Snapshot: complete/l_{service}.zip dated "
    "{snapshot}, SHA-256 {sha256}. Units not stated in the layout doc: frequency assumed MHz, "
    "power unit unknown."
)
SERVICE_RE = re.compile(r"^[A-Za-z0-9_]+$")


def _text(v: str) -> str | None:
    return v or None


def _int(v: str) -> int | None:
    return int(v) if v else None


def _num(v: str) -> str | None:
    """Validated numeric text; passed to Postgres as-is so no precision is lost."""
    if not v:
        return None
    Decimal(v)
    return v


def _date(v: str) -> date | None:
    return datetime.strptime(v, "%m/%d/%Y").date() if v else None


Converter = Callable[[str], object]

# record type -> (table, expected field count, [(column, 1-based position, converter)])
RECORDS: dict[str, tuple[str, int, list[tuple[str, int, Converter]]]] = {
    "HD": (
        "uls_hd",
        59,
        [
            ("usi", 2, _int),
            ("call_sign", 5, _text),
            ("license_status", 6, _text),
            ("radio_service_code", 7, _text),
            ("grant_date", 8, _date),
            ("expired_date", 9, _date),
            ("cancellation_date", 10, _date),
            ("effective_date", 43, _date),
            ("last_action_date", 44, _date),
            ("auction_id", 45, _int),
        ],
    ),
    "EN": (
        "uls_en",
        30,
        [
            ("usi", 2, _int),
            ("entity_type", 6, _text),
            ("licensee_id", 7, _text),
            ("entity_name", 8, _text),
            ("first_name", 9, _text),
            ("last_name", 11, _text),
            ("city", 17, _text),
            ("state", 18, _text),
            ("zip_code", 19, _text),
            ("frn", 23, _text),
        ],
    ),
    "FR": (
        "uls_fr",
        30,
        [
            ("usi", 2, _int),
            ("location_number", 7, _int),
            ("antenna_number", 8, _int),
            ("freq_seq_id", 27, _int),
            ("class_station_code", 9, _text),
            ("frequency_assigned", 11, _num),
            ("frequency_upper_band", 12, _num),
            ("power_output", 16, _num),
            ("power_erp", 17, _num),
            ("eirp", 21, _num),
        ],
    ),
    "LO": (
        "uls_lo",
        51,
        [
            ("usi", 2, _int),
            ("location_number", 9, _int),
            ("location_type_code", 7, _text),
            ("location_class_code", 8, _text),
            ("location_city", 13, _text),
            ("location_county", 14, _text),
            ("location_state", 15, _text),
            ("ground_elevation", 19, _num),
            ("tower_registration_number", 38, _text),
            ("height_of_support_structure", 39, _num),
            ("overall_height_of_structure", 40, _num),
            ("structure_type", 41, _text),
            ("location_name", 43, _text),
        ],
    ),
}
LO_LATITUDE = (20, 21, 22, 23)  # degrees, minutes, seconds, direction
LO_LONGITUDE = (24, 25, 26, 27)
LOAD_ORDER = ("HD", "EN", "FR", "LO")


class CoordinateOutOfRange(ValueError):
    """A degrees/minutes/seconds component is outside its valid range."""


def dms_to_decimal(
    degrees: str, minutes: str, seconds: str, direction: str, axis: str
) -> float | None:
    """Signed decimal degrees from DMS text. `axis` is 'lat' or 'lon'.

    South and West are negative. Returns None if any component is empty. Raises ValueError on a
    wrong direction letter, and CoordinateOutOfRange on an out-of-range component. Seconds of
    exactly 60 are accepted: the source has such rows (a rounding artifact) and the arithmetic is
    still exact.
    """
    directions, limit = ("NS", 90) if axis == "lat" else ("EW", 180)
    if not (degrees and minutes and seconds and direction):
        return None
    if direction not in directions:
        raise ValueError(f"direction {direction!r} invalid for {axis}")
    deg, mins, secs = int(degrees), int(minutes), float(seconds)
    if not (0 <= deg <= limit and 0 <= mins < 60 and 0 <= secs <= 60):
        raise CoordinateOutOfRange(f"{axis} {degrees} {minutes} {seconds} {direction}")
    value = deg + mins / 60 + secs / 3600
    return -value if direction in "SW" else value


def read_counts(counts_text: str) -> tuple[dict[str, int], date]:
    """Parse the zip's `counts` file: (line counts by file stem, snapshot date)."""
    created = re.search(
        r"File Creation Date: \w{3} (\w{3}) (\d{1,2}) [\d:]+ \w+ (\d{4})", counts_text
    )
    if not created:
        raise ValueError("counts file has no 'File Creation Date' line")
    snapshot = datetime.strptime(" ".join(created.groups()), "%b %d %Y").date()
    counts: dict[str, int] = {}
    for line in counts_text.splitlines():
        m = re.match(r"^\s*(\d+)\s+(\S+)$", line)
        if m and m.group(2).endswith(".dat"):
            counts[m.group(2).rsplit("/", 1)[-1][: -len(".dat")]] = int(m.group(1))
    return counts, snapshot


def parse_file(
    record_type: str, raw: bytes, expected_lines: int, warnings: list[str] | None = None
) -> list[tuple]:
    """Validate and convert one .dat file. Rows are (line_no, *values) in table column order.

    An out-of-range LO coordinate is stored as NULL and described in `warnings`.
    """
    _, n_fields, columns = RECORDS[record_type]
    name = f"{record_type}.dat"
    lines = raw.split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()
    if len(lines) != expected_lines:
        raise ValueError(f"{name}: {len(lines)} lines but counts file says {expected_lines}")

    rows = []
    for line_no, line in enumerate(lines, start=1):
        try:
            fields = line.removesuffix(b"\r").decode("utf-8").split("|")
        except UnicodeDecodeError as e:
            raise ValueError(f"{name} line {line_no}: not valid UTF-8") from e
        if len(fields) != n_fields:
            raise ValueError(f"{name} line {line_no}: {len(fields)} fields, expected {n_fields}")
        if fields[0] != record_type:
            raise ValueError(f"{name} line {line_no}: record type {fields[0]!r}")
        try:
            values = [convert(fields[pos - 1]) for _, pos, convert in columns]
            if record_type == "LO":
                coords = []
                for positions, axis in ((LO_LATITUDE, "lat"), (LO_LONGITUDE, "lon")):
                    parts = [fields[p - 1] for p in positions]
                    try:
                        coords.append(dms_to_decimal(*parts, axis=axis))
                    except CoordinateOutOfRange as e:
                        coords.append(None)
                        if warnings is not None:
                            warnings.append(
                                f"{name} line {line_no} (usi {fields[1]}, location {fields[8]}): "
                                f"{e} out of range, stored as NULL"
                            )
                values[8:8] = coords  # after ground_elevation
        except (ValueError, InvalidOperation) as e:
            raise ValueError(f"{name} line {line_no}: {e}") from e
        rows.append((line_no, *values))
    return rows


def table_columns(record_type: str) -> list[str]:
    cols = [c for c, _, _ in RECORDS[record_type][2]]
    if record_type == "LO":
        cols[8:8] = ["latitude", "longitude"]
    return cols


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_zip(service: str) -> Path:
    """Return the cached zip for `service`, downloading only if absent or if its SHA-256 differs
    from the manifest written at download time."""
    if not SERVICE_RE.match(service):
        raise ValueError(f"invalid service code {service!r}")
    folder = RAW_ROOT / service
    zip_path = folder / f"l_{service}.zip"
    manifest = folder / "manifest.json"
    if zip_path.exists():
        sha = sha256_file(zip_path)
        if not manifest.exists():
            manifest.write_text(json.dumps({"sha256": sha}), encoding="utf-8")
            return zip_path
        if json.loads(manifest.read_text(encoding="utf-8"))["sha256"] == sha:
            return zip_path
        print(f"{zip_path.name} does not match its manifest; downloading again")
    folder.mkdir(parents=True, exist_ok=True)
    url = f"{BASE_URL}/l_{service}.zip"
    partial = zip_path.with_suffix(".part")
    with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as out:
        expected = int(response.headers["Content-Length"])
        while chunk := response.read(1 << 20):
            out.write(chunk)
    if partial.stat().st_size != expected:
        raise ValueError(f"download of {url} incomplete: {partial.stat().st_size} of {expected}")
    partial.replace(zip_path)
    manifest.write_text(json.dumps({"sha256": sha256_file(zip_path)}), encoding="utf-8")
    return zip_path


def load(
    conn: Connection, zip_path: Path, service: str, warnings: list[str] | None = None
) -> dict[str, int]:
    """Replace the `service` rows using `conn`'s transaction (caller commits).

    Everything is parsed and validated before the database is touched. Returns rows loaded per
    record type; data warnings (see parse_file) are appended to `warnings`.
    """
    if not SERVICE_RE.match(service):
        raise ValueError(f"invalid service code {service!r}")
    sha256 = sha256_file(zip_path)
    with zipfile.ZipFile(zip_path) as z:
        counts, snapshot = read_counts(z.read("counts").decode("utf-8"))
        missing = [rt for rt in LOAD_ORDER if rt not in counts]
        if missing:
            raise ValueError(f"counts file has no entry for {missing}")
        parsed = {
            rt: parse_file(rt, z.read(f"{rt}.dat"), counts[rt], warnings) for rt in LOAD_ORDER
        }

    for table in ("uls_file_counts", "uls_lo", "uls_fr", "uls_en", "uls_hd"):
        conn.execute(text(f"DELETE FROM {table} WHERE service = :s"), {"s": service})

    driver_conn = conn.connection.driver_connection  # psycopg connection, same transaction
    with driver_conn.cursor() as cur:
        for rt in LOAD_ORDER:
            table = RECORDS[rt][0]
            cols = ", ".join(["service", "line_no", *table_columns(rt)])
            with cur.copy(f"COPY {table} ({cols}) FROM STDIN") as copy:
                for line_no, *values in parsed[rt]:
                    copy.write_row((service, line_no, *values))

    for rt in LOAD_ORDER:
        conn.execute(
            text(
                "INSERT INTO uls_file_counts "
                "(service, record_type, line_count, sha256, snapshot_date) "
                "VALUES (:s, :rt, :n, :sha, :snapshot)"
            ),
            {"s": service, "rt": rt, "n": counts[rt], "sha": sha256, "snapshot": snapshot},
        )

    loaded = {rt: len(parsed[rt]) for rt in LOAD_ORDER}
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
            "name": f"uls_{service}",
            "licence": LICENCE,
            "url": f"{BASE_URL}/l_{service}.zip",
            "access_date": date.today(),
            "row_count": loaded["HD"],  # primary entity: licences
            "notes": NOTES.format(service=service, snapshot=snapshot.isoformat(), sha256=sha256),
        },
    )
    return loaded


def main() -> None:
    parser = argparse.ArgumentParser(description="Load an FCC ULS weekly complete licence file")
    parser.add_argument("service", help="service code, e.g. cell (file complete/l_<service>.zip)")
    service = parser.parse_args().service

    start = time.perf_counter()
    zip_path = ensure_zip(service)
    warnings: list[str] = []
    with get_engine().begin() as conn:
        load(conn, zip_path, service, warnings)
        counts = {
            RECORDS[rt][0]: conn.execute(
                text(f"SELECT count(*) FROM {RECORDS[rt][0]} WHERE service = :s"), {"s": service}
            ).scalar_one()
            for rt in LOAD_ORDER
        }
    elapsed = time.perf_counter() - start
    for warning in warnings:
        print(f"WARNING {warning}")
    print(" ".join(f"{table}={n}" for table, n in counts.items()) + f" elapsed={elapsed:.1f}s")


if __name__ == "__main__":
    main()
