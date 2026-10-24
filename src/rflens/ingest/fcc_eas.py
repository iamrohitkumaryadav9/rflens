"""Load the FCC equipment-authorization open-data CSVs (opendata.fcc.gov) into Postgres.

- fcc_grantees   <- EAS Equipment Authorization Grantee Registrations (3b3k-34jp)
- fcc_test_firms <- EAS Accredited Test Firms (nubx-v54a)

CSVs are read with the stdlib csv module, which applies no NA conversion. Only the explicit
PLACEHOLDERS list (plus empty or whitespace-only values) become NULL; everything else, including
NIL, is kept as written. The load raises (ValueError) on a header mismatch, a row with the wrong
field count, an unparseable date, a grantee_code that fails the table's CHECK, or a repeated key.
Personal contact data and street addresses in the files are never read into the database.
"""

import csv
import hashlib
import json
import re
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from datetime import time as dt_time
from pathlib import Path

from sqlalchemy import Connection, text

from rflens.db import get_engine

RAW_DIR = Path(__file__).resolve().parents[3] / "data" / "raw" / "fcc_eas"
CSV_URL = "https://opendata.fcc.gov/api/views/{dataset_id}/rows.csv?accessType=DOWNLOAD"
PAGE_URL = "https://opendata.fcc.gov/d/{dataset_id}"
LICENCE = "US Government Work (https://www.usa.gov/government-works)"
LAST_UPDATED = "2021-03-22"  # rowsUpdatedAt of both datasets, from the portal's API metadata
PLACEHOLDERS = {"N/A", "NA", "n/a"}
CODE_RE = re.compile(r"[A-Z0-9]{3}|[A-Z0-9]{5}")

NOTES = (
    "FCC open data dataset {dataset_id} ({title}), page {page}, CSV {csv_url}. Dataset last "
    "updated {last_updated}: this is a snapshot cut-off, {cutoff_effect}. SHA-256 {sha256}. "
    "Licence on the dataset page: 'Public Domain U.S. Government' "
    "(https://www.usa.gov/government-works). N/A, NA, n/a and blank values are stored as NULL; "
    "other text such as NIL is kept as-is. Personal contact data and street addresses are not "
    "loaded."
)


def _null(v: str) -> str | None:
    v = v.strip()
    return None if v == "" or v in PLACEHOLDERS else v


def _required(v: str) -> str:
    value = _null(v)
    if value is None:
        raise ValueError("required value is empty")
    return value


def _code(v: str) -> str:
    if not CODE_RE.fullmatch(v):
        raise ValueError(f"grantee_code {v!r} is not 3 or 5 uppercase alphanumeric characters")
    return v


def _date(v: str) -> date | None:
    value = _null(v)
    if value is None:
        return None
    parsed = datetime.strptime(value, "%m/%d/%Y %I:%M:%S %p")
    if parsed.time() != dt_time(0, 0):
        raise ValueError(f"date {value!r} has a time of day; expected midnight")
    return parsed.date()


def _yes_no(v: str) -> str:
    value = _null(v)
    if value not in ("Y", "N"):
        raise ValueError(f"contract {v!r} is not Y or N")
    return value


@dataclass(frozen=True)
class Spec:
    dataset_id: str
    title: str
    filename: str
    table: str
    data_source: str
    cutoff_effect: str
    header: list[str]  # exact CSV header
    columns: list[tuple[str, str, Callable[[str], object]]]  # (table column, csv column, convert)
    key: tuple[str, ...]  # table columns forming the primary key


GRANTEES = Spec(
    dataset_id="3b3k-34jp",
    title="EAS Equipment Authorization Grantee Registrations",
    filename="grantees.csv",
    table="fcc_grantees",
    data_source="fcc_eas_grantees",
    cutoff_effect="grantees registered after it are absent",
    header=[
        "grantee_code", "grantee_name", "mailing_address", "po_box", "city", "state", "country",
        "zip_code", "contact_name", "date_received",
    ],
    columns=[
        ("grantee_code", "grantee_code", _code),
        ("grantee_name", "grantee_name", _null),
        ("city", "city", _null),
        ("state", "state", _null),
        ("country", "country", _null),
        ("date_received", "date_received", _date),
    ],
    key=("grantee_code",),
)
TEST_FIRMS = Spec(
    dataset_id="nubx-v54a",
    title="EAS Accredited Test Firms",
    filename="accredited_test_firms.csv",
    table="fcc_test_firms",
    data_source="fcc_eas_test_firms",
    cutoff_effect="all expiration dates are long past and the list reflects accreditation "
    "around 2021",
    header=[
        "firm_name", "location", "accreditation", "mra", "designation_number", "expiration_date",
        "contact", "contact_title", "address", "po_box", "mail_stop", "city", "state", "zip",
        "country", "email_address", "phone_number", "fax_number", "contract",
    ],
    columns=[
        ("designation_number", "designation_number", _required),
        ("firm_name", "firm_name", _required),
        ("accreditation", "accreditation", _null),
        ("mra", "mra", _null),
        ("expiration_date", "expiration_date", _date),
        ("city", "city", _null),
        ("state", "state", _null),
        ("country", "country", _null),
        ("contract", "contract", _yes_no),
    ],
    key=("designation_number", "firm_name"),
)
SPECS = (GRANTEES, TEST_FIRMS)


def parse_csv(spec: Spec, path: Path) -> list[tuple]:
    """Validate and convert one CSV. Rows are in spec.columns order."""
    rows: list[tuple] = []
    seen: set[tuple] = set()
    key_idx = [i for i, (col, _, _) in enumerate(spec.columns) if col in spec.key]
    with path.open(newline="", encoding="utf-8") as f:  # strict UTF-8: raises on bad bytes
        reader = csv.reader(f)
        header = next(reader, None)
        if header != spec.header:
            raise ValueError(f"{path.name}: unexpected header {header}")
        positions = [spec.header.index(src) for _, src, _ in spec.columns]
        for fields in reader:
            where = f"{path.name} line {reader.line_num}"
            if len(fields) != len(spec.header):
                raise ValueError(f"{where}: {len(fields)} fields, expected {len(spec.header)}")
            try:
                row = tuple(
                    convert(fields[pos])
                    for (_, _, convert), pos in zip(spec.columns, positions, strict=True)
                )
            except ValueError as e:
                raise ValueError(f"{where}: {e}") from e
            key = tuple(row[i] for i in key_idx)
            if key in seen:
                raise ValueError(f"{where}: repeated key {key}")
            seen.add(key)
            rows.append(row)
    if not rows:
        raise ValueError(f"{path.name}: no data rows")
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_file(spec: Spec) -> Path:
    """Return the cached CSV, downloading only if absent or if its SHA-256 differs from the
    manifest written at download time."""
    path = RAW_DIR / spec.filename
    manifest_path = RAW_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8")) if manifest_path.exists() else {}
    if path.exists():
        sha = sha256_file(path)
        if spec.filename not in manifest:
            manifest[spec.filename] = sha
            manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
            return path
        if manifest[spec.filename] == sha:
            return path
        print(f"{path.name} does not match its manifest; downloading again")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    url = CSV_URL.format(dataset_id=spec.dataset_id)
    partial = path.with_suffix(".part")
    with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as out:
        expected = response.headers.get("Content-Length")
        while chunk := response.read(1 << 20):
            out.write(chunk)
    if expected is not None and partial.stat().st_size != int(expected):
        raise ValueError(f"download of {url} incomplete")
    partial.replace(path)
    manifest[spec.filename] = sha256_file(path)
    manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return path


def load(conn: Connection, paths: dict[str, Path]) -> dict[str, int]:
    """Replace both tables from `paths` (keyed by table name) using `conn`'s transaction.

    Everything is parsed and validated before the database is touched. Returns rows per table.
    """
    parsed = {spec.table: parse_csv(spec, paths[spec.table]) for spec in SPECS}

    for spec in reversed(SPECS):
        conn.execute(text(f"DELETE FROM {spec.table}"))
    driver_conn = conn.connection.driver_connection  # psycopg connection, same transaction
    with driver_conn.cursor() as cur:
        for spec in SPECS:
            cols = ", ".join(col for col, _, _ in spec.columns)
            with cur.copy(f"COPY {spec.table} ({cols}) FROM STDIN") as copy:
                for row in parsed[spec.table]:
                    copy.write_row(row)

    for spec in SPECS:
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
                "name": spec.data_source,
                "licence": LICENCE,
                "url": PAGE_URL.format(dataset_id=spec.dataset_id),
                "access_date": date.today(),
                "row_count": len(parsed[spec.table]),  # primary entities: grantees, test firms
                "notes": NOTES.format(
                    dataset_id=spec.dataset_id,
                    title=spec.title,
                    page=PAGE_URL.format(dataset_id=spec.dataset_id),
                    csv_url=CSV_URL.format(dataset_id=spec.dataset_id),
                    last_updated=LAST_UPDATED,
                    cutoff_effect=spec.cutoff_effect,
                    sha256=sha256_file(paths[spec.table]),
                ),
            },
        )
    return {table: len(rows) for table, rows in parsed.items()}


def main() -> None:
    start = time.perf_counter()
    paths = {spec.table: ensure_file(spec) for spec in SPECS}
    with get_engine().begin() as conn:
        load(conn, paths)
        counts = {
            spec.table: conn.execute(text(f"SELECT count(*) FROM {spec.table}")).scalar_one()
            for spec in SPECS
        }
    elapsed = time.perf_counter() - start
    print(" ".join(f"{table}={n}" for table, n in counts.items()) + f" elapsed={elapsed:.1f}s")


if __name__ == "__main__":
    main()
