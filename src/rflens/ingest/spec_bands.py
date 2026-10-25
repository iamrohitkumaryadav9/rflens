"""Operating-band tables of the 3GPP UE specs -> one row per band.

    uv run python -m rflens.ingest.spec_bands        # parse, check and print a report (no writes)

Sources are the table chunks the chunker already wrote (`data/interim/3gpp/<spec>/chunks.jsonl`),
parsed from `raw_text` (HTML): Table 5.5-1 of TS 36.101 (clause 5.5) and Table 5.2-1 of TS 38.101-1
(clause 5.2), every piece of the split table. Nothing here is committed or sent anywhere: the rows
are 3GPP-derived and stay in the local database (the specs are copyrighted, see CLAUDE.md).

Parsing is strict. A row that is not a known header, a reserved or ellipsis row, or a band row that
parses completely (numbers, MHz, a duplex mode, ordered edges) raises `BandParseError` naming the
chunk and the row. Footnote *markers* (superscripts) are kept as notes; the footnote texts are not
in the converted source, so they are not available.
"""

import json
import re
import sys
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
INTERIM = ROOT / "data" / "interim" / "3gpp"
TABLES = {  # spec -> (clause, band label pattern, table name)
    "36.101": ("5.5", re.compile(r"^\d{1,3}$"), "Table 5.5-1"),
    "38.101-1": ("5.2", re.compile(r"^n\d{1,3}$"), "Table 5.2-1"),
}
DUPLEX = {"FDD", "TDD", "SDL", "SUL"}
DASHES = "-‐‑‒–—"
_NUM = r"(\d+(?:\.\d+)?)"
SINGLE_RE = re.compile(rf"^{_NUM}\s*MHz$")
RANGE_RE = re.compile(rf"^{_NUM}\s*MHz\s*[{DASHES}]\s*{_NUM}\s*MHz$")


class BandParseError(ValueError):
    pass


@dataclass(frozen=True)
class BandRow:
    spec: str
    version: str
    band: str
    ul_low_mhz: float | None
    ul_high_mhz: float | None
    dl_low_mhz: float | None
    dl_high_mhz: float | None
    duplex_mode: str
    source_chunk_id: str
    notes: str


class _Cell:
    def __init__(self, colspan: int = 1) -> None:
        self.text: list[str] = []
        self.sup: list[str] = []
        self.colspan = colspan

    def expand(self) -> list["_Cell"]:
        """A cell spanning k columns becomes k cells with the same text (superscripts on the
        first only), so every row has one cell per column."""
        cells = [self]
        for _ in range(self.colspan - 1):
            copy = _Cell()
            copy.text = list(self.text)
            cells.append(copy)
        return cells


class _TableParser(HTMLParser):
    """Rows of cells; text inside <sup> is kept apart (footnote markers). `header` flags a row that
    uses <th> or sits in <thead>."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[tuple[bool, list[_Cell]]] = []
        self._row: list[_Cell] | None = None
        self._header = False
        self._in_thead = False
        self._cell: _Cell | None = None
        self._sup = False

    def handle_starttag(self, tag, attrs):
        if tag == "thead":
            self._in_thead = True
        elif tag == "tr":
            self._row, self._header = [], self._in_thead
        elif tag in ("td", "th") and self._row is not None:
            span = dict(attrs).get("colspan") or "1"
            self._cell = _Cell(int(span) if span.isdigit() else 1)
            self._header = self._header or tag == "th"
        elif tag == "sup":
            self._sup = True
        elif tag in ("br", "p") and self._cell is not None:
            self._cell.text.append(" ")

    def handle_endtag(self, tag):
        if tag == "thead":
            self._in_thead = False
        elif tag in ("td", "th") and self._cell is not None and self._row is not None:
            self._row.extend(self._cell.expand())
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self.rows.append((self._header, self._row))
            self._row = None
        elif tag == "sup":
            self._sup = False

    def handle_data(self, data):
        if self._cell is not None:
            (self._cell.sup if self._sup else self._cell.text).append(data)


def _clean(parts: list[str]) -> str:
    return re.sub(r"\s+", " ", "".join(parts).replace("\xa0", " ")).strip()


def _single(text: str, where: str) -> float:
    m = SINGLE_RE.match(text)
    if not m:
        raise BandParseError(f"{where}: {text!r} is not '<number> MHz'")
    return float(m.group(1))


def _is_na(text: str) -> bool:
    return text.strip().upper() == "N/A"


def _check_row(row: BandRow, where: str) -> None:
    if row.duplex_mode not in DUPLEX:
        raise BandParseError(f"{where}: unknown duplex mode {row.duplex_mode!r}")
    for low, high, side in (
        (row.ul_low_mhz, row.ul_high_mhz, "UL"),
        (row.dl_low_mhz, row.dl_high_mhz, "DL"),
    ):
        if (low is None) != (high is None):
            raise BandParseError(f"{where}: {side} has only one edge")
        if low is not None and not 0 < low < high:
            raise BandParseError(f"{where}: {side} edges {low}-{high} are not ordered")
    has_ul, has_dl = row.ul_low_mhz is not None, row.dl_low_mhz is not None
    # the specs label a band whose UL is N/A "FDD" in some rows (e.g. a DL-only band paired by CA)
    expected = {
        "FDD": {(True, True), (False, True)},
        "TDD": {(True, True)},
        "SDL": {(False, True)},
        "SUL": {(True, False)},
    }
    if (has_ul, has_dl) not in expected[row.duplex_mode]:
        raise BandParseError(f"{where}: {row.duplex_mode} with UL={has_ul}, DL={has_dl}")


def _notes(band_sup: str, duplex_sup: str, row: BandRow) -> str:
    parts = []
    if band_sup:
        parts.append(f"footnote ref {band_sup} on the band")
    if duplex_sup:
        parts.append(f"footnote ref {duplex_sup} on the duplex mode")
    if row.ul_low_mhz is None:
        parts.append("UL: N/A")
    if row.dl_low_mhz is None:
        parts.append("DL: N/A")
    return "; ".join(parts)


def _lte_row(cells: list[_Cell], meta: tuple[str, str, str], where: str) -> BandRow | None:
    texts = [_clean(c.text) for c in cells]
    if len(cells) != 8:
        raise BandParseError(f"{where}: {len(cells)} cells, expected 8: {texts}")
    spec, version, chunk_id = meta

    def edges(group: list[str], what: str) -> tuple[float | None, float | None]:
        if any(_is_na(t) for t in group):
            if any(t for t in group if not _is_na(t)):
                raise BandParseError(f"{where}: {what} mixes N/A with values: {group}")
            return None, None
        if group[1] == "":  # the converted source lost the dash; both edges still parse below
            extra.append(f"{what} dash missing in the source")
        elif not re.fullmatch(f"[{DASHES}]", group[1]):
            raise BandParseError(f"{where}: {what} separator {group[1]!r}")
        return _single(group[0], where), _single(group[2], where)

    extra: list[str] = []

    ul = edges(texts[1:4], "UL")
    dl = edges(texts[4:7], "DL")
    row = BandRow(
        spec, version, texts[0], ul[0], ul[1], dl[0], dl[1], texts[7].upper(), chunk_id, ""
    )
    notes = "; ".join(
        p for p in (_notes(_clean(cells[0].sup), _clean(cells[7].sup), row), *extra) if p
    )
    return BandRow(**{**asdict(row), "notes": notes})


def _nr_row(cells: list[_Cell], meta: tuple[str, str, str], where: str) -> BandRow:
    texts = [_clean(c.text) for c in cells]
    if len(cells) != 4:
        raise BandParseError(f"{where}: {len(cells)} cells, expected 4: {texts}")
    spec, version, chunk_id = meta

    def edges(text: str, what: str) -> tuple[float | None, float | None]:
        if _is_na(text):
            return None, None
        m = RANGE_RE.match(text)
        if not m:
            raise BandParseError(f"{where}: {what} {text!r} is not '<n> MHz - <n> MHz' or N/A")
        return float(m.group(1)), float(m.group(2))

    ul, dl = edges(texts[1], "UL"), edges(texts[2], "DL")
    row = BandRow(
        spec, version, texts[0], ul[0], ul[1], dl[0], dl[1], texts[3].upper(), chunk_id, ""
    )
    return BandRow(
        **{**asdict(row), "notes": _notes(_clean(cells[0].sup), _clean(cells[3].sup), row)}
    )


def _is_lte_header(texts: list[str]) -> bool:
    joined = " ".join(texts)
    return "Operating Band" in joined or "UL_low" in joined or "DL_low" in joined


def _skip_reason(texts: list[str]) -> str | None:
    """Rows that are in the table but carry no band: the ellipsis row and reserved bands."""
    if texts[0].startswith("…") or texts[0] == "...":
        return "ellipsis row"
    if any(t.lower() == "reserved" for t in texts[1:]):
        return "reserved"
    return None


def parse_table_chunk(spec: str, chunk: dict, skipped: list[str] | None = None) -> list[BandRow]:
    parser = _TableParser()
    parser.feed(chunk["raw_text"])
    meta = (spec, chunk["version"], chunk["chunk_id"])
    label = TABLES[spec][1]
    out: list[BandRow] = []
    for number, (header, cells) in enumerate(parser.rows, start=1):
        where = f"{chunk['chunk_id']} row {number}"
        texts = [_clean(c.text) for c in cells]
        if not cells:
            raise BandParseError(f"{where}: empty row")
        if header or (spec == "36.101" and _is_lte_header(texts)):
            continue
        reason = _skip_reason(texts)
        if reason:
            if skipped is not None:
                skipped.append(f"{spec} {texts[0]!r}: {reason} ({chunk['chunk_id']})")
            continue
        row = _nr_row(cells, meta, where) if spec == "38.101-1" else _lte_row(cells, meta, where)
        if not label.match(row.band):
            raise BandParseError(f"{where}: band label {row.band!r} does not look like a band")
        _check_row(row, where)
        out.append(row)
    return out


def load_chunks(spec: str, interim: Path = INTERIM) -> list[dict]:
    path = interim / spec / "chunks.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `make chunk-3gpp` first")
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def table_chunks(spec: str, chunks: list[dict]) -> list[dict]:
    """Every piece of the operating-bands table, in document order (the notes piece, which holds
    only a heading, has no <table>)."""
    clause = TABLES[spec][0]
    found = [
        c
        for c in chunks
        if c["clause"] == clause
        and c["kind"] == "table"
        and c["raw_text"].lstrip().startswith("<table")
    ]
    if not found:
        raise BandParseError(f"{spec}: no table chunk in clause {clause}")
    return sorted(found, key=lambda c: c["position"])


def parse_spec(
    spec: str, chunks: list[dict] | None = None, skipped: list[str] | None = None
) -> list[BandRow]:
    rows: list[BandRow] = []
    for chunk in table_chunks(spec, chunks if chunks is not None else load_chunks(spec)):
        rows += parse_table_chunk(spec, chunk, skipped)
    seen: set[str] = set()
    for row in rows:
        if row.band in seen:
            raise BandParseError(f"{spec}: band {row.band} appears twice")
        seen.add(row.band)
    return rows


# --- report ---------------------------------------------------------------------------------------


def _fmt(row: BandRow) -> str:
    def span(low, high):
        return "N/A" if low is None else f"{low:g}-{high:g}"

    return (
        f"{row.band:>5} UL {span(row.ul_low_mhz, row.ul_high_mhz):>15} "
        f"DL {span(row.dl_low_mhz, row.dl_high_mhz):>15} {row.duplex_mode}  "
        f"{row.source_chunk_id}" + (f"  [{row.notes}]" if row.notes else "")
    )


def checks(
    rows_by_spec: dict[str, list[BandRow]], existing_ids: set[str]
) -> list[tuple[str, bool]]:
    def get(spec: str, band: str) -> BandRow | None:
        return next((r for r in rows_by_spec[spec] if r.band == band), None)

    def same(r: BandRow | None, ul, dl, duplex) -> bool:
        return (
            r is not None
            and (r.ul_low_mhz, r.ul_high_mhz) == ul
            and (r.dl_low_mhz, r.dl_high_mhz) == dl
            and r.duplex_mode == duplex
        )

    b5, n5, n78 = get("36.101", "5"), get("38.101-1", "n5"), get("38.101-1", "n78")
    all_rows = [r for rows in rows_by_spec.values() for r in rows]
    return [
        ("band 5 is UL 824-849 / DL 869-894 FDD", same(b5, (824, 849), (869, 894), "FDD")),
        (
            "n5 matches band 5",
            n5 is not None
            and b5 is not None
            and (n5.ul_low_mhz, n5.ul_high_mhz, n5.dl_low_mhz, n5.dl_high_mhz, n5.duplex_mode)
            == (b5.ul_low_mhz, b5.ul_high_mhz, b5.dl_low_mhz, b5.dl_high_mhz, b5.duplex_mode),
        ),
        ("n78 is 3300-3800 TDD (UL = DL)", same(n78, (3300, 3800), (3300, 3800), "TDD")),
        (
            f"every source_chunk_id exists ({len({r.source_chunk_id for r in all_rows})} ids)",
            all(r.source_chunk_id in existing_ids for r in all_rows),
        ),
    ]


INSERT_SQL = """
    INSERT INTO spec_operating_bands (spec, version, band, ul_low_mhz, ul_high_mhz, dl_low_mhz,
        dl_high_mhz, duplex_mode, source_chunk_id, notes)
    VALUES (:spec, :version, :band, :ul_low_mhz, :ul_high_mhz, :dl_low_mhz, :dl_high_mhz,
        :duplex_mode, :source_chunk_id, :notes)
"""


def load_bands(conn, rows_by_spec: dict[str, list[BandRow]]) -> dict[str, int]:
    """Replace the rows of each given spec in one transaction (the caller's), so a second load
    leaves the same rows. Every source chunk must exist in `chunks`. Returns rows per spec."""
    from sqlalchemy import text

    wanted = {r.source_chunk_id for rows in rows_by_spec.values() for r in rows}
    have = {
        r[0]
        for r in conn.execute(
            text("SELECT chunk_id FROM chunks WHERE chunk_id = ANY(CAST(:ids AS text[]))"),
            {"ids": sorted(wanted)},
        )
    }
    if wanted - have:
        raise BandParseError(
            f"source chunks missing from the chunks table: {sorted(wanted - have)}"
        )
    counts: dict[str, int] = {}
    for spec, rows in rows_by_spec.items():
        conn.execute(text("DELETE FROM spec_operating_bands WHERE spec = :s"), {"s": spec})
        conn.execute(text(INSERT_SQL), [asdict(r) for r in rows])
        counts[spec] = len(rows)
    return counts


def load_main() -> int:
    from rflens.db import analyze, get_engine

    rows_by_spec = {spec: parse_spec(spec) for spec in TABLES}
    with get_engine().begin() as conn:
        counts = load_bands(conn, rows_by_spec)
    analyze(["spec_operating_bands"])
    with get_engine().connect() as conn:
        from sqlalchemy import text

        total = conn.execute(text("SELECT count(*) FROM spec_operating_bands")).scalar_one()
    for spec, n in counts.items():
        print(f"spec_operating_bands: {spec} {n} rows")
    print(f"spec_operating_bands: {total} rows in the table")
    return 0


def main() -> int:
    from sqlalchemy import text

    from rflens.db import get_engine

    if "--load" in sys.argv[1:]:
        return load_main()

    rows_by_spec: dict[str, list[BandRow]] = {}
    skipped: list[str] = []
    ids: set[str] = set()
    for spec in TABLES:
        chunks = load_chunks(spec)
        ids |= {c["chunk_id"] for c in chunks}
        rows_by_spec[spec] = parse_spec(spec, chunks, skipped)
    with get_engine().connect() as conn:
        in_db = {r[0] for r in conn.execute(text("SELECT chunk_id FROM chunks"))}
    print("counts per spec:")
    for spec, rows in rows_by_spec.items():
        by_mode: dict[str, int] = {}
        for r in rows:
            by_mode[r.duplex_mode] = by_mode.get(r.duplex_mode, 0) + 1
        print(
            f"  {spec}: {len(rows)} bands {by_mode}, "
            f"{len({r.source_chunk_id for r in rows})} chunks"
        )
    print(f"\nrows skipped (no band): {len(skipped)}")
    for line in skipped:
        print("  " + line)
    print("\nrows with notes:")
    for rows in rows_by_spec.values():
        for r in rows:
            if r.notes:
                print(f"  {r.spec:8} {_fmt(r)}")
    print("\nchecks:")
    results = checks(rows_by_spec, ids)
    results.append(
        (
            "every source_chunk_id exists in the chunks table",
            all(r.source_chunk_id in in_db for rows in rows_by_spec.values() for r in rows),
        )
    )
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    for spec, rows in rows_by_spec.items():
        print(f"\nfirst 10 rows, {spec} (V{rows[0].version}):")
        for r in rows[:10]:
            print("  " + _fmt(r))
    return 0 if all(ok for _, ok in results) else 1


if __name__ == "__main__":
    sys.exit(main())
