"""Operating-band table parser: synthetic, hand-written HTML (no 3GPP text), strict failures."""

import pytest

from rflens.ingest.spec_bands import BandParseError, parse_spec, parse_table_chunk


def cells(*values: str) -> str:
    return "<tr>" + "".join(f"<td>{v}</td>" for v in values) + "</tr>"


def lte_table(*rows: str) -> str:
    head = (
        '<table><tbody><tr><td rowspan="2">E-UTRA Operating Band</td><td colspan="3">UL</td>'
        '<td colspan="3">DL</td><td rowspan="2">Duplex Mode</td></tr>'
        '<tr><td colspan="3">F<sub>UL_low</sub> - F<sub>UL_high</sub></td>'
        '<td colspan="3">F<sub>DL_low</sub> - F<sub>DL_high</sub></td></tr>'
    )
    return head + "".join(rows) + "</tbody></table>"


def nr_table(*rows: str) -> str:
    head = (
        "<table><thead><tr><th>NR operating band</th><th>UL</th><th>DL</th><th>Duplex Mode</th>"
        "</tr></thead><tbody>"
    )
    return head + "".join(rows) + "</tbody></table>"


def chunk(raw: str, spec: str, clause: str, i: int = 2) -> dict:
    return {
        "chunk_id": f"{spec}@1.0.0/{clause}/{i}",
        "version": "1.0.0",
        "raw_text": raw,
        "clause": clause,
        "kind": "table",
        "position": i,
    }


LTE_ROWS = (
    cells("5", "100 MHz", "-", "110 MHz", "120 MHz", "-", "130MHz", "FDD"),
    cells(
        "6<sup>1</sup>",
        "200 MHz",
        "",
        "210 MHz",
        "220 MHz",
        "–",
        "230 MHz",
        "TDD<sup>9</sup>",
    ),
    '<tr><td>7</td><td colspan="3">N/A</td><td>300 MHz</td><td>-</td><td>310 MHz</td>'
    "<td>SDL</td></tr>",
    '<tr><td>8</td><td colspan="6">Reserved</td><td></td></tr>',
    cells("…", "", "", "", "", "", "", ""),
)


def test_lte_rows_footnotes_na_sides_colspans_and_skips() -> None:
    skipped: list[str] = []
    rows = parse_table_chunk("36.101", chunk(lte_table(*LTE_ROWS), "36.101", "5.5"), skipped)
    assert [r.band for r in rows] == ["5", "6", "7"]
    b5, b6, b7 = rows
    assert (b5.ul_low_mhz, b5.ul_high_mhz, b5.dl_low_mhz, b5.dl_high_mhz) == (100, 110, 120, 130)
    assert b5.duplex_mode == "FDD" and b5.notes == ""
    assert b6.duplex_mode == "TDD" and "footnote ref 1 on the band" in b6.notes
    assert "footnote ref 9 on the duplex mode" in b6.notes and "UL dash missing" in b6.notes
    assert (b7.ul_low_mhz, b7.dl_low_mhz, b7.duplex_mode) == (None, 300, "SDL")
    assert "UL: N/A" in b7.notes
    assert [s.split(":")[0] for s in skipped] == ["36.101 '8'", "36.101 '…'"]


def test_nr_rows() -> None:
    raw = nr_table(
        cells("n5", "100 MHz – 110 MHz", "120 MHz – 130 MHz", "FDD"),
        cells("n80<sup>3</sup>", "100.5 MHz – 110 MHz", "N/A", "SUL"),
    )
    n5, n80 = parse_table_chunk("38.101-1", chunk(raw, "38.101-1", "5.2"))
    assert (n5.band, n5.ul_high_mhz, n5.dl_low_mhz, n5.duplex_mode) == ("n5", 110, 120, "FDD")
    assert (n80.ul_low_mhz, n80.dl_low_mhz, n80.duplex_mode) == (100.5, None, "SUL")
    assert n80.notes == "footnote ref 3 on the band; DL: N/A"


@pytest.mark.parametrize(
    "row",
    [
        cells("n5", "100 MHz – 110", "120 MHz – 130 MHz", "FDD"),  # no unit
        cells("n5", "110 MHz – 100 MHz", "120 MHz – 130 MHz", "FDD"),  # edges reversed
        cells("n5", "100 MHz – 110 MHz", "120 MHz – 130 MHz", "XDD"),  # unknown duplex
        cells("n5", "100 MHz – 110 MHz", "N/A", "FDD"),  # FDD needs a DL side
        cells("5", "100 MHz – 110 MHz", "120 MHz – 130 MHz", "FDD"),  # not an n-band
        cells("n5", "100 MHz – 110 MHz", "FDD"),  # wrong number of cells
    ],
)
def test_a_row_that_does_not_parse_fails_loudly(row) -> None:
    with pytest.raises(BandParseError, match="38.101-1@1.0.0/5.2/2 row"):
        parse_table_chunk("38.101-1", chunk(nr_table(row), "38.101-1", "5.2"))


def test_a_band_that_appears_twice_fails() -> None:
    row = cells("n5", "100 MHz – 110 MHz", "120 MHz – 130 MHz", "FDD")
    chunks = [
        chunk(nr_table(row), "38.101-1", "5.2", 2),
        chunk(nr_table(row), "38.101-1", "5.2", 3),
    ]
    with pytest.raises(BandParseError, match="appears twice"):
        parse_spec("38.101-1", chunks)


def test_pieces_are_read_in_document_order_and_the_notes_piece_is_ignored() -> None:
    first = chunk(
        nr_table(cells("n1", "1 MHz – 2 MHz", "3 MHz – 4 MHz", "FDD")), "38.101-1", "5.2", 2
    )
    second = chunk(
        nr_table(cells("n2", "1 MHz – 2 MHz", "3 MHz – 4 MHz", "FDD")), "38.101-1", "5.2", 3
    )
    notes = {**chunk("Table (notes)\n", "38.101-1", "5.2", 1)}
    rows = parse_spec("38.101-1", [second, notes, first])
    assert [r.band for r in rows] == ["n1", "n2"]
    assert [r.source_chunk_id for r in rows] == [first["chunk_id"], second["chunk_id"]]


# --- the table, the loader and the reader role (real test database) -------------------------------


def synthetic_chunk(spec: str, version: str, clause: str) -> dict:
    body = "synthetic band table"
    return {
        "chunk_id": f"{spec}@{version}/{clause}/1",
        "spec": spec,
        "version": version,
        "release": 19,
        "clause": clause,
        "clause_title": "Title",
        "parent_path": [],
        "parent_titles": [],
        "kind": "table",
        "position": 1,
        "table_caption": None,
        "table_number": None,
        "table_part": None,
        "source_file": "x.md",
        "text": body,
        "raw_text": body,
        "media": [],
        "image_placeholders": 0,
        "forced_split": False,
        "char_count": len(body),
    }


def synthetic_rows():
    from rflens.ingest.spec_bands import BandRow

    lte, nr = "36.101@9.0.0/5.5/1", "38.101-1@9.0.0/5.2/1"
    return {
        "36.101": [
            BandRow("36.101", "9.0.0", "5", 100, 110, 120, 130, "FDD", lte, ""),
            BandRow("36.101", "9.0.0", "29", None, None, 300, 310, "FDD", lte, "UL: N/A"),
        ],
        "38.101-1": [
            BandRow("38.101-1", "9.0.0", "n80", 400, 410, None, None, "SUL", nr, "DL: N/A"),
        ],
    }


@pytest.fixture
def conn():
    """Synthetic source chunks inside a rolled-back transaction of the test database."""
    from rflens.db import get_engine
    from rflens.ingest.spec_fetch import SpecSource
    from rflens.retrieval.index import load_chunks

    with get_engine().connect() as connection:
        transaction = connection.begin()
        try:
            for spec, clause in (("36.101", "5.5"), ("38.101-1", "5.2")):
                source = SpecSource(spec, "x", "t.zip", "9.0.0", 19, "0" * 64)
                load_chunks(
                    connection, source, [synthetic_chunk(spec, "9.0.0", clause)], prune=False
                )
            yield connection
        finally:
            transaction.rollback()


def stored(conn) -> list[tuple]:
    from sqlalchemy import text

    return [
        tuple(r)
        for r in conn.execute(
            text(
                "SELECT spec, band, ul_low_mhz, dl_low_mhz, duplex_mode, has_uplink, has_downlink "
                "FROM spec_operating_bands WHERE version = '9.0.0' ORDER BY spec, band"
            )
        )
    ]


def test_load_is_idempotent_and_derives_the_booleans(conn) -> None:
    from decimal import Decimal

    from rflens.ingest.spec_bands import load_bands

    rows = synthetic_rows()
    assert load_bands(conn, rows) == {"36.101": 2, "38.101-1": 1}
    first = stored(conn)
    assert load_bands(conn, rows) == {"36.101": 2, "38.101-1": 1}  # a second load: same rows
    assert stored(conn) == first and len(first) == 3
    assert first == [
        ("36.101", "29", None, Decimal("300.000"), "FDD", False, True),  # FDD kept as labelled
        ("36.101", "5", Decimal("100.000"), Decimal("120.000"), "FDD", True, True),
        ("38.101-1", "n80", Decimal("400.000"), None, "SUL", True, False),
    ]


def test_load_refuses_rows_whose_source_chunk_does_not_exist(conn) -> None:
    from dataclasses import replace

    from rflens.ingest.spec_bands import BandParseError, load_bands

    rows = synthetic_rows()
    rows["36.101"][0] = replace(rows["36.101"][0], source_chunk_id="36.101@9.0.0/9.9/1")
    with pytest.raises(BandParseError, match="missing from the chunks table"):
        load_bands(conn, rows)


def test_the_table_rejects_a_bad_duplex_mode_and_a_half_pair(conn) -> None:
    from sqlalchemy import text
    from sqlalchemy.exc import IntegrityError

    insert = (
        "INSERT INTO spec_operating_bands (spec, version, band, ul_low_mhz, ul_high_mhz, "
        "duplex_mode, source_chunk_id) VALUES ('36.101', '9.0.0', '1', :lo, :hi, :m, 'x')"
    )
    for params in ({"lo": 1, "hi": 2, "m": "XDD"}, {"lo": 1, "hi": None, "m": "FDD"}):
        with pytest.raises(IntegrityError), conn.begin_nested():
            conn.execute(text(insert), params)


def test_the_sql_reader_role_can_select_the_table_and_cannot_change_it() -> None:
    from sqlalchemy import create_engine, text
    from sqlalchemy.exc import DBAPIError

    from rflens.config import get_sql_mcp_settings

    engine = create_engine(get_sql_mcp_settings().reader_url())
    try:
        with engine.connect() as c:
            c.execute(text("SELECT count(*) FROM spec_operating_bands")).scalar_one()
        for statement in (
            "INSERT INTO spec_operating_bands (spec, version, band, duplex_mode, source_chunk_id) "
            "VALUES ('x', 'x', 'x', 'FDD', 'x')",
            "DELETE FROM spec_operating_bands",
        ):
            with pytest.raises(DBAPIError), engine.begin() as c:
                c.execute(text(statement))
    finally:
        engine.dispose()


def test_the_table_is_commented_for_describe_table() -> None:
    from sqlalchemy import text

    from rflens.db import get_engine

    with get_engine().connect() as c:
        comments = dict(
            c.execute(
                text(
                    "SELECT a.attname, col_description(a.attrelid, a.attnum) FROM pg_attribute a "
                    "WHERE a.attrelid = 'spec_operating_bands'::regclass AND a.attnum > 0"
                )
            ).all()
        )
    assert all(comments.values()), [k for k, v in comments.items() if not v]
    assert "MHz" in comments["ul_low_mhz"] and "labels" in comments["duplex_mode"]
    assert "NOT in the converted source" in comments["notes"]
