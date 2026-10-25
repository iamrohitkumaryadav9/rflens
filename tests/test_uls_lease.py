"""v_uls_licences.is_lease: spectrum leases (call sign L + 9 digits) vs licences."""

from collections.abc import Iterator

import pytest
from sqlalchemy import Connection, text

from rflens.db import get_engine

SOURCE_URL = "public_access_database_definitions_v9_0.pdf"


@pytest.fixture
def conn() -> Iterator[Connection]:
    with get_engine().connect() as connection:
        transaction = connection.begin()
        try:
            yield connection
        finally:
            transaction.rollback()


def seed(conn: Connection, call_signs: dict[int, str | None]) -> None:
    for i, (usi, call_sign) in enumerate(call_signs.items(), 1):
        conn.execute(
            text(
                "INSERT INTO uls_hd (service, usi, line_no, call_sign, license_status) "
                "VALUES ('zz', :usi, :n, :cs, 'A')"
            ),
            {"usi": usi, "n": i, "cs": call_sign},
        )


def test_is_lease_is_true_only_for_l_plus_nine_digits(conn: Connection) -> None:
    seed(
        conn,
        {
            900001: "L000070487",  # a lease identifier
            900002: "L000007317",  # another (week-1 Verizon lease)
            900003: "KNKA796",  # a licence
            900004: "L00007048",  # L + 8 digits
            900005: "L0000704871",  # L + 10 digits
            900006: "l000070487",  # lower case
            900007: "XL000070487",  # prefix
            900008: "L00007048A",  # not all digits
            900009: None,  # no call sign
        },
    )
    flags = dict(
        conn.execute(
            text("SELECT usi, is_lease FROM v_uls_licences WHERE service = 'zz' ORDER BY usi")
        ).all()
    )
    assert flags == {
        900001: True,
        900002: True,
        900003: False,
        900004: False,
        900005: False,
        900006: False,
        900007: False,
        900008: False,
        900009: False,  # never NULL
    }


def test_is_lease_matches_the_pattern_on_every_row_and_splits_the_counts(conn: Connection) -> None:
    seed(conn, {900001: "L000070487", 900002: "KNKA796", 900003: "KNKN259", 900004: "L000000001"})
    mismatches = conn.execute(
        text(
            "SELECT count(*) FROM v_uls_licences v JOIN uls_hd h USING (service, usi) "
            "WHERE v.is_lease IS DISTINCT FROM (h.call_sign ~ '^L[0-9]{9}$')"
        )
    ).scalar_one()
    assert mismatches == 0
    licences, leases = conn.execute(
        text(
            "SELECT count(*) FILTER (WHERE NOT is_lease), count(*) FILTER (WHERE is_lease) "
            "FROM v_uls_licences WHERE service = 'zz'"
        )
    ).one()
    assert (licences, leases) == (2, 2)


def test_the_view_gained_one_column_at_the_end_and_keeps_its_dependants(conn: Connection) -> None:
    columns = [
        r[0]
        for r in conn.execute(
            text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'v_uls_licences' ORDER BY ordinal_position"
            )
        )
    ]
    assert columns[-2:] == ["licensee_state", "is_lease"] and len(columns) == 17
    frequency_columns = conn.execute(text("SELECT * FROM v_uls_frequencies LIMIT 0")).keys()
    assert len(list(frequency_columns)) == 14  # v_uls_frequencies is unchanged


def test_the_comments_cite_the_source_and_no_longer_call_every_row_a_licence(
    conn: Connection,
) -> None:
    def column_comment(relation: str, column: str) -> str:
        return conn.execute(
            text(
                "SELECT col_description(a.attrelid, a.attnum) FROM pg_attribute a "
                "WHERE a.attrelid = CAST(:r AS regclass) AND a.attname = :c"
            ),
            {"r": relation, "c": column},
        ).scalar_one()

    def relation_comment(relation: str) -> str:
        return conn.execute(
            text("SELECT obj_description(CAST(:r AS regclass), 'pg_class')"), {"r": relation}
        ).scalar_one()

    flag = column_comment("v_uls_licences", "is_lease")
    assert SOURCE_URL in flag and "Call Sign (Lease Id)" in flag and "Lease Link" in flag
    assert "5 (status T" in flag and "not defined in the layout document" in flag
    assert "is_lease = false" in flag
    for comment in (relation_comment("v_uls_licences"), relation_comment("uls_hd")):
        assert "lease" in comment and "2,290 licences" in comment and "3,174" in comment
    for relation in ("v_uls_licences", "uls_hd"):
        assert "lease identifier" in column_comment(relation, "call_sign")


def test_licensee_comments_say_companies_are_identified_by_frn_alone(conn: Connection) -> None:
    """Migration 0011: one FRN can appear under several name spellings."""

    def comment(column: str) -> str:
        return conn.execute(
            text(
                "SELECT col_description(a.attrelid, a.attnum) FROM pg_attribute a "
                "WHERE a.attrelid = 'v_uls_licences'::regclass AND a.attname = :c"
            ),
            {"c": column},
        ).scalar_one()

    for column in ("licensee_name", "licensee_frn"):
        c = comment(column)
        assert "several name spellings or capitalisations" in c
        assert "licensee_frn" in c and "min(licensee_name)" in c
    assert "do not count or rank companies by licensee_name" in comment("licensee_name")
    assert "together with licensee_name" in comment("licensee_name")
    assert "NULL for some licences" in comment("licensee_frn")
    # general advice only: no company names in the comments (they would leak golden answers)
    for column in ("licensee_name", "licensee_frn"):
        assert not any(w in comment(column).lower() for w in ("cellco", "cingular", "alltel"))
