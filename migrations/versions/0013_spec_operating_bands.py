"""spec_operating_bands: the operating-band tables of the UE specs as rows

One row per band of Table 5.5-1 (TS 36.101) and Table 5.2-1 (TS 38.101-1), parsed from the spec
chunks by `make load-spec-bands`. The band edges are structured facts, so the orchestrator looks
them up (through sql-mcp) instead of searching the large tables semantically. The rows are
3GPP-derived: they live in the local database only and are never committed. rflens_sql_reader gets
SELECT; sql_guard.ALLOWED_TABLES lists the table. Footnote texts are not in the converted source
(the notes chunk of each table holds only a heading), so only footnote markers are stored.

Revision ID: 0013
Revises: 0012
Create Date: 2026-10-26
"""

import sqlalchemy as sa
from alembic import op

from rflens.config import get_sql_mcp_settings

# ruff: noqa: E501
# (the strings are text for describe_table; they are not wrapped)

revision = "0013"
down_revision = "0012"
branch_labels = None
depends_on = None

COMMENTS = {
    "table": "Operating bands of the 3GPP UE specs, one row per band: Table 5.5-1 of TS 36.101 (LTE / E-UTRA, bands as plain numbers such as 5) and Table 5.2-1 of TS 38.101-1 (NR FR1, bands as n-numbers such as n5). Parsed from the spec chunks; 3GPP-derived, so it exists in the local database only. Key (spec, version, band). To get a band's edges filter on spec and band, for example spec = '36.101' AND band = '5'.",
    "spec": "Specification number: 36.101 (LTE) or 38.101-1 (NR). The same frequency range can be a band in both (LTE band 5 and NR n5 have the same edges).",
    "version": "Specification version the row was parsed from, for example 19.6.0.",
    "band": "Band label as the spec writes it: a number for LTE (5), n-number for NR (n5). The footnote superscript is not part of the label.",
    "ul_low_mhz": "Lower edge of the uplink operating band (FUL_low), in MHz as stated in the spec table. NULL when the spec table says N/A for the uplink (has_uplink is false).",
    "ul_high_mhz": "Upper edge of the uplink operating band (FUL_high), in MHz as stated in the spec table. NULL when the uplink is N/A.",
    "dl_low_mhz": "Lower edge of the downlink operating band (FDL_low), in MHz as stated in the spec table. NULL when the downlink is N/A (supplementary uplink, has_downlink is false).",
    "dl_high_mhz": "Upper edge of the downlink operating band (FDL_high), in MHz as stated in the spec table. NULL when the downlink is N/A.",
    "duplex_mode": "Duplex mode exactly as the spec table labels it: FDD, TDD, SDL (supplemental downlink) or SUL (supplemental uplink). Some LTE bands that have no uplink (29, 32, 67, 69, 75, 76) are labelled FDD in TS 36.101; that label is kept. Use has_uplink and has_downlink for which sides exist.",
    "has_uplink": "Derived: true when the uplink edges are present (ul_low_mhz is not NULL).",
    "has_downlink": "Derived: true when the downlink edges are present (dl_low_mhz is not NULL).",
    "source_chunk_id": "The spec chunk (chunks.chunk_id) holding this band's table row, for citations. Read it through docs-mcp get_chunk.",
    "notes": "Footnote reference numbers printed with the band or the duplex mode (the numbers only: the footnote texts are NOT in the converted source, the table's notes chunk holds just a heading), which side is N/A, and parsing remarks such as a dash missing in the source. Empty when there is nothing to say.",
}


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE spec_operating_bands (
            spec text NOT NULL,
            version text NOT NULL,
            band text NOT NULL,
            ul_low_mhz numeric(9, 3),
            ul_high_mhz numeric(9, 3),
            dl_low_mhz numeric(9, 3),
            dl_high_mhz numeric(9, 3),
            duplex_mode text NOT NULL,
            has_uplink boolean GENERATED ALWAYS AS (ul_low_mhz IS NOT NULL) STORED,
            has_downlink boolean GENERATED ALWAYS AS (dl_low_mhz IS NOT NULL) STORED,
            source_chunk_id text NOT NULL,
            notes text NOT NULL DEFAULT '',
            PRIMARY KEY (spec, version, band),
            CONSTRAINT spec_operating_bands_duplex CHECK (duplex_mode IN ('FDD', 'TDD', 'SDL', 'SUL')),
            CONSTRAINT spec_operating_bands_ul_pair CHECK ((ul_low_mhz IS NULL) = (ul_high_mhz IS NULL)),
            CONSTRAINT spec_operating_bands_dl_pair CHECK ((dl_low_mhz IS NULL) = (dl_high_mhz IS NULL))
        )
        """
    )
    for key, text in COMMENTS.items():
        quoted = text.replace("'", "''")
        target = (
            "TABLE spec_operating_bands" if key == "table" else f"COLUMN spec_operating_bands.{key}"
        )
        op.execute(f"COMMENT ON {target} IS '{quoted}'")
    bind = op.get_bind()
    bind.execute(
        sa.text("SELECT set_config('rflens.reader_role', :r, false)"),
        {"r": get_sql_mcp_settings().sql_reader_user},
    )
    op.execute(
        """
        DO $$
        DECLARE r text := current_setting('rflens.reader_role');
        BEGIN
            IF EXISTS (SELECT FROM pg_roles WHERE rolname = r) THEN
                EXECUTE format('GRANT SELECT ON spec_operating_bands TO %I', r);
            END IF;
        END
        $$
        """
    )
    bind.execute(sa.text("SELECT set_config('rflens.reader_role', '', false)"))


def downgrade() -> None:
    op.execute("DROP TABLE spec_operating_bands")
