"""FCC ULS: uls_hd, uls_en, uls_fr, uls_lo, uls_file_counts

Source: weekly complete licence files, https://data.fcc.gov/download/pub/uls/complete/l_<service>.zip,
pipe-delimited, layouts from the FCC "ULS Data File Format" document (public access database
definitions v9.0). Every table carries `service` (the zip code, e.g. 'cell') so one service can be
replaced without touching the others, and `line_no` (1-based line in the source .dat file).

Missing values are empty fields in the source and are stored as NULL.

Assumptions that the layout document does not settle (see CLAUDE.md "Open data questions"):
- frequency_assigned / frequency_upper_band are assumed MHz (the cellular values run 824-892).
- the unit of power_erp / power_output / eirp is unknown, so the columns are unitless.
- the meaning of license_status, entity_type and class_station_code codes is not defined there.

uls_lo.latitude / longitude are derived: signed decimal degrees from the source degrees, minutes,
seconds and direction fields (S and W negative); NULL if any component is empty.

uls_fr has no natural key: (usi, location_number, antenna_number, freq_seq_id) repeats in the
cellular file, so (service, line_no) is the primary key and the natural key is a plain index.

Columns not kept: HD eligibility, ownership and certification flags, certifier and demographic
fields; EN phone, fax, email, street address, PO box (contact data); FR transmit times, tolerance,
transmitter make and model, unit counts, status fields; LO street address, max lat/long box, radius,
NEPA, quiet zone, airport, unit counts, status fields.

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-24
"""

import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def _hd_fk() -> sa.ForeignKeyConstraint:
    return sa.ForeignKeyConstraint(
        ["service", "usi"], ["uls_hd.service", "uls_hd.usi"], ondelete="CASCADE"
    )


def upgrade() -> None:
    op.create_table(
        "uls_hd",
        sa.Column("service", sa.Text, nullable=False),
        sa.Column("usi", sa.BigInteger, nullable=False),
        sa.Column("line_no", sa.Integer, nullable=False),
        sa.Column("call_sign", sa.Text),
        sa.Column("license_status", sa.Text),
        sa.Column("radio_service_code", sa.Text),
        sa.Column("grant_date", sa.Date),
        sa.Column("expired_date", sa.Date),
        sa.Column("cancellation_date", sa.Date),
        sa.Column("effective_date", sa.Date),
        sa.Column("last_action_date", sa.Date),
        sa.Column("auction_id", sa.Integer),
        sa.PrimaryKeyConstraint("service", "usi"),
    )
    op.create_index("ix_uls_hd_call_sign", "uls_hd", ["call_sign"])

    op.create_table(
        "uls_en",
        sa.Column("service", sa.Text, nullable=False),
        sa.Column("usi", sa.BigInteger, nullable=False),
        sa.Column("line_no", sa.Integer, nullable=False),
        sa.Column("entity_type", sa.Text, nullable=False),
        sa.Column("licensee_id", sa.Text),
        sa.Column("entity_name", sa.Text),
        sa.Column("first_name", sa.Text),
        sa.Column("last_name", sa.Text),
        sa.Column("city", sa.Text),
        sa.Column("state", sa.Text),
        sa.Column("zip_code", sa.Text),
        sa.Column("frn", sa.Text),
        sa.PrimaryKeyConstraint("service", "usi", "entity_type"),
        _hd_fk(),
    )
    op.create_index("ix_uls_en_entity_name", "uls_en", ["entity_name"])

    op.create_table(
        "uls_fr",
        sa.Column("service", sa.Text, nullable=False),
        sa.Column("line_no", sa.Integer, nullable=False),
        sa.Column("usi", sa.BigInteger, nullable=False),
        sa.Column("location_number", sa.Integer),
        sa.Column("antenna_number", sa.Integer),
        sa.Column("freq_seq_id", sa.Integer),
        sa.Column("class_station_code", sa.Text),
        sa.Column("frequency_assigned", sa.Numeric(16, 8)),
        sa.Column("frequency_upper_band", sa.Numeric(16, 8)),
        sa.Column("power_output", sa.Numeric(15, 3)),
        sa.Column("power_erp", sa.Numeric(15, 3)),
        sa.Column("eirp", sa.Numeric(7, 1)),
        sa.PrimaryKeyConstraint("service", "line_no"),
        _hd_fk(),
    )
    op.create_index(
        "ix_uls_fr_natural_key",
        "uls_fr",
        ["service", "usi", "location_number", "antenna_number", "freq_seq_id"],
    )
    op.create_index("ix_uls_fr_frequency_assigned", "uls_fr", ["frequency_assigned"])

    op.create_table(
        "uls_lo",
        sa.Column("service", sa.Text, nullable=False),
        sa.Column("usi", sa.BigInteger, nullable=False),
        sa.Column("line_no", sa.Integer, nullable=False),
        sa.Column("location_number", sa.Integer, nullable=False),
        sa.Column("location_type_code", sa.Text),
        sa.Column("location_class_code", sa.Text),
        sa.Column("location_city", sa.Text),
        sa.Column("location_county", sa.Text),
        sa.Column("location_state", sa.Text),
        sa.Column("ground_elevation", sa.Numeric(7, 1)),
        sa.Column("latitude", sa.Double),
        sa.Column("longitude", sa.Double),
        sa.Column("tower_registration_number", sa.Text),
        sa.Column("height_of_support_structure", sa.Numeric(7, 1)),
        sa.Column("overall_height_of_structure", sa.Numeric(7, 1)),
        sa.Column("structure_type", sa.Text),
        sa.Column("location_name", sa.Text),
        sa.PrimaryKeyConstraint("service", "usi", "location_number"),
        _hd_fk(),
    )
    op.create_index("ix_uls_lo_lat_lon", "uls_lo", ["latitude", "longitude"])

    op.create_table(
        "uls_file_counts",
        sa.Column("service", sa.Text, nullable=False),
        sa.Column("record_type", sa.Text, nullable=False),
        sa.Column("line_count", sa.Integer, nullable=False),
        sa.Column(
            "loaded_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.PrimaryKeyConstraint("service", "record_type"),
    )


def downgrade() -> None:
    op.drop_table("uls_file_counts")
    op.drop_table("uls_lo")
    op.drop_table("uls_fr")
    op.drop_table("uls_en")
    op.drop_table("uls_hd")
