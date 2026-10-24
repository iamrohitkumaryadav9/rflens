"""FCC equipment authorization open data: fcc_grantees, fcc_test_firms

Sources (opendata.fcc.gov, licence "Public Domain U.S. Government"):
- EAS Equipment Authorization Grantee Registrations, dataset 3b3k-34jp (50,153 rows)
- EAS Accredited Test Firms, dataset nubx-v54a (591 rows)
Both snapshots were last updated 2021-03-22, so grantees registered later are absent and every
test-firm expiration_date is long past.

grantee_code is the first part of an FCC ID. It is 3 characters (registered 1998-2013) or 5
characters (2013-2021), uppercase letters and digits.

designation_number is NOT unique in the test-firm data (US5037 and US1273 each belong to two
unrelated firms), so the key is (designation_number, firm_name).

In the source, the text N/A, NA, n/a and blank fields mean "no value" and are stored as NULL. Any
other text is kept as-is, including NIL.

Columns not kept: grantee mailing_address, po_box, zip_code, contact_name; test firm contact,
contact_title, email_address, phone_number, fax_number, address, po_box, mail_stop, zip, location
(personal contact data and street or postal addresses).

Revision ID: 0005
Revises: 0004
Create Date: 2026-10-24
"""

import sqlalchemy as sa
from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "fcc_grantees",
        sa.Column("grantee_code", sa.Text, primary_key=True),
        sa.Column("grantee_name", sa.Text),
        sa.Column("city", sa.Text),
        sa.Column("state", sa.Text),
        sa.Column("country", sa.Text),
        sa.Column("date_received", sa.Date),
        sa.CheckConstraint(
            "grantee_code ~ '^([A-Z0-9]{3}|[A-Z0-9]{5})$'", name="ck_fcc_grantees_code"
        ),
    )
    op.create_index("ix_fcc_grantees_country", "fcc_grantees", ["country"])

    op.create_table(
        "fcc_test_firms",
        sa.Column("designation_number", sa.Text, nullable=False),
        sa.Column("firm_name", sa.Text, nullable=False),
        sa.Column("accreditation", sa.Text),
        sa.Column("mra", sa.Text),
        sa.Column("expiration_date", sa.Date),
        sa.Column("city", sa.Text),
        sa.Column("state", sa.Text),
        sa.Column("country", sa.Text),
        sa.Column("contract", sa.Text),
        sa.PrimaryKeyConstraint("designation_number", "firm_name"),
        sa.CheckConstraint("contract IN ('Y', 'N')", name="ck_fcc_test_firms_contract"),
    )
    op.create_index("ix_fcc_test_firms_country", "fcc_test_firms", ["country"])


def downgrade() -> None:
    op.drop_table("fcc_test_firms")
    op.drop_table("fcc_grantees")
