"""uls_file_counts: add sha256 and snapshot_date

Provenance of the zip each row's counts came from. Nullable so rows loaded before this migration
survive; the loader fills both on every load.

Revision ID: 0004
Revises: 0003
Create Date: 2026-10-24
"""

import sqlalchemy as sa
from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("uls_file_counts", sa.Column("sha256", sa.Text))
    op.add_column("uls_file_counts", sa.Column("snapshot_date", sa.Date))


def downgrade() -> None:
    op.drop_column("uls_file_counts", "snapshot_date")
    op.drop_column("uls_file_counts", "sha256")
