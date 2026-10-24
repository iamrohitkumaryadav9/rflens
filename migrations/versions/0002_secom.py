"""secom_runs and secom_measurements

UCI SECOM (semiconductor fab, not RF). Source label encoding: -1 = pass, 1 = fail.
It is stored here as text 'pass' / 'fail'.
secom_measurements is long format: one row per (run, sensor feature), value NULL where the
source has NaN.

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-24
"""

import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "secom_runs",
        sa.Column("run_id", sa.Integer, primary_key=True, autoincrement=False),
        sa.Column("label", sa.Text, nullable=False),
        sa.Column("run_ts", sa.DateTime, nullable=False),
        sa.CheckConstraint("label IN ('pass', 'fail')", name="ck_secom_runs_label"),
    )
    op.create_index("ix_secom_runs_label", "secom_runs", ["label"])
    op.create_table(
        "secom_measurements",
        sa.Column(
            "run_id",
            sa.Integer,
            sa.ForeignKey("secom_runs.run_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("feature_idx", sa.Integer, nullable=False),
        sa.Column("value", sa.Double),
        sa.PrimaryKeyConstraint("run_id", "feature_idx"),
    )
    op.create_index("ix_secom_measurements_feature_idx", "secom_measurements", ["feature_idx"])


def downgrade() -> None:
    op.drop_table("secom_measurements")
    op.drop_table("secom_runs")
