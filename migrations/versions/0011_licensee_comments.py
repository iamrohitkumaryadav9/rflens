"""v_uls_licences: say that companies are identified by licensee_frn, not by name

One company (one FRN) can appear under several name spellings or capitalisations, so counting or
ranking licensees by licensee_name splits it. The column comments of licensee_name and licensee_frn
now say to group by licensee_frn alone and show a name with it (for example min(licensee_name)).
Comments only; no data, view or privilege changes. Found when the SQL agent grouped by FRN and
name together (eval run eval-sql-6d590871, question sql-06).

Revision ID: 0011
Revises: 0010
Create Date: 2026-10-25
"""

from alembic import op

# ruff: noqa: E501
# (the strings are text for describe_table; they are not wrapped)

revision = "0011"
down_revision = "0010"
branch_labels = None
depends_on = None

NEW_NAME = "entity_name of the licence's 'L' entity (the lessee for a spectrum lease); the meaning of 'L' is unconfirmed (open question b). The same company (one licensee_frn) can appear under several name spellings or capitalisations, so do not count or rank companies by licensee_name, and do not group by licensee_frn together with licensee_name (that splits one company too): group by licensee_frn alone and show a name with it, for example min(licensee_name)."
NEW_FRN = "FCC Registration Number (frn) of the 'L' entity: it identifies the company. One FRN can appear under several name spellings or capitalisations, so count or rank companies (licensees) by licensee_frn, grouping by licensee_frn alone, and show a name with it, for example min(licensee_name). NULL for some licences: those are not identified by an FRN."

# The comments as 0008 created them.
OLD_NAME = (
    "entity_name of the licence's 'L' entity; the meaning of 'L' is unconfirmed (open question b)."
)
OLD_FRN = "FCC registration number (frn) of the 'L' entity."


def _comment(column: str, text: str) -> None:
    quoted = text.replace("'", "''")
    op.execute(f"COMMENT ON COLUMN v_uls_licences.{column} IS '{quoted}'")


def upgrade() -> None:
    _comment("licensee_name", NEW_NAME)
    _comment("licensee_frn", NEW_FRN)


def downgrade() -> None:
    _comment("licensee_name", OLD_NAME)
    _comment("licensee_frn", OLD_FRN)
