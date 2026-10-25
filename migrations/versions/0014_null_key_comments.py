"""Column comments: a NULL key is not an entity

In the first end-to-end run the SQL agent counted the NULL country of fcc_grantees as a country
("Unknown", 158 grantees, first place of a ranking) and the NULL FRN of v_uls_licences as a licensee
(41 expired licences). The comments of the three key columns now say that rankings and counts of
entities leave out NULL keys unless the question asks about missing values. Comments only; no data,
view or privilege changes.

Revision ID: 0014
Revises: 0013
Create Date: 2026-10-26
"""

from alembic import op

# ruff: noqa: E501
# (the strings are text for describe_table; they are not wrapped)

revision = "0014"
down_revision = "0013"
branch_labels = None
depends_on = None

COLUMNS = (
    ("fcc_grantees", "country"),
    ("fcc_test_firms", "country"),
    ("v_uls_licences", "licensee_frn"),
)
NOTE = " A NULL key is not an entity: a ranking or a count of entities (countries, companies) leaves out rows where this column is NULL (add `{column} IS NOT NULL`), unless the question asks about missing values."


def _note(column: str) -> str:
    return NOTE.format(column=column)


def _current(table: str, column: str) -> str:
    row = (
        op.get_bind()
        .exec_driver_sql(
            "SELECT col_description(a.attrelid, a.attnum) FROM pg_attribute a "
            "WHERE a.attrelid = %s::regclass AND a.attname = %s",
            (table, column),
        )
        .scalar()
    )
    return row or ""


def _set(table: str, column: str, text: str) -> None:
    quoted = text.replace("'", "''").replace("%", "%%")
    op.get_bind().exec_driver_sql(f"COMMENT ON COLUMN {table}.{column} IS '{quoted}'")


def upgrade() -> None:
    for table, column in COLUMNS:
        text = _current(table, column)
        if _note(column) not in text:
            _set(table, column, text.rstrip() + _note(column))


def downgrade() -> None:
    for table, column in COLUMNS:
        _set(table, column, _current(table, column).replace(_note(column), ""))
