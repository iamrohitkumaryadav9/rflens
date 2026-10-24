"""rflens_docs_reader: the read-only database role that docs-mcp connects as

The role can log in and SELECT from documents and chunks, nothing else: no other table, no
INSERT/UPDATE/DELETE, no DDL. Defence in depth: its sessions default to read-only transactions and
have a statement timeout.

The password is not in this file. It comes from DOCS_READER_PASSWORD (environment or .env), is
passed to the server as a bind parameter of set_config() (a DO block body cannot take bind
parameters), and is quoted by format('%L') inside the database. To rotate the password on an
existing database run ALTER ROLE rflens_docs_reader PASSWORD '...'.

Revision ID: 0007
Revises: 0006
Create Date: 2026-10-24
"""

import sqlalchemy as sa
from alembic import op

from rflens.config import get_docs_reader_settings

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def _set_vars(**values: str) -> None:
    bind = op.get_bind()
    for name, value in values.items():
        bind.execute(
            sa.text("SELECT set_config(:name, :value, false)"),
            {"name": f"rflens.{name}", "value": value},
        )


def upgrade() -> None:
    settings = get_docs_reader_settings()
    _set_vars(docs_role=settings.docs_reader_user, docs_password=settings.docs_reader_password)
    op.execute(
        """
        DO $$
        DECLARE
            r text := current_setting('rflens.docs_role');
        BEGIN
            IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = r) THEN
                EXECUTE format(
                    'CREATE ROLE %I LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE PASSWORD %L',
                    r, current_setting('rflens.docs_password'));
            END IF;
            EXECUTE format('GRANT CONNECT ON DATABASE %I TO %I', current_database(), r);
            EXECUTE format('GRANT USAGE ON SCHEMA public TO %I', r);
            EXECUTE format('GRANT SELECT ON documents, chunks TO %I', r);
            EXECUTE format('ALTER ROLE %I SET default_transaction_read_only = on', r);
            EXECUTE format('ALTER ROLE %I SET statement_timeout = ''30s''', r);
        END
        $$
        """
    )
    _set_vars(docs_role="", docs_password="")


def downgrade() -> None:
    _set_vars(docs_role=get_docs_reader_settings().docs_reader_user)
    op.execute(
        """
        DO $$
        DECLARE
            r text := current_setting('rflens.docs_role');
        BEGIN
            IF EXISTS (SELECT FROM pg_roles WHERE rolname = r) THEN
                EXECUTE format('REVOKE ALL ON documents, chunks FROM %I', r);
                EXECUTE format('REVOKE ALL ON SCHEMA public FROM %I', r);
                EXECUTE format('REVOKE ALL ON DATABASE %I FROM %I', current_database(), r);
                EXECUTE format('DROP ROLE %I', r);
            END IF;
        END
        $$
        """
    )
    _set_vars(docs_role="")
