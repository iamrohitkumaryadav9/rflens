"""Orchestrator storage: the checkpoints schema, rflens_checkpointer, and rflens_llm_ledger

1. Schema `checkpoints` with the four tables of langgraph-checkpoint-postgres, created by the
   library's own PostgresSaver.setup() under the owner. setup() needs an autocommit connection
   (CREATE INDEX CONCURRENTLY), so it runs on its own connection, not in Alembic's transaction,
   and it is idempotent. The running graph never calls setup(): it fails for a role without
   CREATE on the schema. When the library is upgraded and adds migrations, a new Alembic
   revision must call setup() again (a test fails until it does).
2. rflens_checkpointer: LOGIN, CONNECT, USAGE on schema checkpoints, SELECT/INSERT/UPDATE/DELETE
   on its four tables, search_path fixed to `checkpoints` on the role. Nothing else: it cannot
   read any other table or create anything.
3. rflens_llm_ledger: the role the LLM gateway's ledger connects as: SELECT/INSERT/UPDATE on
   llm_calls and the use of its id sequence. Nothing else.

Passwords come from CHECKPOINTER_PASSWORD and LLM_LEDGER_PASSWORD (environment or .env) via
set_config() and format('%L'), as in migrations 0007 and 0008. Roles are cluster-wide: if a role
already exists (the other database of the same server) it is kept and only the grants are applied
to this database.

Revision ID: 0012
Revises: 0011
Create Date: 2026-10-25
"""

import psycopg
import sqlalchemy as sa
from alembic import op
from langgraph.checkpoint.postgres import PostgresSaver
from sqlalchemy.engine import make_url

from rflens.config import CheckpointerSettings, LlmSettings, get_settings

revision = "0012"
down_revision = "0011"
branch_labels = None
depends_on = None

SCHEMA = "checkpoints"
LIBRARY_TABLES = ("checkpoints", "checkpoint_blobs", "checkpoint_writes", "checkpoint_migrations")
GRANT_TABLES = ", ".join(f"{SCHEMA}.{table}" for table in LIBRARY_TABLES)


def _owner_conninfo(search_path: str | None = None) -> str:
    url = make_url(get_settings().database_url).set(drivername="postgresql")
    if search_path:
        url = url.set(query={"options": f"-csearch_path={search_path}"})
    return url.render_as_string(hide_password=False)


def _set_vars(**values: str) -> None:
    bind = op.get_bind()
    for name, value in values.items():
        bind.execute(
            sa.text("SELECT set_config(:name, :value, false)"),
            {"name": f"rflens.{name}", "value": value},
        )


def upgrade() -> None:
    checkpointer, ledger = CheckpointerSettings(), LlmSettings()
    if ledger.llm_ledger_password is None:
        raise RuntimeError("LLM_LEDGER_PASSWORD is not set; migration 0012 creates that role")

    # 1. the schema and the library's tables, committed before the grants below need them
    with psycopg.connect(_owner_conninfo(), autocommit=True) as conn:
        conn.execute(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")
    with PostgresSaver.from_conn_string(_owner_conninfo(SCHEMA)) as saver:
        saver.setup()

    # 2 and 3. roles and grants
    _set_vars(
        cp_role=checkpointer.checkpointer_user,
        cp_password=checkpointer.checkpointer_password.get_secret_value(),
        lg_role=ledger.llm_ledger_user,
        lg_password=ledger.llm_ledger_password.get_secret_value(),
    )
    op.execute(
        f"""
        DO $$
        DECLARE
            cp text := current_setting('rflens.cp_role');
            lg text := current_setting('rflens.lg_role');
        BEGIN
            IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = cp) THEN
                EXECUTE format(
                    'CREATE ROLE %I LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE PASSWORD %L',
                    cp, current_setting('rflens.cp_password'));
            END IF;
            IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = lg) THEN
                EXECUTE format(
                    'CREATE ROLE %I LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE PASSWORD %L',
                    lg, current_setting('rflens.lg_password'));
            END IF;

            EXECUTE format('GRANT CONNECT ON DATABASE %I TO %I, %I', current_database(), cp, lg);

            EXECUTE format('GRANT USAGE ON SCHEMA {SCHEMA} TO %I', cp);
            EXECUTE format('GRANT SELECT, INSERT, UPDATE, DELETE ON {GRANT_TABLES} TO %I', cp);
            EXECUTE format('ALTER ROLE %I SET search_path = {SCHEMA}', cp);

            EXECUTE format('GRANT USAGE ON SCHEMA public TO %I', lg);
            EXECUTE format('GRANT SELECT, INSERT, UPDATE ON llm_calls TO %I', lg);
            EXECUTE format('GRANT USAGE, SELECT ON SEQUENCE llm_calls_id_seq TO %I', lg);
        END
        $$
        """
    )
    _set_vars(cp_role="", cp_password="", lg_role="", lg_password="")


def downgrade() -> None:
    _set_vars(
        cp_role=CheckpointerSettings.model_fields["checkpointer_user"].default,
        lg_role=LlmSettings.model_fields["llm_ledger_user"].default,
    )
    op.execute(
        f"""
        DO $$
        DECLARE r text;
        BEGIN
            FOREACH r IN ARRAY ARRAY[current_setting('rflens.cp_role'),
                                     current_setting('rflens.lg_role')] LOOP
                IF EXISTS (SELECT FROM pg_roles WHERE rolname = r) THEN
                    EXECUTE format('REVOKE ALL ON SEQUENCE llm_calls_id_seq FROM %I', r);
                    EXECUTE format('REVOKE ALL ON llm_calls FROM %I', r);
                    EXECUTE format('REVOKE ALL ON SCHEMA public FROM %I', r);
                    EXECUTE format('REVOKE ALL ON SCHEMA {SCHEMA} FROM %I', r);
                    EXECUTE format('REVOKE ALL ON DATABASE %I FROM %I', current_database(), r);
                END IF;
            END LOOP;
        END
        $$
        """
    )
    op.execute(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE")
    # Roles are cluster-wide: drop them unless the other database still uses them.
    op.execute(
        """
        DO $$
        DECLARE r text;
        BEGIN
            FOREACH r IN ARRAY ARRAY[current_setting('rflens.cp_role'),
                                     current_setting('rflens.lg_role')] LOOP
                BEGIN
                    EXECUTE format('DROP ROLE IF EXISTS %I', r);
                EXCEPTION WHEN dependent_objects_still_exist THEN
                    RAISE NOTICE 'role % is still used in another database; kept', r;
                END;
            END LOOP;
        END
        $$
        """
    )
    _set_vars(cp_role="", lg_role="")
