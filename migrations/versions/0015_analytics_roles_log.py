"""analytics-mcp: its read-only data role, its INSERT-only log role and the call log

rflens_analytics_reader: login role with SELECT on the SECOM tables only (secom_runs,
secom_measurements, v_secom_values), read-only sessions, a statement timeout. analytics-mcp reads
the extract that goes into the sandbox through it; the sandbox container never gets a credential.

rflens_analytics_logger: login role with INSERT on analytics_call_log and nothing else. Every call
to every analytics-mcp tool is a row there, written through this role, so the reader role cannot
touch the log. Passwords come from ANALYTICS_READER_PASSWORD / ANALYTICS_LOGGER_PASSWORD (.env, CI)
and are set with format('%L') inside the database. To rotate: ALTER ROLE <role> PASSWORD '...'.

Revision ID: 0015
Revises: 0014
Create Date: 2026-10-26
"""

import sqlalchemy as sa
from alembic import op

from rflens.config import get_analytics_settings

revision = "0015"
down_revision = "0014"
branch_labels = None
depends_on = None

READER_TABLES = "secom_runs, secom_measurements, v_secom_values"
TOOLS = (
    "list_analyses",
    "describe",
    "missing_summary",
    "distribution",
    "group_compare",
    "correlation",
    "correlation_top",
    "feature_ranking",
    "outliers",
    "plot",
)

# ruff: noqa: E501
COMMENTS = {
    "TABLE analytics_call_log": "analytics-mcp audit log: one row per tool call, accepted, rejected, failed or timed out. Written through the INSERT-only role rflens_analytics_logger.",
    "COLUMN analytics_call_log.tool": "The analytics-mcp tool called (list_analyses or one of the nine analyses).",
    "COLUMN analytics_call_log.question": "Optional: the user's question, as passed by the caller.",
    "COLUMN analytics_call_log.params": "The call's parameters as received (JSON text, cut at 20,000 characters), before validation.",
    "COLUMN analytics_call_log.analysis_id": "SHA-256 of the analysis, its normalised parameters and the data extract (64 hex characters); also the name of the artifact files. NULL when the call was rejected before the data was read.",
    "COLUMN analytics_call_log.n_runs": "Number of runs in the extract that was given to the sandbox; NULL when none was built.",
    "COLUMN analytics_call_log.status": "ok, rejected (invalid parameters or an analysis the data cannot answer, for example too few values), error (sandbox or internal failure) or timeout.",
    "COLUMN analytics_call_log.reason": "For anything but ok: the code and message. SECOM feature values have no published units; missing values are NULL in secom_measurements.",
}


def _set_vars(**values: str) -> None:
    bind = op.get_bind()
    for name, value in values.items():
        bind.execute(
            sa.text("SELECT set_config(:name, :value, false)"),
            {"name": f"rflens.{name}", "value": value},
        )


def upgrade() -> None:
    tools = ", ".join(f"'{t}'" for t in TOOLS)
    op.execute(
        f"""
        CREATE TABLE analytics_call_log (
            id          bigserial PRIMARY KEY,
            logged_at   timestamptz NOT NULL DEFAULT now(),
            tool        text NOT NULL CHECK (tool IN ({tools})),
            question    text,
            params      text,
            analysis_id text,
            n_runs      integer,
            duration_ms integer,
            status      text NOT NULL CHECK (status IN ('ok', 'rejected', 'error', 'timeout')),
            reason      text
        )
        """
    )
    for target, comment in COMMENTS.items():
        op.execute(f"COMMENT ON {target} IS '{comment.replace(chr(39), chr(39) * 2)}'")

    s = get_analytics_settings()
    _set_vars(
        reader_role=s.analytics_reader_user,
        reader_password=s.analytics_reader_password,
        logger_role=s.analytics_logger_user,
        logger_password=s.analytics_logger_password,
        timeout=f"{s.analytics_statement_timeout_ms}ms",
    )
    op.execute(
        f"""
        DO $$
        DECLARE
            rd text := current_setting('rflens.reader_role');
            lg text := current_setting('rflens.logger_role');
        BEGIN
            IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = rd) THEN
                EXECUTE format(
                    'CREATE ROLE %I LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE PASSWORD %L',
                    rd, current_setting('rflens.reader_password'));
            END IF;
            IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = lg) THEN
                EXECUTE format(
                    'CREATE ROLE %I LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE PASSWORD %L',
                    lg, current_setting('rflens.logger_password'));
            END IF;

            EXECUTE format('GRANT CONNECT ON DATABASE %I TO %I, %I', current_database(), rd, lg);
            EXECUTE format('GRANT USAGE ON SCHEMA public TO %I, %I', rd, lg);

            EXECUTE format('GRANT SELECT ON {READER_TABLES} TO %I', rd);
            EXECUTE format('ALTER ROLE %I SET default_transaction_read_only = on', rd);
            EXECUTE format('ALTER ROLE %I SET statement_timeout = %L', rd,
                           current_setting('rflens.timeout'));

            EXECUTE format('GRANT INSERT ON analytics_call_log TO %I', lg);
            EXECUTE format('GRANT USAGE ON SEQUENCE analytics_call_log_id_seq TO %I', lg);
        END
        $$
        """
    )
    _set_vars(reader_role="", reader_password="", logger_role="", logger_password="", timeout="")


def downgrade() -> None:
    s = get_analytics_settings()
    _set_vars(reader_role=s.analytics_reader_user, logger_role=s.analytics_logger_user)
    op.execute(
        f"""
        DO $$
        DECLARE
            r text;
        BEGIN
            FOREACH r IN ARRAY ARRAY[current_setting('rflens.reader_role'),
                                     current_setting('rflens.logger_role')] LOOP
                IF EXISTS (SELECT FROM pg_roles WHERE rolname = r) THEN
                    EXECUTE format('REVOKE ALL ON {READER_TABLES}, analytics_call_log FROM %I', r);
                    EXECUTE format('REVOKE ALL ON SEQUENCE analytics_call_log_id_seq FROM %I', r);
                    EXECUTE format('REVOKE ALL ON SCHEMA public FROM %I', r);
                    EXECUTE format('REVOKE ALL ON DATABASE %I FROM %I', current_database(), r);
                    EXECUTE format('DROP ROLE %I', r);
                END IF;
            END LOOP;
        END
        $$
        """
    )
    _set_vars(reader_role="", logger_role="")
    op.execute("DROP TABLE analytics_call_log")
