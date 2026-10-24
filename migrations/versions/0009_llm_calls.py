"""llm_calls: the ledger of every LLM call attempt (spend tracking and budget)

One row per attempt (a retry is its own row). A paid attempt first inserts a 'pending' row holding
its worst-case cost, so the budget check sees it and a crash or a failed update never loses spend;
the row is updated to 'ok' or 'error' with the real figures. 'blocked' rows record calls refused
by the budget. The gateway sums cost_usd over all rows.

Revision ID: 0009
Revises: 0008
Create Date: 2026-10-24
"""

from alembic import op

revision = "0009"
down_revision = "0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE llm_calls (
            id             bigserial PRIMARY KEY,
            created_at     timestamptz NOT NULL DEFAULT now(),
            run_id         text NOT NULL,
            question_id    text,
            attempt        integer NOT NULL DEFAULT 1,
            provider       text NOT NULL,
            model          text NOT NULL,
            service_tier   text,
            purpose        text NOT NULL,
            prompt_hash    text,
            input_tokens   integer,
            output_tokens  integer,
            thinking_tokens integer,
            cost_usd       numeric(14, 8) NOT NULL DEFAULT 0,
            cost_inr       numeric(14, 6) NOT NULL DEFAULT 0,
            latency_ms     integer,
            status         text NOT NULL
                CHECK (status IN ('pending', 'ok', 'error', 'blocked')),
            error          text
        )
        """
    )
    op.execute("CREATE INDEX llm_calls_created_at ON llm_calls (created_at)")
    op.execute("CREATE INDEX llm_calls_run_id ON llm_calls (run_id)")
    op.execute(
        "COMMENT ON TABLE llm_calls IS 'LLM gateway ledger: one row per call attempt. cost_usd of "
        "a pending row is its worst-case estimate; output_tokens excludes thinking_tokens, both "
        "are billed as output.'"
    )


def downgrade() -> None:
    op.execute("DROP TABLE llm_calls")
