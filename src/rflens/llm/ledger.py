"""The llm_calls ledger: the budget check and one row per call attempt.

Protocol for a paid attempt (fail closed):
1. reserve(): under an advisory lock, sum cost_usd over every row; if the total reaches the budget
   or the total plus this call's worst case would exceed it, write a 'blocked' row and raise
   BudgetExceeded; otherwise insert a 'pending' row holding the worst-case cost.
2. the provider is called;
3. finish(): update the row with the real tokens and cost. If that fails the caller withholds the
   result, and the pending row keeps counting at its worst case, so spend is never under-counted.
If step 1 cannot write, no call is made.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import Engine, text
from sqlalchemy.exc import SQLAlchemyError

from rflens.llm.base import BudgetExceeded, LedgerError, Usage

LOCK_KEY = 7_203_451  # pg_advisory_xact_lock key: serializes budget checks across processes


@dataclass(frozen=True)
class SpendReport:
    total_usd: float
    today_usd: float
    last_calls: list[Any]
    top_purposes: list[Any]


class Ledger:
    def __init__(self, engine: Engine, budget_usd: float, usd_to_inr: float):
        self.engine = engine
        self.budget_usd = budget_usd
        self.usd_to_inr = usd_to_inr

    # --- writes -----------------------------------------------------------------------------------

    def reserve(
        self,
        *,
        run_id: str,
        question_id: str | None,
        attempt: int,
        provider: str,
        model: str,
        service_tier: str | None,
        purpose: str,
        prompt_hash: str,
        est_input_tokens: int,
        worst_case_usd: float,
        paid: bool,
    ) -> int:
        """Insert the pending row and return its id; raise BudgetExceeded for a paid call that
        does not fit; raise LedgerError if the row cannot be written."""
        params = {
            "run": run_id,
            "q": question_id,
            "attempt": attempt,
            "provider": provider,
            "model": model,
            "tier": service_tier,
            "purpose": purpose,
            "hash": prompt_hash,
            "est_in": est_input_tokens,
        }
        try:
            with self.engine.begin() as conn:
                conn.execute(text("SELECT pg_advisory_xact_lock(:k)"), {"k": LOCK_KEY})
                spent = float(
                    conn.execute(
                        text("SELECT COALESCE(SUM(cost_usd), 0) FROM llm_calls")
                    ).scalar_one()
                )
                if paid and (spent >= self.budget_usd or spent + worst_case_usd > self.budget_usd):
                    conn.execute(
                        text(
                            """
                            INSERT INTO llm_calls (run_id, question_id, attempt, provider, model,
                                service_tier, purpose, prompt_hash, status, error)
                            VALUES (:run, :q, :attempt, :provider, :model, :tier, :purpose, :hash,
                                    'blocked', :err)
                            """
                        ),
                        params | {"err": "budget"},
                    )
                    blocked = spent
                else:
                    blocked = None
                    row_id = conn.execute(
                        text(
                            """
                            INSERT INTO llm_calls (run_id, question_id, attempt, provider, model,
                                service_tier, purpose, prompt_hash, input_tokens, cost_usd,
                                cost_inr, status)
                            VALUES (:run, :q, :attempt, :provider, :model, :tier, :purpose, :hash,
                                    :est_in, :usd, :inr, 'pending')
                            RETURNING id
                            """
                        ),
                        params
                        | {
                            "usd": worst_case_usd if paid else 0,
                            "inr": worst_case_usd * self.usd_to_inr if paid else 0,
                        },
                    ).scalar_one()
        except SQLAlchemyError as e:
            raise LedgerError(
                f"the LLM ledger could not be written, so no call was made ({type(e).__name__})"
            ) from e
        if blocked is not None:
            raise BudgetExceeded(
                f"LLM budget reached: ${blocked:.4f} spent of LLM_BUDGET_USD "
                f"${self.budget_usd:.2f}; this call could cost up to ${worst_case_usd:.4f}. "
                "Paid calls are blocked. Raise LLM_BUDGET_USD to continue."
            )
        return int(row_id)

    def finish(
        self,
        row_id: int,
        *,
        status: str,
        usage: Usage | None,
        cost_usd: float,
        latency_ms: int,
        error: str | None = None,
    ) -> None:
        """Replace the pending row's estimates with the real figures."""
        try:
            with self.engine.begin() as conn:
                conn.execute(
                    text(
                        """
                        UPDATE llm_calls
                        SET status = :status, input_tokens = :i, output_tokens = :o,
                            thinking_tokens = :t, cost_usd = :usd, cost_inr = :inr,
                            latency_ms = :ms, error = :err
                        WHERE id = :id
                        """
                    ),
                    {
                        "id": row_id,
                        "status": status,
                        "i": usage.input_tokens if usage else None,
                        "o": usage.output_tokens if usage else None,
                        "t": usage.thinking_tokens if usage else None,
                        "usd": cost_usd,
                        "inr": cost_usd * self.usd_to_inr,
                        "ms": latency_ms,
                        "err": error[:500] if error else None,
                    },
                )
        except SQLAlchemyError as e:
            raise LedgerError(
                f"the LLM ledger update failed, so the result is withheld ({type(e).__name__}); "
                f"row {row_id} stays 'pending' and counts at its worst-case cost"
            ) from e

    # --- reads ------------------------------------------------------------------------------------

    def total_usd(self) -> float:
        with self.engine.connect() as conn:
            return float(
                conn.execute(text("SELECT COALESCE(SUM(cost_usd), 0) FROM llm_calls")).scalar_one()
            )

    def report(self, last: int = 10, top: int = 5) -> SpendReport:
        with self.engine.connect() as conn:
            total, today = conn.execute(
                text(
                    """
                    SELECT COALESCE(SUM(cost_usd), 0),
                           COALESCE(SUM(cost_usd) FILTER (
                               WHERE created_at >= date_trunc('day', now() AT TIME ZONE 'UTC')
                                                   AT TIME ZONE 'UTC'), 0)
                    FROM llm_calls
                    """
                )
            ).one()
            last_calls = conn.execute(
                text(
                    """
                    SELECT created_at, provider, model, purpose, input_tokens, output_tokens,
                           thinking_tokens, cost_usd, latency_ms, status, run_id
                    FROM llm_calls ORDER BY id DESC LIMIT :n
                    """
                ),
                {"n": last},
            ).all()
            purposes = conn.execute(
                text(
                    """
                    SELECT purpose, count(*) AS calls, SUM(cost_usd) AS cost
                    FROM llm_calls GROUP BY purpose
                    HAVING SUM(cost_usd) > 0 ORDER BY SUM(cost_usd) DESC LIMIT :n
                    """
                ),
                {"n": top},
            ).all()
        return SpendReport(float(total), float(today), list(last_calls), list(purposes))


def format_when(ts: datetime) -> str:
    return ts.strftime("%Y-%m-%d %H:%M:%S")
