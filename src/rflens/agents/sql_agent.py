"""SQL agent: one plain function (week 4 wraps it as a LangGraph node).

    result = answer_question(question, run=run, tools=SqlMcpTools(url), question_id="q07")

Flow: schema context (list_tables + describe_table through sql-mcp, cached per `tools` object) goes
into the system prompt; the gateway returns JSON {refuse, refuse_reason, sql, tables_used,
assumptions}; a refusal ends the question; otherwise the SQL runs through sql-mcp, with the
question text so sql_query_log links SQL to question. If sql-mcp rejects the SQL or Postgres
errors, the message goes back to the model and it answers again, at most `max_corrections` times
(so at most 1 + 2 generations). The gateway's per-question cap is the hard limit: when it stops a
question the result is `failed`. Stops that are not about one question (run cap, budget, loop
detector, paid-call block, ledger failure) propagate: they stop the command.
"""

from dataclasses import dataclass, field
from typing import Any, Literal

import sqlglot
from sqlglot import exp

from rflens.agents.sql_prompt import ANSWER_SCHEMA, build_system_prompt
from rflens.agents.sql_tools import QueryOutcome, SqlMcpTools
from rflens.llm.base import BadStructuredOutput, CallCapExceeded, Message, ProviderError
from rflens.llm.gateway import LLMRun

Status = Literal["answered", "refused", "failed"]
MAX_CORRECTIONS = 2


@dataclass
class AgentResult:
    question_id: str
    question: str
    status: Status
    sql: str | None = None
    columns: list[str] = field(default_factory=list)
    rows: list[list[Any]] = field(default_factory=list)
    row_count: int = 0
    truncated: bool = False
    tables_used: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    corrections: int = 0
    calls: int = 0  # gateway attempts for this question: retries and corrections included
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    cost_usd: float = 0.0
    run_id: str = ""
    plan_cost: float | None = None


def tables_in(sql: str, fallback: list[str]) -> list[str]:
    """Relations the SQL reads (CTE names excluded); the model's own list if the SQL won't parse."""
    try:
        tree = sqlglot.parse_one(sql, read="postgres")
    except sqlglot.errors.SqlglotError:
        return sorted(set(fallback))
    ctes = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}
    names = {t.name.lower() for t in tree.find_all(exp.Table) if t.name} - ctes
    return sorted(names)


def _correction_message(problem: str) -> str:
    return (
        f"{problem}\nFix it and answer again with the JSON object only. "
        "If the request really cannot be answered, set refuse=true."
    )


def answer_question(
    question: str,
    *,
    run: LLMRun,
    tools: SqlMcpTools,
    question_id: str,
    purpose: str = "sql-agent",
    max_corrections: int = MAX_CORRECTIONS,
    max_output_tokens: int = 800,
) -> AgentResult:
    before = (
        run.guard.per_question[question_id],
        run.input_tokens,
        run.output_tokens,
        run.thinking_tokens,
        run.cost_usd,
    )
    result = AgentResult(question_id=question_id, question=question, status="failed")
    result.run_id = run.run_id
    messages = [Message.system(build_system_prompt(tools.schema_context())), Message.user(question)]
    try:
        _converse(result, messages, run, tools, purpose, max_corrections, max_output_tokens)
    finally:
        result.calls = run.guard.per_question[question_id] - before[0]
        result.input_tokens = run.input_tokens - before[1]
        result.output_tokens = run.output_tokens - before[2]
        result.thinking_tokens = run.thinking_tokens - before[3]
        result.cost_usd = run.cost_usd - before[4]
    return result


def _converse(
    result: AgentResult,
    messages: list[Message],
    run: LLMRun,
    tools: SqlMcpTools,
    purpose: str,
    max_corrections: int,
    max_output_tokens: int,
) -> None:
    last_problem = ""
    for attempt in range(max_corrections + 1):
        try:
            generated = run.generate(
                messages,
                purpose=purpose,
                json_schema=ANSWER_SCHEMA,
                question_id=result.question_id,
                max_output_tokens=max_output_tokens,
            )
        except CallCapExceeded as e:
            if e.scope != "question":
                raise  # the run cap stops the whole command
            result.notes.append(f"stopped by the per-question call cap: {e}")
            return
        except ProviderError as e:
            result.notes.append(f"the model call failed: {e}")
            return
        except BadStructuredOutput as e:
            answer, raw = None, e.result.text
            last_problem = f"Your answer was not valid JSON ({e})."
        else:
            answer, raw = generated.parsed, generated.text
            last_problem = _check_answer(answer)

        if not last_problem:
            if answer["refuse"]:
                result.status = "refused"
                result.notes.append(answer["refuse_reason"].strip())
                result.notes += [a for a in answer["assumptions"] if a]
                return
            sql = answer["sql"].strip()
            outcome = tools.run_query(sql, result.question)
            if outcome.ok:
                _fill_answered(result, answer, sql, outcome)
                return
            result.sql = sql
            last_problem = f"The SQL gateway said: {outcome.error}"

        if attempt == max_corrections:
            break
        result.corrections += 1
        messages = [
            *messages,
            Message.assistant(raw),
            Message.user(_correction_message(last_problem)),
        ]
    result.notes.append(
        f"gave up after {result.corrections} correction(s); last problem: {last_problem}"
    )


def _check_answer(answer: Any) -> str:
    """'' if the parsed answer is usable, else what to tell the model."""
    if not isinstance(answer, dict) or not {"refuse", "sql"} <= set(answer):
        return "Your answer did not have the required fields."
    if answer["refuse"]:
        if not str(answer.get("refuse_reason", "")).strip():
            return "You set refuse=true but gave no refuse_reason."
        return ""
    if not str(answer["sql"]).strip():
        return "You set refuse=false but sql is empty."
    return ""


def _fill_answered(result: AgentResult, answer: dict, sql: str, outcome: QueryOutcome) -> None:
    result.status = "answered"
    result.sql = sql
    result.columns, result.rows = outcome.columns, outcome.rows
    result.row_count, result.truncated = outcome.row_count, outcome.truncated
    result.plan_cost = outcome.plan_cost
    result.tables_used = tables_in(sql, answer.get("tables_used", []))
    result.notes += [a for a in answer.get("assumptions", []) if a]
    if outcome.truncated:
        result.notes.append(f"the result was cut at the row limit ({outcome.row_count} rows)")
