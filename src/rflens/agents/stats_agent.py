"""Stats agent: one plain function (the graph wraps it as a node).

    result = answer_stats(question, run=run, tools=AnalyticsMcpTools(url), question_id="q07")

Flow: one gateway call returns JSON {refuse, refuse_reason, analysis, params_json}; a refusal ends
the question; otherwise the analysis runs through analytics-mcp (which validates the parameters
again and runs fixed code in a sandbox). If analytics-mcp rejects the call, its message goes back to
the model once (so at most 1 + 1 generations). The gateway's per-question cap is the hard limit:
when it stops a question the result is `failed`. Stops that are not about one question (run cap,
budget, loop detector, paid-call block, ledger failure) propagate and stop the command. The agent
never sees the database and never runs anything itself.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Literal

from rflens.agents.analytics_tools import AnalysisOutcome, AnalyticsMcpTools
from rflens.agents.stats_prompt import CORRECTION, STATS_SCHEMA, STATS_SYSTEM
from rflens.analytics import menu
from rflens.llm.base import BadStructuredOutput, CallCapExceeded, Message, ProviderError
from rflens.llm.gateway import LLMRun

Status = Literal["answered", "refused", "failed"]
MAX_CORRECTIONS = 1
MAX_OUTPUT_TOKENS = 400


@dataclass
class StatsResult:
    question_id: str
    question: str
    status: Status
    analysis: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    call_id: int | None = None  # the successful call's row in analytics_call_log
    analysis_id: str | None = None
    result: dict[str, Any] = field(default_factory=dict)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    dataset: dict[str, Any] = field(default_factory=dict)
    n_runs_used: int | None = None
    n_runs_in_extract: int | None = None
    missing_policy: str = ""
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    corrections: int = 0
    calls: int = 0
    cost_usd: float = 0.0
    run_id: str = ""


def answer_stats(
    question: str,
    *,
    run: LLMRun,
    tools: AnalyticsMcpTools,
    question_id: str,
    purpose: str = "stats-agent",
    max_corrections: int = MAX_CORRECTIONS,
    max_output_tokens: int = MAX_OUTPUT_TOKENS,
) -> StatsResult:
    before = (run.guard.per_question[question_id], run.cost_usd)
    result = StatsResult(question_id=question_id, question=question, status="failed")
    result.run_id = run.run_id
    messages = [Message.system(STATS_SYSTEM), Message.user(question)]
    try:
        _converse(result, messages, run, tools, purpose, max_corrections, max_output_tokens)
    finally:
        result.calls = run.guard.per_question[question_id] - before[0]
        result.cost_usd = run.cost_usd - before[1]
    return result


def _converse(
    result: StatsResult,
    messages: list[Message],
    run: LLMRun,
    tools: AnalyticsMcpTools,
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
                json_schema=STATS_SCHEMA,
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

        params: dict[str, Any] = {}
        if not last_problem:
            if answer["refuse"]:
                result.status = "refused"
                result.notes.append(answer["refuse_reason"].strip())
                return
            try:
                params = json.loads(answer["params_json"] or "{}")
            except ValueError:
                last_problem = "params_json is not valid JSON."
            if not last_problem and not isinstance(params, dict):
                last_problem = "params_json must be a JSON object."
        if not last_problem:
            outcome = tools.run_analysis(answer["analysis"], params, result.question)
            if outcome.ok:
                _fill_answered(result, answer["analysis"], outcome)
                return
            result.analysis = answer["analysis"]
            last_problem = outcome.error or "the analysis failed"
            if outcome.error_kind == "failed":  # sandbox failure: not something to correct
                result.notes.append(last_problem)
                return

        if attempt == max_corrections:
            break
        result.corrections += 1
        messages = [
            *messages,
            Message.assistant(raw),
            Message.user(CORRECTION.format(problem=last_problem)),
        ]
    result.notes.append(
        f"gave up after {result.corrections} correction(s); last problem: {last_problem}"
    )


def _check_answer(answer: Any) -> str:
    """'' if the parsed answer is usable, else what to tell the model."""
    if not isinstance(answer, dict) or not {"refuse", "analysis", "params_json"} <= set(answer):
        return "Your answer did not have the required fields."
    if answer["refuse"]:
        if not str(answer.get("refuse_reason", "")).strip():
            return "You set refuse=true but gave no refuse_reason."
        return ""
    if answer["analysis"] not in menu.ANALYSES:
        return f"analysis must be one of {', '.join(menu.ANALYSES)}; got {answer['analysis']!r}."
    return ""


def _fill_answered(result: StatsResult, analysis: str, outcome: AnalysisOutcome) -> None:
    r = outcome.response
    result.status = "answered"
    result.analysis = analysis
    result.params = r["params"]
    result.call_id = r["call_id"]
    result.analysis_id = r["analysis_id"]
    result.result = r["result"]
    result.artifacts = r.get("artifacts") or []
    result.dataset = r.get("dataset") or {}
    result.n_runs_used = r["n_runs_used"]
    result.n_runs_in_extract = r["n_runs_in_extract"]
    result.missing_policy = r.get("missing_policy", "")
    result.warnings = r.get("warnings") or []
