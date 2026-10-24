"""make llm-spend: what the LLM gateway has spent, from the llm_calls ledger."""

from rflens.config import get_llm_settings
from rflens.db import get_engine
from rflens.llm.ledger import Ledger, format_when


def render(ledger: Ledger) -> str:
    report = ledger.report(last=10, top=5)
    budget, rate = ledger.budget_usd, ledger.usd_to_inr
    remaining = max(budget - report.total_usd, 0)
    lines = [
        f"Budget (LLM_BUDGET_USD)   ${budget:.2f}",
        f"Total spent               ${report.total_usd:.6f}  (INR {report.total_usd * rate:.4f})",
        f"Remaining budget          ${remaining:.6f}  (INR {remaining * rate:.4f})",
        f"Spent today (UTC)         ${report.today_usd:.6f}  (INR {report.today_usd * rate:.4f})",
        "",
        "Last 10 calls (pending rows are counted at their worst-case cost):",
    ]
    if not report.last_calls:
        lines.append("  none")
    for c in report.last_calls:
        lines.append(
            f"  {format_when(c.created_at)}  {c.status:8} {c.provider}/{c.model}  "
            f"{c.purpose}  in={c.input_tokens} out={c.output_tokens} think={c.thinking_tokens}  "
            f"${float(c.cost_usd):.6f}  {c.latency_ms} ms  {c.run_id}"
        )
    lines += ["", "Top 5 purposes by cost:"]
    if not report.top_purposes:
        lines.append("  none")
    for p in report.top_purposes:
        lines.append(f"  {p.purpose:30} {p.calls:4} calls  ${float(p.cost):.6f}")
    return "\n".join(lines)


def main() -> None:
    s = get_llm_settings()
    print(render(Ledger(get_engine(), s.llm_budget_usd, s.llm_usd_to_inr)))


if __name__ == "__main__":
    main()
